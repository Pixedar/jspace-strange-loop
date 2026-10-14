"""Research v7 (rev 2): a learned state that has to regulate its own information intake — restricted access.

Task: ten animals, ten colours; a stream of K chunks, each with N_UPD update records ("The fox is now red."); chunk 0 also
says which animal to keep track of; some streams later switch it. After every chunk the frozen model is asked, with only
the loop in view, "what color is the animal you are keeping track of now?" (the animal is never named in the question).

Restricted access (DESIGN.md): every record is encoded by the frozen model on its own (no other record, no loop), once.
Key of a record = the model's block-L_r representation of its animal token (it precedes the colour, so it cannot carry
it); payload of a record = a fixed colour read-out of the model's representation of its colour token (a 10-way
classifier fitted once and frozen, so the payload carries the colour only). Per chunk ONE read: an attention over the
chunk's records plus a "nothing new" option, with a query from the previous state ('state'), the self-model's forecast
('fcst'), a learned constant ('const', control) or the right record ('oracle', upper bound). Instructions enter through
their own route: a fixed animal read-out of the model's representation of the instruction.

The loop runs through the frozen model: each chunk, the previous state (4 placeholders), the read colour (1) and the
instruction (1) are written at block L_w into the placeholders of the question prompt; the model answers from them (colour
logits at the last position), and the next state is read from the model's block-L_r representation of the last position.
A self-model S forecasts the next state (own loss); lam x contrastive self-predictability on the system. Objective:
cross-entropy of the right colour at every question. Nothing rewards a particular query or attention pattern.

    python focus_sel.py prep  --out runs/sel05 --model Qwen/Qwen2.5-0.5B-Instruct --local_only 1
    python focus_sel.py train --out runs/sel05 --query state --lam 0 --seed 0
    python focus_sel.py eval  --out runs/sel05 --tag sel-state_l0
"""
import argparse, json, math, pathlib, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

DEV = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'
ANIMALS = ['cat', 'dog', 'fox', 'cow', 'pig', 'owl', 'wolf', 'bear', 'duck', 'goat']
COLORS = ['red', 'blue', 'green', 'black', 'white', 'pink', 'gray', 'brown', 'gold', 'orange']
TEMPLATES = ['The {a} is now {c}.', 'Now the {a} is {c}.', 'The {a} turned {c}.']
INSTR = ['Keep track of the {a}.', 'New instruction: from now on keep track of the {a}.']
PREFIX = 'Notes:'
QUESTION = ' Question: what color is the animal you are keeping track of now? Answer:'
DZ, NZ, RQ = 32, 4, 32
NPH = NZ+2                                                   # state, read colour, instruction


def log(*a):
    print(time.strftime('%F %T'), *a, flush=True)


def sph(x):
    return math.sqrt(DZ)*F.normalize(x, dim=-1)


# ---------------------------------------------------------------------------------------------------------------------
# streams
# ---------------------------------------------------------------------------------------------------------------------
def make_stream(rng, K=6, n_upd=4, p_switch=0.3, p_focus=0.5, n=10, focus=None, records=None):
    """Records per chunk as (animal, colour, template); instruction per chunk as (type, animal) or None; the tracked
    animal and its colour after every chunk; the chunk where a switch happens (-1: none)."""
    col = {};f = int(rng.integers(0, n)) if focus is None else focus
    ks = int(rng.integers(2, K-1)) if (rng.random() < p_switch and K >= 4) else -1;f2 = None
    R, I, foc, gold = [], [], [], []
    for k in range(K):
        if records is not None:S = records[k]
        else:
            must = [f] if k == 0 else []
            if k == ks:f2 = int(rng.choice([x for x in range(n) if x != f]));must = [f2]
            elif k > 0 and rng.random() < p_focus:must = [f2 if (ks >= 0 and k > ks) else f]
            others = [int(x) for x in rng.permutation(n) if x not in must][:n_upd-len(must)]
            S = []
            for an in must+others:
                c = int(rng.integers(0, 10))
                while col.get(an) == c:c = int(rng.integers(0, 10))
                S.append((an, c, int(rng.integers(0, len(TEMPLATES)))))
            S = [S[i] for i in rng.permutation(len(S))]
        for an, c, t in S:col[an] = c
        I.append((0, f) if k == 0 else ((1, f2) if k == ks else None))
        cur = f2 if (ks >= 0 and k >= ks) else f;foc.append(cur);gold.append(col.get(cur, -1));R.append(S)
    return dict(records=R, instr=I, focus=foc, gold=gold, switch=ks)


def tensors(streams, K, n_upd):
    """Batch arrays: record ids (B, K, n_upd) into the 300 encoded sentences, instruction ids (B, K) into the 20 encoded
    instructions (-1: none), focus and gold (B, K), and whether the tracked animal was updated in each chunk."""
    B = len(streams);rid = np.zeros((B, K, n_upd), np.int64);iid = -np.ones((B, K), np.int64)
    foc = np.zeros((B, K), np.int64);gold = np.zeros((B, K), np.int64);upd = np.zeros((B, K), bool);ran = np.zeros((B, K, n_upd), np.int64)
    for b, s in enumerate(streams):
        for k in range(K):
            for j, (an, c, t) in enumerate(s['records'][k]):rid[b, k, j] = (t*10+an)*10+c;ran[b, k, j] = an
            if s['instr'][k] is not None:iid[b, k] = s['instr'][k][0]*10+s['instr'][k][1]
            foc[b, k] = s['focus'][k];gold[b, k] = s['gold'][k];upd[b, k] = any(an == s['focus'][k] for an, _, _ in s['records'][k])
    t = lambda x:torch.tensor(x, device=DEV)
    return dict(rid=t(rid), iid=t(iid), focus=t(foc), gold=t(gold), upd=t(upd), ran=t(ran))


# ---------------------------------------------------------------------------------------------------------------------
# the frozen model
# ---------------------------------------------------------------------------------------------------------------------
class Frozen:
    def __init__(self, name, lw, lr, local_only=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, local_files_only=bool(local_only))
        dt = torch.bfloat16 if DEV == 'cuda' else torch.float32
        try:m = AutoModelForCausalLM.from_pretrained(name, dtype=dt, local_files_only=bool(local_only), attn_implementation='sdpa')
        except TypeError:m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dt, local_files_only=bool(local_only), attn_implementation='sdpa')
        self.model = m.to(DEV).eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.layers = self.model.model.layers;self.d = self.model.config.hidden_size;self.lw, self.lr = lw, lr
        self.W = None;self.H = None
        for i, b in enumerate(self.layers):b.register_forward_hook(self._hook(i))
        self.pre = self.tok.encode(PREFIX, add_special_tokens=False);self.ph = self.tok.encode('...', add_special_tokens=False)[0]
        self.P0 = len(self.pre);self.q_ids = self.tok.encode(QUESTION, add_special_tokens=False)
        cid = [self.tok.encode(' '+c, add_special_tokens=False) for c in COLORS];assert all(len(x) == 1 for x in cid);self.col_ids = [x[0] for x in cid]
        self.prompt = torch.tensor(self.pre+[self.ph]*NPH+self.q_ids, device=DEV)

    def _hook(self, i):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if i == self.lw and self.W is not None:
                h = torch.cat([h[:, :self.P0], h[:, self.P0:self.P0+NPH]+self.W.to(h.dtype), h[:, self.P0+NPH:]], 1)
                out = (h,)+tuple(out[1:]) if isinstance(out, tuple) else h
            if i == self.lr:self.H = h
            return out
        return fn

    def step(self, W):
        """The question prompt with W written into its placeholders: colour logits and the block-L_r last-position state."""
        B = W.shape[0];self.W = W
        try:o = self.model(input_ids=self.prompt[None].expand(B, -1), use_cache=False, logits_to_keep=1)
        finally:self.W = None
        h = self.H[:, -1].float();self.H = None
        return o.logits[:, -1, self.col_ids].float(), h

    @torch.no_grad()
    def encode(self, sentences, positions_of):
        """Block-L_r representations at chosen token positions of separately encoded sentences."""
        out = []
        for s in sentences:
            ids = self.pre+self.tok.encode(' '+s, add_special_tokens=False);self.model(input_ids=torch.tensor([ids], device=DEV), use_cache=False)
            H = self.H[0].float();self.H = None;out.append(torch.stack([H[p] for p in positions_of(s, ids)]))
        return torch.stack(out)

    def word_pos(self, ids, word, last=True):
        w = self.tok.encode(' '+word, add_special_tokens=False)[0];pos = [i for i, x in enumerate(ids) if x == w]
        return pos[-1] if last else pos[0]


def prep(a):
    """Encode the 300 records and 20 instructions once; fit the fixed colour and animal read-outs; save everything."""
    out = pathlib.Path(a.out);out.mkdir(parents=True, exist_ok=True);fm = Frozen(a.model, a.lw, a.lr, a.local_only)
    recs = [(t, an, c) for t in range(len(TEMPLATES)) for an in range(10) for c in range(10)]
    sents = [TEMPLATES[t].format(a=ANIMALS[an], c=COLORS[c]) for t, an, c in recs]
    lab = {s:(an, c) for s, (t, an, c) in zip(sents, recs)}
    E = fm.encode(sents, lambda s, ids:[fm.word_pos(ids, ANIMALS[lab[s][0]]), fm.word_pos(ids, COLORS[lab[s][1]]), len(ids)-1])
    keys, colv = E[:, 0], E[:, 1]
    isent = [INSTR[i].format(a=ANIMALS[an]) for i in range(2) for an in range(10)]
    IE = fm.encode(isent, lambda s, ids:[len(ids)-1])[:, 0]
    # fixed read-outs (fitted once, frozen): colour of a record from its colour token; animal of an instruction
    def fit(X, y, n_cls, steps=300):
        mu, sd = X.mean(0), X.std(0)+1e-4;Xs = ((X-mu)/sd).to(DEV);lin = nn.Linear(X.shape[1], n_cls).to(DEV)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-2);yt = torch.tensor(y, device=DEV)
        for _ in range(steps):
            loss = F.cross_entropy(lin(Xs), yt)+1e-3*lin.weight.pow(2).sum();opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():P = torch.softmax(lin(Xs), -1);acc = float((P.argmax(-1) == yt).float().mean())
        return P.detach().cpu(), acc
    # colour read-out validated on held-out templates: fit on templates 0-1, test on template 2, then refit on all
    tr = [i for i, (t, _, _) in enumerate(recs) if t < 2];te = [i for i, (t, _, _) in enumerate(recs) if t == 2]
    mu, sd = colv[tr].mean(0), colv[tr].std(0)+1e-4;lin = nn.Linear(colv.shape[1], 10).to(DEV);opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for _ in range(300):
        loss = F.cross_entropy(lin(((colv[tr]-mu)/sd).to(DEV)), torch.tensor([recs[i][2] for i in tr], device=DEV));opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():ho = float((lin(((colv[te]-mu)/sd).to(DEV)).argmax(-1).cpu() == torch.tensor([recs[i][2] for i in te])).float().mean())
    CP, cacc = fit(colv, [c for _, _, c in recs], 10);IP, iacc = fit(IE, [an for i in range(2) for an in range(10)], 10)
    torch.save(dict(keys=keys.cpu(), colour=CP, instr=IP, model=a.model, lw=a.lw, lr=a.lr), out/'records.pt')
    json.dump(dict(colour_readout_acc=cacc, colour_readout_heldout_template_acc=ho, instr_readout_acc=iacc, d=fm.d,
                   n_layers=len(fm.layers)), open(out/'prep.json', 'w'), indent=1)
    log('prep: colour read-out', round(cacc, 3), '(held-out template', round(ho, 3), ') | instruction read-out', round(iacc, 3))


# ---------------------------------------------------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------------------------------------------------
class Loop(nn.Module):
    def __init__(self, d, query, keys_dim, norm=False):
        super().__init__();self.query_src, self.d, self.norm = query, d, bool(norm)
        self.register_buffer('mu_h', torch.zeros(d));self.register_buffer('sd_h', torch.ones(d))
        self.register_buffer('mu_k', torch.zeros(keys_dim));self.register_buffer('sd_k', torch.ones(keys_dim))
        self.Wz = nn.Linear(DZ, NZ*d);self.Wc = nn.Linear(10+1, d);self.Wi = nn.Linear(10+1, d)
        self.Rl = nn.Linear(d, DZ);self.S = nn.Sequential(nn.Linear(DZ, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, DZ))
        self.z0 = nn.Parameter(torch.randn(DZ));self.Wk = nn.Linear(keys_dim, RQ, bias=False);self.k_null = nn.Parameter(torch.randn(RQ)*0.1)
        if query == 'const':self.q0 = nn.Parameter(torch.randn(RQ))
        elif query in ('state', 'fcst', 'instr'):self.Wq = nn.Linear(DZ+11, RQ)     # previous state (or forecast, or nothing) + instruction
        self.register_buffer('n_w', torch.tensor(1.0))

    def forecast(self, z):
        return sph(self.S(z))

    def forecast_frozen(self, z):
        return sph(functional_call(self.S, {n:p.detach() for n, p in self.S.named_parameters()}, (z,)))

    def query(self, z_prev, instr):
        """The read's query: from the previous state (or the self-model's forecast) and the current instruction, if any
        (instructions reach the selection before it happens; in chunks without one only the state can steer it)."""
        if self.query_src == 'const':return self.q0[None].expand(z_prev.shape[0], RQ)
        if self.query_src == 'instr':return self.Wq(torch.cat([torch.zeros_like(z_prev), instr], -1))   # matched control: no persistent state
        return self.Wq(torch.cat([self.forecast_frozen(z_prev) if self.query_src == 'fcst' else z_prev, instr], -1))

    def select(self, q, K_):
        """Attention over the chunk's records (B, n_upd, RQ keys) and the 'nothing new' option."""
        if self.norm:K_ = (K_-self.mu_k)/self.sd_k
        s = torch.cat([(self.Wk(K_)@q[:, :, None]).squeeze(-1), (self.k_null[None]*q).sum(-1, keepdim=True)], -1)/math.sqrt(RQ)
        return torch.softmax(s, -1)

    def write(self, z, colour, instr):
        w = torch.cat([self.Wz(z).view(-1, NZ, self.d), self.Wc(colour)[:, None], self.Wi(instr)[:, None]], 1)
        return self.n_w*w/w.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def nce(pred, tgt, tau=0.1):
    p = F.normalize(pred, dim=-1);t = F.normalize(tgt, dim=-1)
    return F.cross_entropy(p@t.T/tau, torch.arange(len(p), device=p.device))


def episode(fm, loop, rec, T, lam=0., oracle=False, q_fn=None, z_fn=None, keep=False):
    """One pass over the K chunks. T: batch tensors. Returns colour logits (B, K, 10), states, attentions, queries,
    the self-model's loss and the pressure loss."""
    B, K, n_upd = T['rid'].shape;z = sph(loop.z0.expand(B, DZ));keys = rec['keys'][T['rid']]       # (B, K, n_upd, d)
    col = rec['colour'][T['rid']]                                                                 # (B, K, n_upd, 10)
    LG, Z, A, Q, LS, LP = [], [], [], [], [], [];z_prev = z
    for k in range(K):
        zin = z_prev if z_fn is None else z_fn(k, z_prev)
        has = (T['iid'][:, k] >= 0).float()[:, None]
        instr = torch.cat([rec['instr'][T['iid'][:, k].clamp_min(0)]*has, has], -1)
        if oracle:
            tgt = (T['ran'][:, k] == T['focus'][:, k][:, None]).float();q = torch.zeros(B, RQ, device=DEV)
            att = torch.cat([tgt, (tgt.sum(-1, keepdim=True) == 0).float()], -1)
        else:
            q = loop.query(zin, instr)
            if q_fn is not None:q = q_fn(k, q)
            att = loop.select(q, keys[:, k])
        colour = torch.cat([(att[:, :-1, None]*col[:, k]).sum(1), att[:, -1:]], -1)               # read colour + 'nothing new'
        lg, h = fm.step(loop.write(zin, colour, instr))
        z = sph(loop.Rl((h-loop.mu_h)/loop.sd_h if loop.norm else h));LG.append(lg)
        if k > 0:
            LS.append(nce(loop.forecast(z_prev.detach()), z.detach()))
            if lam > 0:LP.append(nce(loop.forecast_frozen(z_prev), z))
        Z.append(z)
        if keep:A.append(att.detach());Q.append(q.detach())
        z_prev = z
    return dict(logits=torch.stack(LG, 1), z=torch.stack(Z, 1), att=A, q=Q, ls=torch.stack(LS).mean() if LS else torch.zeros((), device=DEV),
                lp=torch.stack(LP).mean() if LP else None)


def load_rec(out):
    r = torch.load(out/'records.pt', map_location='cpu');return {k:(v.to(DEV) if torch.is_tensor(v) else v) for k, v in r.items()}


def tag_of(a):
    return f'sel-{a.query}_l{a.lam:g}'+(f'_s{a.seed}' if a.seed else '')


def train(a):
    out = pathlib.Path(a.out);rec = load_rec(out);fm = Frozen(rec['model'], rec['lw'], rec['lr'], a.local_only);tag = tag_of(a)
    torch.manual_seed(a.seed);loop = Loop(fm.d, a.query, rec['keys'].shape[-1], a.norm).to(DEV)
    with torch.no_grad():                                    # write strength: the typical size of the model's states at block L_w
        o = fm.model(input_ids=fm.prompt[None], use_cache=False, output_hidden_states=True);fm.H = None
        loop.n_w.fill_(float(o.hidden_states[fm.lw+1][0, fm.P0+NPH:].float().norm(dim=-1).median()))
        if a.norm:                                           # per-dimension statistics: record keys; read state under random writes
            kk = rec['keys'].float();loop.mu_k.copy_(kk.mean(0));loop.sd_k.copy_(kk.std(0)+1e-3*kk.std(0).mean())
            Hs = []
            for _ in range(4):
                zr = sph(torch.randn(64, DZ, device=DEV));cr = torch.softmax(torch.randn(64, 11, device=DEV), -1)
                _, h = fm.step(loop.write(zr, cr, torch.cat([torch.softmax(torch.randn(64, 10, device=DEV), -1), torch.ones(64, 1, device=DEV)], -1)))
                Hs.append(h)
            Hs = torch.cat(Hs);loop.mu_h.copy_(Hs.mean(0));loop.sd_h.copy_(Hs.std(0)+1e-3*Hs.std(0).mean())
    opt = torch.optim.AdamW([p for p in loop.parameters() if p.requires_grad], lr=a.lrate, weight_decay=0.)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s:min(1., (s+1)/50)*(0.1+0.9*0.5*(1+math.cos(math.pi*min(1., s/a.steps)))))
    rng = np.random.default_rng(1000+a.seed);vrng = np.random.default_rng(777);hist = [];t0 = time.time()
    VT = tensors([make_stream(vrng, a.K, a.n_upd, a.p_switch, a.p_focus) for _ in range(256)], a.K, a.n_upd)
    for s in range(a.steps):
        T = tensors([make_stream(rng, a.K, a.n_upd, a.p_switch, a.p_focus) for _ in range(a.bs)], a.K, a.n_upd)
        R = episode(fm, loop, rec, T, lam=a.lam, oracle=a.query == 'oracle')
        la = F.cross_entropy(R['logits'].reshape(-1, 10), T['gold'].reshape(-1))
        loss = la+R['ls']+(a.lam*R['lp'] if R['lp'] is not None else 0.)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(loop.parameters(), 1.0);opt.step();sched.step()
        rec_ = dict(step=s, ans=float(la), ls=float(R['ls']), t=time.time()-t0)
        if s % 100 == 0 or s == a.steps-1:
            with torch.no_grad():
                Rv = episode(fm, loop, rec, VT, oracle=a.query == 'oracle');ok = (Rv['logits'].argmax(-1) == VT['gold']).float()
            rec_['val_acc'] = float(ok.mean());rec_['val_updated'] = float(ok[VT['upd']].mean());rec_['val_kept'] = float(ok[~VT['upd']].mean())
            log(tag, f"step {s} answer {la:.3f} | val acc {rec_['val_acc']:.3f} (updated {rec_['val_updated']:.3f}, kept {rec_['val_kept']:.3f}) "
                     f"by chunk {np.round(ok.mean(0).tolist(), 2).tolist()} | ls {rec_['ls']:.3f} | {rec_['t']:.0f}s")
        hist.append(rec_)
    (out/'ckpt').mkdir(exist_ok=True);torch.save(loop.state_dict(), out/'ckpt'/f'{tag}.pt')
    json.dump(dict(cfg=vars(a), tag=tag, hist=hist), open(out/'ckpt'/f'{tag}.json', 'w'));log(tag, 'trained', f'{time.time()-t0:.0f}s')


@torch.no_grad()
def evaluate(a):
    out = pathlib.Path(a.out);rec = load_rec(out);cfg = json.load(open(out/'ckpt'/f'{a.tag}.json'))['cfg']
    fm = Frozen(rec['model'], rec['lw'], rec['lr'], a.local_only);loop = Loop(fm.d, cfg['query'], rec['keys'].shape[-1], cfg.get('norm', 0)).to(DEV)
    missing, unexpected = loop.load_state_dict(torch.load(out/'ckpt'/f'{a.tag}.pt', map_location=DEV), strict=False);loop.eval()
    assert set(missing) <= {'mu_h', 'sd_h', 'mu_k', 'sd_k'} and not unexpected, (missing, unexpected)   # older checkpoints: no statistics
    orc = cfg['query'] == 'oracle';K, nu = cfg['K'], cfg['n_upd']
    rng = np.random.default_rng(4242);S = [make_stream(rng, K, nu, cfg['p_switch'], cfg['p_focus']) for _ in range(a.n_eval)];T = tensors(S, K, nu)
    def run(T_, **kw):
        Rs = [];B = T_['rid'].shape[0]
        for i in range(0, B, a.bs):
            Rs.append(episode(fm, loop, rec, {k:v[i:i+a.bs] for k, v in T_.items()}, oracle=orc, keep=True, **kw))
        return dict(logits=torch.cat([r['logits'] for r in Rs]), z=torch.cat([r['z'] for r in Rs]),
                    att=[torch.cat([r['att'][k] for r in Rs]) for k in range(K)], q=[torch.cat([r['q'][k] for r in Rs]) for k in range(K)])
    R0 = run(T);ok = (R0['logits'].argmax(-1) == T['gold']).float();sw = torch.tensor([s['switch'] for s in S], device=DEV)
    after = torch.zeros_like(ok, dtype=torch.bool)
    for b, s in enumerate(S):
        if s['switch'] >= 0:after[b, s['switch']] = True
    res = dict(tag=a.tag, cfg=cfg, acc=float(ok.mean()), acc_by_chunk=ok.mean(0).tolist(), acc_updated=float(ok[T['upd']].mean()),
               acc_kept=float(ok[~T['upd']].mean()), acc_at_switch=float(ok[after].mean()) if after.any() else None)
    # selection: attention on the tracked animal's record (when present) and on 'nothing new' (when absent)
    att = torch.stack(R0['att'], 1);isf = (T['ran'] == T['focus'][:, :, None]).float()
    res['selection'] = dict(on_tracked_record_when_present=float((att[:, :, :-1]*isf).sum(-1)[T['upd']].mean()),
                            on_nothing_new_when_absent=float(att[:, :, -1][~T['upd']].mean()),
                            on_nothing_new_when_present=float(att[:, :, -1][T['upd']].mean()))
    # no memory: the initial state in every chunk
    Rn = run(T, z_fn=lambda k, z:sph(loop.z0.expand(z.shape[0], DZ)));res['no_memory_acc_by_chunk'] = (Rn['logits'].argmax(-1) == T['gold']).float().mean(0).tolist()
    # query-only intervention: twins (same records, tracked animal A vs B, no switch), both updated with different colours at chunk t
    t = a.at;rng2 = np.random.default_rng(99);A_, B_ = [], []
    while len(A_) < a.n_pairs:
        s = make_stream(rng2, K, nu, 0., cfg['p_focus'])
        fa = s['focus'][0];cand = [an for an, c, tt in s['records'][t] if an != fa]
        if fa not in [an for an, c, tt in s['records'][t]] or not cand:continue
        fb = int(rng2.choice(cand));tw = make_stream(np.random.default_rng(0), K, nu, 0., 0., focus=fb, records=s['records'])
        if s['gold'][t] == tw['gold'][t] or tw['gold'][0] < 0:continue
        A_.append(s);B_.append(tw)
    TA, TB = tensors(A_, K, nu), tensors(B_, K, nu);RA, RB = run(TA), run(TB)
    def swap_q(src, ks):
        st = dict(i=0)
        def fn(k, q):
            b = slice(st['i'], st['i']+q.shape[0])
            if k == K-1:st['i'] += q.shape[0]
            return src[k][b] if k in ks else q
        return fn
    def outcome(R):
        p = R['logits'][:, t].argmax(-1)
        return dict(answers_A=float((p == TA['gold'][:, t]).float().mean()), answers_B=float((p == TB['gold'][:, t]).float().mean()),
                    next_answers_A=float((R['logits'][:, t+1].argmax(-1) == TA['gold'][:, t+1]).float().mean()) if t+1 < K else None)
    res['query_swap'] = dict(at_chunk=t, own=outcome(RA))
    if not orc and cfg['query'] != 'const':
        res['query_swap']['sham_identical_query'] = outcome(run(TA, q_fn=swap_q(RA['q'], {t})))
        res['query_swap']['query_from_B'] = outcome(run(TA, q_fn=swap_q(RB['q'], {t})))
        # counterfactual replay: B's queries for chunks 1..t, then the closed loop again
        RR = run(TA, q_fn=swap_q(RB['q'], set(range(1, t+1))));pr = RR['logits'].argmax(-1)
        res['replay'] = dict(chunks_1_to_t_follow_B=float((pr[:, 1:t+1] == TB['gold'][:, 1:t+1]).float().mean()),
                             chunks_1_to_t_follow_A=float((pr[:, 1:t+1] == TA['gold'][:, 1:t+1]).float().mean()),
                             after_restoring_follow_A=float((pr[:, t+1:] == TA['gold'][:, t+1:]).float()[TA['upd'][:, t+1:]].mean()) if t+1 < K else None)
    # code edits: change only the 'which animal' part, or only the 'which colour' part, of the input state at chunk t.
    # Directions from class means of the states (normalised): by tracked animal averaged over colours, by colour averaged
    # over animals.
    Zu = F.normalize(R0['z'], dim=-1).reshape(-1, DZ);fo = T['focus'].reshape(-1);go = T['gold'].reshape(-1)
    cell = torch.zeros(10, 10, DZ, device=DEV);cnt = torch.zeros(10, 10, device=DEV)
    cell.index_put_((fo, go), Zu, accumulate=True);cnt.index_put_((fo, go), torch.ones_like(fo, dtype=torch.float), accumulate=True)
    ok_ = cnt > 0;cm = cell/cnt.clamp_min(1)[..., None]
    m_f = torch.stack([cm[f][ok_[f]].mean(0) for f in range(10)]);m_c = torch.stack([cm[:, c][ok_[:, c]].mean(0) for c in range(10)])
    def edit_fn(delta):
        st_ = dict(i=0)
        def fn(k, z):
            b = slice(st_['i'], st_['i']+z.shape[0])
            if k == K-1:st_['i'] += z.shape[0]
            return sph(F.normalize(z, dim=-1)+delta[b]) if k == t else z
        return fn
    if not orc:
        fa, fb = TA['focus'][:, 0], TB['focus'][:, 0]
        res['focus_edit'] = dict(at_chunk=t, own=outcome(RA), edited=outcome(run(TA, z_fn=edit_fn(m_f[fb]-m_f[fa]))))
        # colour edit where the tracked animal is NOT updated at chunk t (the colour must come from memory)
        keep_ = [i for i, s in enumerate(S) if s['switch'] < 0 and not T['upd'][i, t] and T['gold'][i, t] >= 0]
        if keep_:
            TK = {k_:v[keep_] for k_, v in T.items()};c_old = TK['gold'][:, t];c_new = (c_old+1+torch.randint(0, 9, c_old.shape, device=DEV)) % 10
            RK0, RK = run(TK), run(TK, z_fn=edit_fn(m_c[c_new]-m_c[c_old]))
            pk0, pk = RK0['logits'][:, t].argmax(-1), RK['logits'][:, t].argmax(-1)
            res['colour_edit'] = dict(at_chunk=t, n=len(keep_), own_answers_old=float((pk0 == c_old).float().mean()),
                                      edited_answers_old=float((pk == c_old).float().mean()), edited_answers_new=float((pk == c_new).float().mean()))
    np.savez_compressed(out/f'eval_{a.tag}.npz', z=R0['z'].cpu().numpy(), q=torch.stack(R0['q'], 1).cpu().numpy(), att=att.cpu().numpy(),
                        focus=T['focus'].cpu().numpy(), gold=T['gold'].cpu().numpy(), upd=T['upd'].cpu().numpy(), switch=sw.cpu().numpy(), ok=ok.cpu().numpy())
    json.dump(res, open(out/f'eval_{a.tag}.json', 'w'), indent=1)
    log(a.tag, json.dumps({k:v for k, v in res.items() if k != 'cfg'}))


@torch.no_grad()
def edits(a):
    """Controls for the code edits. Directions are class means of the states over separate
    calibration streams (focus: by tracked animal averaged over colours; colour: by colour averaged over animals). On
    twin streams (tracked animal A or B, both updated with different colours at chunk t) the focus edit A -> B is applied
    (a) to the state as a whole, (b) only to the query (the answer prompt gets the original state), (c) only to the answer
    prompt (the query comes from the original state), (d) with strengths -1, 0.5, 1, 2, (e) as random directions of the
    same size, (f) toward another animal present in the chunk, (g) toward an animal absent from the chunk (the read should
    then choose 'nothing new' and the stored colour should stay); each at chunks 2, 3 and 4. Measured: answers (A's, B's,
    C's colour; A's stored colour), attention at chunk t on A's record, B's record, C's record and 'nothing new', the
    displacement of the state (1 - cos), and at the next chunk agreement with A and with B. The colour edit: on streams
    where the tracked animal is not updated at t, the remembered colour, and at the next update of the animal whether it
    is acquired (focus kept)."""
    out = pathlib.Path(a.out);rec = load_rec(out);cfg = json.load(open(out/'ckpt'/f'{a.tag}.json'))['cfg']
    fm = Frozen(rec['model'], rec['lw'], rec['lr'], a.local_only);loop = Loop(fm.d, cfg['query'], rec['keys'].shape[-1], cfg.get('norm', 0)).to(DEV)
    missing, unexpected = loop.load_state_dict(torch.load(out/'ckpt'/f'{a.tag}.pt', map_location=DEV), strict=False);loop.eval()
    assert set(missing) <= {'mu_h', 'sd_h', 'mu_k', 'sd_k'} and not unexpected, (missing, unexpected)
    K, nu = cfg['K'], cfg['n_upd'];bs = a.bs;res = dict(tag=a.tag, cfg=cfg)
    def run(T_, **kw):
        Rs = []
        for i in range(0, T_['rid'].shape[0], bs):
            Rs.append(episode(fm, loop, rec, {k:v[i:i+bs] for k, v in T_.items()}, keep=True, **kw))
        return dict(logits=torch.cat([r['logits'] for r in Rs]), z=torch.cat([r['z'] for r in Rs]),
                    att=[torch.cat([r['att'][k] for r in Rs]) for k in range(K)], q=[torch.cat([r['q'][k] for r in Rs]) for k in range(K)])
    # calibration directions from separate streams
    crng = np.random.default_rng(777);CS = [make_stream(crng, K, nu, cfg['p_switch'], cfg['p_focus']) for _ in range(a.n_eval)];CT = tensors(CS, K, nu)
    RC = run(CT);Zu = F.normalize(RC['z'], dim=-1).reshape(-1, DZ);fo = CT['focus'].reshape(-1);go = CT['gold'].reshape(-1)
    cell = torch.zeros(10, 10, DZ, device=DEV);cnt = torch.zeros(10, 10, device=DEV)
    cell.index_put_((fo, go), Zu, accumulate=True);cnt.index_put_((fo, go), torch.ones_like(fo, dtype=torch.float), accumulate=True)
    ok_ = cnt > 0;cm = cell/cnt.clamp_min(1)[..., None]
    m_f = torch.stack([cm[f][ok_[f]].mean(0) for f in range(10)]);m_c = torch.stack([cm[:, c][ok_[:, c]].mean(0) for c in range(10)])
    def per_batch(full, k_at, fn_at):
        st_ = dict(i=0)
        def fn(k, x):
            b = slice(st_['i'], st_['i']+x.shape[0])
            if k == K-1:st_['i'] += x.shape[0]
            return fn_at(x, b) if k == k_at else x
        return fn
    res['by_chunk'] = {}
    for t in (2, 3, 4):
        rng2 = np.random.default_rng(99+t);A_, B_, C_ = [], [], []
        while len(A_) < a.n_pairs:
            s_ = make_stream(rng2, K, nu, 0., cfg['p_focus']);fa = s_['focus'][0];inT = [an for an, c, tt in s_['records'][t]]
            if fa not in inT:continue
            cand = [an for an in inT if an != fa]
            if len(cand) < 2:continue
            fb = int(rng2.choice(cand));fc = int(rng2.choice([x for x in cand if x != fb]))
            absent = [x for x in range(10) if x not in inT]
            tw = make_stream(np.random.default_rng(0), K, nu, 0., 0., focus=fb, records=s_['records'])
            if s_['gold'][t] == tw['gold'][t] or tw['gold'][0] < 0:continue
            A_.append(s_);B_.append(tw);C_.append((fc, int(rng2.choice(absent))))
        TA, TB = tensors(A_, K, nu), tensors(B_, K, nu);RA = run(TA)
        fa = TA['focus'][:, 0];fb = TB['focus'][:, 0];fc = torch.tensor([c for c, _ in C_], device=DEV);fd = torch.tensor([d for _, d in C_], device=DEV)
        colC = torch.stack([torch.tensor([dict((an, c) for an, c, tt in s_['records'][t])[c_] for s_, (c_, _) in zip(A_, C_)], device=DEV)])[0]
        rec_pos = lambda who:(TA['ran'][:, t] == who[:, None]).float()               # (B, n_upd) which record belongs to whom
        z_prev = RA['z'][:, t-1];zu = F.normalize(z_prev, dim=-1);instr0 = torch.cat([torch.zeros(len(A_), 10, device=DEV), torch.zeros(len(A_), 1, device=DEV)], -1)
        def outcome(R, d_vec=None):
            p = R['logits'][:, t].argmax(-1);at = R['att'][t]
            o = dict(answers_A=float((p == TA['gold'][:, t]).float().mean()), answers_B=float((p == TB['gold'][:, t]).float().mean()),
                     answers_C=float((p == colC).float().mean()), answers_A_stored=float((p == TA['gold'][:, t-1]).float().mean()),
                     att_A=float((at[:, :-1]*rec_pos(fa)).sum(-1).mean()), att_B=float((at[:, :-1]*rec_pos(fb)).sum(-1).mean()),
                     att_C=float((at[:, :-1]*rec_pos(fc)).sum(-1).mean()), att_nothing_new=float(at[:, -1].mean()))
            if t+1 < K:
                pn = R['logits'][:, t+1].argmax(-1);o['next_answers_A'] = float((pn == TA['gold'][:, t+1]).float().mean())
                o['next_answers_B'] = float((pn == TB['gold'][:, t+1]).float().mean())
            if d_vec is not None:o['displacement_1_minus_cos'] = float((1-(F.normalize(zu+d_vec, dim=-1)*zu).sum(-1)).mean())
            return o
        def zedit(d_vec):
            return per_batch(None, t, lambda x, b:sph(F.normalize(x, dim=-1)+d_vec[b]))
        r_ = dict(own=outcome(RA))
        dAB = m_f[fb]-m_f[fa]
        for alpha in (-1., 0.5, 1., 2.):
            r_[f'focus_edit_x{alpha:g}'] = outcome(run(TA, z_fn=zedit(alpha*dAB)), alpha*dAB)
        if cfg['query'] not in ('const', 'oracle'):
            qe = loop.query(sph(zu+dAB), instr0);q0 = RA['q'][t]
            r_['focus_edit_query_only'] = outcome(run(TA, q_fn=per_batch(None, t, lambda x, b:qe[b])))
            r_['focus_edit_answer_only'] = outcome(run(TA, z_fn=zedit(dAB), q_fn=per_batch(None, t, lambda x, b:q0[b])))
        g = torch.Generator(device='cpu');g.manual_seed(5+t);rnd = F.normalize(torch.randn(len(A_), DZ, generator=g).to(DEV), dim=-1)*dAB.norm(dim=-1, keepdim=True)
        r_['random_edit_same_size'] = outcome(run(TA, z_fn=zedit(rnd)), rnd)
        dAC = m_f[fc]-m_f[fa];r_['focus_edit_toward_present_C'] = outcome(run(TA, z_fn=zedit(dAC)), dAC)
        dAD = m_f[fd]-m_f[fa];r_['focus_edit_toward_absent_D'] = outcome(run(TA, z_fn=zedit(dAD)), dAD)
        res['by_chunk'][t] = r_
        log(a.tag, 'edits at chunk', t, json.dumps({k:{kk:round(vv, 3) for kk, vv in v.items()} for k, v in r_.items()}))
    # colour edit on held-out streams: the tracked animal not updated at t (memory), then updated at t+1 (is the focus kept?)
    t = 3;rng3 = np.random.default_rng(4343);KS_ = []
    while len(KS_) < a.n_pairs:
        s_ = make_stream(rng3, K, nu, 0., cfg['p_focus']);f_ = s_['focus'][0]
        if f_ in [an for an, c, tt in s_['records'][t]] or f_ not in [an for an, c, tt in s_['records'][t+1]]:continue
        KS_.append(s_)
    TK = tensors(KS_, K, nu);c_old = TK['gold'][:, t];c_new = (c_old+1+torch.randint(0, 9, c_old.shape, device=DEV)) % 10
    RK0, RK = run(TK), run(TK, z_fn=zedit(m_c[c_new]-m_c[c_old]))
    p0, p1 = RK0['logits'][:, t].argmax(-1), RK['logits'][:, t].argmax(-1);n1 = RK['logits'][:, t+1].argmax(-1)
    res['colour_edit'] = dict(at_chunk=t, own_answers_old=float((p0 == c_old).float().mean()), edited_answers_old=float((p1 == c_old).float().mean()),
                              edited_answers_new=float((p1 == c_new).float().mean()),
                              next_chunk_acquires_tracked_update=float((n1 == TK['gold'][:, t+1]).float().mean()),
                              own_next_chunk_acquires=float((RK0['logits'][:, t+1].argmax(-1) == TK['gold'][:, t+1]).float().mean()))
    rg = torch.Generator(device='cpu');rg.manual_seed(11);dc = m_c[c_new]-m_c[c_old]
    rr = F.normalize(torch.randn(len(KS_), DZ, generator=rg).to(DEV), dim=-1)*dc.norm(dim=-1, keepdim=True)
    pr = run(TK, z_fn=zedit(rr))['logits'][:, t].argmax(-1)
    res['colour_edit']['random_same_size_answers_old'] = float((pr == c_old).float().mean());res['colour_edit']['random_same_size_answers_new'] = float((pr == c_new).float().mean())
    json.dump(res, open(out/f'edits_{a.tag}.json', 'w'), indent=1);log(a.tag, 'colour edit', json.dumps(res['colour_edit']))


def main():
    p = argparse.ArgumentParser();p.add_argument('stage', choices=['prep', 'train', 'eval', 'edits']);p.add_argument('--out', required=True)
    p.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct');p.add_argument('--local_only', type=int, default=0)
    p.add_argument('--lw', type=int, default=8);p.add_argument('--lr', type=int, default=16)
    p.add_argument('--query', default='state', choices=['state', 'fcst', 'const', 'instr', 'oracle']);p.add_argument('--lam', type=float, default=0.)
    p.add_argument('--steps', type=int, default=1500);p.add_argument('--bs', type=int, default=64);p.add_argument('--lrate', type=float, default=2e-3)
    p.add_argument('--seed', type=int, default=0);p.add_argument('--tag', default=None)
    p.add_argument('--K', type=int, default=6);p.add_argument('--n_upd', type=int, default=4);p.add_argument('--p_switch', type=float, default=0.3)
    p.add_argument('--p_focus', type=float, default=0.5);p.add_argument('--n_eval', type=int, default=512);p.add_argument('--n_pairs', type=int, default=256)
    p.add_argument('--at', type=int, default=3);p.add_argument('--norm', type=int, default=0)
    a = p.parse_args();{'prep':prep, 'train':train, 'eval':evaluate, 'edits':edits}[a.stage](a)


if __name__ == '__main__':
    main()
