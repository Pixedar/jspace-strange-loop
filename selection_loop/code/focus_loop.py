"""Research v7: a self-representation that has to regulate its own information selection (see DESIGN.md).

A frozen language model reads a stream of K short chunks of colour updates about 10 animals, one chunk at a time. The
first chunk says which animal to keep track of; that instruction is never shown again; some streams later switch the
animal. After every chunk the model is asked, with only the loop in view, what colour the tracked animal is now. The only
link between chunks is the loop: a 32-number state read from the model's processing of a chunk and written into
placeholder positions of the next chunk (and of the question), plus a small self-model forecasting the next state.

Readers (how the next state is built from the chunk):
  attn   an attention pool over the chunk's positions and the placeholders (the loop's own previous content); its query
         comes from the previous state ('state'), the self-model's forecast ('fcst'), a learned constant ('const',
         control) or a fixed random projection of the previous state ('rand', control). The pooled vector is the only
         route into the next state, so whatever the state takes in, it takes in by selecting.
  last   the last token's state: selection is left to the frozen model's own attention, steered only by what the loop
         wrote ('--heads state' additionally lets the state set gains on the model's attention heads).
Objective: cross-entropy of the right colour at every question (10-way over the colour tokens), the self-model's own
loss, and lam x the system's contrastive self-predictability. Nothing rewards a particular query or attention pattern.

    python focus_loop.py train --out runs/v7 --model Qwen/Qwen3-1.7B --reader attn --query state --lam 0.05
    python focus_loop.py eval  --out runs/v7 --model Qwen/Qwen3-1.7B --tag attn-state_l0.05
"""
import argparse, json, math, os, pathlib, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ANIMALS = ['cat', 'dog', 'fox', 'cow', 'pig', 'owl', 'wolf', 'bear', 'duck', 'goat']        # as in focus_sel.py
COLORS = ['red', 'blue', 'green', 'black', 'white', 'pink', 'gray', 'brown', 'gold', 'orange']
TEMPLATES = ['The {a} is now {c}.', 'Now the {a} is {c}.', 'The {a} turned {c}.']
QUESTION = ' Question: what color is the animal you are keeping track of now? Answer:'
PREFIX = 'Notes:'
DZ, NZ, NF, RQ = 32, 4, 2, 64
NPH = NZ+NF


def log(*a):
    print(time.strftime('%F %T'), *a, flush=True)


def sph(x):
    return math.sqrt(DZ)*F.normalize(x, dim=-1)


# ---------------------------------------------------------------------------------------------------------------------
# the task
# ---------------------------------------------------------------------------------------------------------------------
def make_stream(rng, n_an, n_col, K, n_upd, p_switch, p_focus, focus=None, chunk0=None):
    """One stream: K chunks of update sentences; returns the chunk texts, per chunk the sentence list [(animal, colour,
    text)], the focus after each chunk and the tracked animal's colour after each chunk."""
    col = rng.integers(0, n_col, n_an);f = int(rng.integers(0, n_an)) if focus is None else focus
    ks = int(rng.integers(2, K-1)) if (rng.random() < p_switch and K >= 4) else -1
    chunks, sents, focus_k, gold_k, f2 = [], [], [], [], None
    for k in range(K):
        S = []
        if k == 0 and chunk0 is not None:S = [tuple(x) for x in chunk0]
        else:
            must = [f] if k == 0 else []
            if k == ks:
                f2 = int(rng.choice([x for x in range(n_an) if x != f]));must = [f2]
            elif rng.random() < p_focus and k > 0:must = [f2 if (ks >= 0 and k > ks) else f]
            others = [x for x in rng.permutation(n_an) if x not in must][:n_upd-len(must)]
            for a in list(must)+others:
                c = int(rng.integers(0, n_col))
                while c == col[a]:c = int(rng.integers(0, n_col))
                S.append((int(a), c, int(rng.integers(0, len(TEMPLATES)))))
            order = rng.permutation(len(S));S = [S[i] for i in order]
        for a, c, _ in S:col[a] = c
        cur = (f2 if (ks >= 0 and k >= ks) else f)
        head = (f'Keep track of the {ANIMALS[f]}.' if k == 0 else '')+(f' New instruction: from now on keep track of the {ANIMALS[f2]}.' if k == ks else '')
        chunks.append((head, S));focus_k.append(cur);gold_k.append(int(col[cur]))
    return dict(chunks=chunks, focus=focus_k, gold=gold_k, switch=ks)


def render(tok, head, S):
    """Token ids of one chunk (leading space), and per sentence its animal, colour and token span (relative)."""
    ids, spans = [], []
    if head:ids += tok.encode(' '+head.strip(), add_special_tokens=False)
    for a, c, t in S:
        s = tok.encode(' '+TEMPLATES[t].format(a=ANIMALS[a], c=COLORS[c]), add_special_tokens=False)
        spans.append((a, c, len(ids), len(ids)+len(s)));ids += s
    return ids, spans


# ---------------------------------------------------------------------------------------------------------------------
# the frozen model with read / write / stop hooks and optional head gains
# ---------------------------------------------------------------------------------------------------------------------
class Stop(Exception):
    pass


class Frozen:
    def __init__(self, name, lw, lr, local_only=False, dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name, local_files_only=local_only)
        try:self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, local_files_only=local_only, attn_implementation='sdpa')
        except TypeError:self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, local_files_only=local_only, attn_implementation='sdpa')
        self.model.to(DEV).eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.layers = self.model.model.layers;self.n_layers = len(self.layers);self.d = self.model.config.hidden_size
        self.lw, self.lr = lw, lr;self.W = None;self.stop = False;self.H = None;self.gains = None
        cfg = self.model.config;self.n_heads = cfg.num_attention_heads;self.hd = getattr(cfg, 'head_dim', None) or self.d//self.n_heads
        self.gate_layers = list(range(lw+1, lr+1))
        for i, b in enumerate(self.layers):b.register_forward_hook(self._hook(i))
        for j, i in enumerate(self.gate_layers):self.layers[i].self_attn.o_proj.register_forward_pre_hook(self._gate(j))
        self.pre = self.tok.encode(PREFIX, add_special_tokens=False);ph = self.tok.encode('...', add_special_tokens=False)
        assert len(ph) == 1, ph
        self.ph = ph[0];self.P0 = len(self.pre)                       # placeholders at [P0, P0+NPH)
        self.col_ids = [self.tok.encode(' '+c, add_special_tokens=False) for c in COLORS]
        assert all(len(x) == 1 for x in self.col_ids), 'colour words must be single tokens'
        self.col_ids = [x[0] for x in self.col_ids]
        self.q_ids = self.tok.encode(QUESTION, add_special_tokens=False)

    def _hook(self, i):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if i == self.lw and self.W is not None:
                h = torch.cat([h[:, :self.P0], h[:, self.P0:self.P0+NPH]+self.W.to(h.dtype), h[:, self.P0+NPH:]], 1)
                out = (h,)+tuple(out[1:]) if isinstance(out, tuple) else h
            if i == self.lr:
                self.H = h
                if self.stop:raise Stop()
            return out
        return fn

    def _gate(self, j):
        def pre(mod, args):
            if self.gains is None:return None
            x = args[0];B, T, D = x.shape;g = self.gains[:, j*self.n_heads:(j+1)*self.n_heads].to(x.dtype)
            return ((x.reshape(B, T, self.n_heads, self.hd)*g[:, None, :, None]).reshape(B, T, D),)
        return pre

    def batch(self, seqs):
        """Right-padded [prefix][placeholders][tokens]; returns ids, attention mask, lengths."""
        full = [self.pre+[self.ph]*NPH+s for s in seqs];T = max(len(x) for x in full)
        ids = torch.full((len(full), T), self.tok.pad_token_id or 0, dtype=torch.long);att = torch.zeros_like(ids)
        for i, x in enumerate(full):ids[i, :len(x)] = torch.tensor(x);att[i, :len(x)] = 1
        return ids.to(DEV), att.to(DEV), torch.tensor([len(x) for x in full], device=DEV)

    def read_chunk(self, seqs, W, gains=None):
        """Forward up to block L_r with the write; returns the block-L_r states (B, T, d), mask, lengths."""
        ids, att, ln = self.batch(seqs);self.W, self.stop, self.gains = W, True, gains
        try:self.model(input_ids=ids, attention_mask=att, use_cache=False)
        except Stop:pass
        finally:self.W, self.stop, self.gains = None, False, None
        H = self.H;self.H = None;return H, att, ln

    def answer(self, B, W, gains=None):
        """Logits over the colour tokens after the question, with the written state in the placeholders."""
        ids, att, ln = self.batch([self.q_ids]*B);self.W, self.gains = W, gains
        try:o = self.model(input_ids=ids, attention_mask=att, use_cache=False, logits_to_keep=1)
        finally:self.W, self.gains = None, None
        return o.logits[:, -1, self.col_ids].float()


# ---------------------------------------------------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------------------------------------------------
class Loop(nn.Module):
    def __init__(self, d, n_gates, reader, query, heads, mu, sd, n_w):
        super().__init__();self.reader, self.query_src, self.heads, self.d, self.n_w = reader, query, heads, d, n_w
        self.register_buffer('mu', mu);self.register_buffer('sd', sd)
        self.Wz = nn.Linear(DZ, NZ*d);self.Wf = nn.Linear(DZ, NF*d)
        self.S = nn.Sequential(nn.Linear(DZ, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, DZ))
        self.z0 = nn.Parameter(torch.randn(DZ));self.f0 = nn.Parameter(torch.randn(DZ))
        if reader == 'attn':
            self.Wk = nn.Linear(d, RQ, bias=False);nn.init.normal_(self.Wk.weight, std=1/math.sqrt(d))
            self.Rp = nn.Linear(d, DZ)
            if query == 'const':self.q0 = nn.Parameter(torch.randn(RQ))
            elif query == 'rand':self.register_buffer('Rq', torch.randn(RQ, DZ)/math.sqrt(DZ)*2)
            else:self.Wq = nn.Linear(DZ, RQ)
        else:self.Rl = nn.Linear(d, DZ)
        if heads == 'state':self.Wg = nn.Linear(DZ, n_gates);nn.init.zeros_(self.Wg.weight);nn.init.zeros_(self.Wg.bias)
        elif heads == 'const':self.g0 = nn.Parameter(torch.zeros(n_gates))

    def forecast(self, z):
        return sph(self.S(z))

    def forecast_frozen(self, z):
        return sph(functional_call(self.S, {n:p.detach() for n, p in self.S.named_parameters()}, (z,)))

    def gains(self, z):
        if self.heads == 'state':return 1+0.5*torch.tanh(self.Wg(z))
        if self.heads == 'const':return 1+0.5*torch.tanh(self.g0)[None].expand(z.shape[0], -1)
        return None

    def query(self, z_prev, zh):
        s = self.query_src
        if s == 'const':return self.q0[None].expand(z_prev.shape[0], RQ)
        if s == 'rand':return z_prev@self.Rq.T
        return self.Wq(zh if s == 'fcst' else z_prev)

    def write(self, z, c):
        w = torch.cat([self.Wz(z).view(-1, NZ, self.d), self.Wf(c).view(-1, NF, self.d)], 1)
        return self.n_w*w/w.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def read(self, H, att, ln, P0, q=None):
        """Next state from the block-L_r states. attn: equal-norm standardised states of the placeholders and the chunk
        (not the prefix, not padding), pooled with the query; last: the last real token."""
        Hs = (H.float()-self.mu)/self.sd
        if self.reader != 'attn':
            hl = Hs[torch.arange(H.shape[0], device=H.device), ln-1];return sph(self.Rl(hl)), None
        Hn = Hs*math.sqrt(Hs.shape[-1])/Hs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        elig = att.bool().clone();elig[:, :P0] = False
        a = torch.softmax(((self.Wk(Hn)@q[:, :, None]).squeeze(-1)/math.sqrt(RQ)).masked_fill(~elig, -1e9), -1)
        return sph(self.Rp((a[:, :, None]*Hn).sum(1))), a


def nce(pred, tgt, tau=0.1):
    """Contrastive: the forecast must pick its own target among all targets in the batch."""
    p = F.normalize(pred, dim=-1);t = F.normalize(tgt, dim=-1);lg = p@t.T/tau
    return F.cross_entropy(lg, torch.arange(len(p), device=p.device))


def episode(fm, loop, streams, lam=0., z_fn=None, q_fn=None, keep=False):
    """Run the loop over a batch of rendered streams (list of dicts with 'ids': K token lists). Returns colour logits per
    chunk (B, K, n_col), states, forecasts, attentions, the self-model's loss and the pressure loss."""
    B = len(streams);K = len(streams[0]['ids']);zc = sph(loop.z0.expand(B, DZ));fc = sph(loop.f0.expand(B, DZ));z_prev = zc
    LG, Z, ZH, Q, AT, LS, LP = [], [], [], [], [], [], []
    for k in range(K):
        zh = loop.forecast_frozen(z_prev);zin = z_prev if k > 0 else zc
        if z_fn is not None:zin = z_fn(k, zin)
        cin = fc if k == 0 else zh
        g = loop.gains(zin)
        H, att, ln = fm.read_chunk([s['ids'][k] for s in streams], loop.write(zin, cin), g)
        q = None
        if loop.reader == 'attn':
            q = loop.query(zin, loop.forecast_frozen(zin) if k > 0 else zh)
            if q_fn is not None:q = q_fn(k, q)
        z, a = loop.read(H, att, ln, fm.P0, q)
        LG.append(fm.answer(B, loop.write(z, loop.forecast_frozen(z)), loop.gains(z)))
        if k > 0:
            LS.append(nce(loop.forecast(z_prev.detach()), z.detach()))
            if lam > 0:LP.append(nce(loop.forecast_frozen(z_prev), z))
        Z.append(z);ZH.append(zh)
        if keep:Q.append(q.detach() if q is not None else None);AT.append((a.detach(), att.detach()) if a is not None else None)
        z_prev = z
    ls = torch.stack(LS).mean() if LS else torch.zeros((), device=DEV);lp = torch.stack(LP).mean() if LP else None
    return dict(logits=torch.stack(LG, 1), z=torch.stack(Z, 1), zh=torch.stack(ZH, 1), q=Q, attn=AT, ls=ls, lp=lp)


# ---------------------------------------------------------------------------------------------------------------------
# data, statistics, training, evaluation
# ---------------------------------------------------------------------------------------------------------------------
def task_args(a):
    return dict(n_an=a.n_an, n_col=a.n_col, K=a.K, n_upd=a.n_upd, p_switch=a.p_switch, p_focus=a.p_focus)


def rendered(fm, st):
    R = [render(fm.tok, h, S) for h, S in st['chunks']]
    return dict(st, ids=[r[0] for r in R], spans=[r[1] for r in R])


@torch.no_grad()
def stats(fm, a, n=64):
    rng = np.random.default_rng(12345);Hs = []
    for i in range(0, n, 16):
        sts = [rendered(fm, make_stream(rng, **task_args(a))) for _ in range(16)]
        for k in range(a.K):
            H, att, ln = fm.read_chunk([s['ids'][k] for s in sts], None)
            m = att.bool().clone();m[:, :fm.P0+NPH] = False;Hs.append(H.float()[m].cpu())
    Hs = torch.cat(Hs);mu = Hs.mean(0);sd = Hs.std(0)+1e-3*Hs.std(0).mean()
    ids, att, ln = fm.batch([s['ids'][0] for s in sts]);fm.W = None
    with torch.no_grad():
        out = fm.model(input_ids=ids, attention_mask=att, use_cache=False, output_hidden_states=True)
    n_w = float(out.hidden_states[fm.lw+1][:, fm.P0+NPH:].float().norm(dim=-1).median())
    return mu, sd, n_w


def tag_of(a):
    r = f'attn-{a.query}' if a.reader == 'attn' else 'last'
    return r+(f'-heads-{a.heads}' if a.heads != 'none' else '')+f'_l{a.lam:g}'+(f'_s{a.seed}' if a.seed else '')


def build(fm, a, out):
    sp = out/'stats.pt'
    if sp.exists():mu, sd, n_w = torch.load(sp)
    else:mu, sd, n_w = stats(fm, a);torch.save((mu, sd, n_w), sp)
    ng = len(fm.gate_layers)*fm.n_heads
    return Loop(fm.d, ng, a.reader, a.query, a.heads, mu.to(DEV), sd.to(DEV), n_w).to(DEV)


def train(a):
    out = pathlib.Path(a.out);out.mkdir(parents=True, exist_ok=True);tag = tag_of(a);ck = out/'ckpt'/tag;ck.mkdir(parents=True, exist_ok=True)
    fm = Frozen(a.model, a.lw, a.lr, a.local_only);torch.manual_seed(a.seed);loop = build(fm, a, out)
    opt = torch.optim.AdamW([p for p in loop.parameters() if p.requires_grad], lr=a.lrate, weight_decay=0.)
    warm = 30;sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s:min(1., (s+1)/warm)*(0.1+0.9*0.5*(1+math.cos(math.pi*min(1., s/a.steps)))))
    rng = np.random.default_rng(1000+a.seed);hist = [];t0 = time.time()
    vrng = np.random.default_rng(777);val = [rendered(fm, make_stream(vrng, **task_args(a))) for _ in range(64)]
    for s in range(a.steps):
        sts = [rendered(fm, make_stream(rng, **task_args(a))) for _ in range(a.bs)]
        gold = torch.tensor([x['gold'] for x in sts], device=DEV)
        R = episode(fm, loop, sts, lam=a.lam)
        la = F.cross_entropy(R['logits'].reshape(-1, a.n_col), gold.reshape(-1))
        loss = la+R['ls']+(a.lam*R['lp'] if R['lp'] is not None else 0.)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(loop.parameters(), 1.0);opt.step();sched.step()
        rec = dict(step=s, ans=float(la), acc=float((R['logits'].argmax(-1) == gold).float().mean()), ls=float(R['ls']),
                   lp=float(R['lp']) if R['lp'] is not None else None, t=time.time()-t0)
        if s % 50 == 0 or s == a.steps-1:
            with torch.no_grad():
                acc = []
                for i in range(0, len(val), a.bs):
                    vb = val[i:i+a.bs];Rv = episode(fm, loop, vb);gv = torch.tensor([x['gold'] for x in vb], device=DEV)
                    acc.append((Rv['logits'].argmax(-1) == gv).float().cpu())
                acc = torch.cat(acc);rec['val_acc'] = float(acc.mean());rec['val_acc_by_chunk'] = acc.mean(0).tolist()
            log(tag, f"step {s} answer {rec['ans']:.3f} acc {rec['acc']:.3f} | val acc {rec['val_acc']:.3f} by chunk "
                     f"{np.round(rec['val_acc_by_chunk'], 2).tolist()} | ls {rec['ls']:.3f} lp {rec['lp']} | {rec['t']:.0f}s")
        hist.append(rec)
    torch.save(loop.state_dict(), ck/'loop.pt');json.dump(dict(cfg=vars(a), tag=tag, hist=hist), open(ck/'train.json', 'w'))
    log(tag, 'trained', f'{time.time()-t0:.0f}s')


@torch.no_grad()
def evaluate(a):
    out = pathlib.Path(a.out);tag = a.tag;ck = out/'ckpt'/tag;cfg = json.load(open(ck/'train.json'))['cfg']
    for k in ('reader', 'query', 'heads', 'lam'):setattr(a, k, cfg[k])
    fm = Frozen(a.model, a.lw, a.lr, a.local_only);loop = build(fm, a, out);loop.load_state_dict(torch.load(ck/'loop.pt', map_location=DEV));loop.eval()
    rng = np.random.default_rng(4242);n = a.n_eval;bs = a.bs;res = dict(tag=tag, cfg=cfg)
    sts = [rendered(fm, make_stream(rng, **task_args(a))) for _ in range(n)]
    def run(S, **kw):
        R = [episode(fm, loop, S[i:i+bs], keep=True, **kw) for i in range(0, len(S), bs)]
        return R
    def acc_of(R, S):
        g = torch.tensor([x['gold'] for x in S]);p = torch.cat([r['logits'].argmax(-1).cpu() for r in R]);return (p == g).float()
    R0 = run(sts);A0 = acc_of(R0, sts);sw = np.array([x['switch'] >= 0 for x in sts])
    res['acc_by_chunk'] = A0.mean(0).tolist();res['acc_final'] = float(A0[:, -1].mean())
    res['acc_final_switch'] = float(A0[sw, -1].mean()) if sw.any() else None;res['acc_final_noswitch'] = float(A0[~sw, -1].mean())
    # no memory: the initial constant in every chunk
    Rn = run(sts, z_fn=lambda k, z:sph(loop.z0.expand(z.shape[0], DZ)));An = acc_of(Rn, sts)
    res['no_memory_acc_by_chunk'] = An.mean(0).tolist()
    # where the selection goes (attention readers): mass on the tracked animal's sentence, other sentences, placeholders
    if loop.reader == 'attn':
        mf, mo, mp, mh = [], [], [], []
        for r, i0 in zip(R0, range(0, n, bs)):
            for k, (at, msk) in enumerate(r['attn']):
                for j in range(at.shape[0]):
                    st = sts[i0+j];base = fm.P0+NPH;foc = st['focus'][k]
                    span_f = [(base+s0, base+s1) for (an, c, s0, s1) in st['spans'][k] if an == foc]
                    span_o = [(base+s0, base+s1) for (an, c, s0, s1) in st['spans'][k] if an != foc]
                    mp.append(float(at[j, fm.P0:fm.P0+NPH].sum()))
                    mf.append(float(sum(at[j, x:y].sum() for x, y in span_f)) if span_f else np.nan)
                    mo.append(float(sum(at[j, x:y].sum() for x, y in span_o))/max(1, len(span_o)))
                    mh.append(1.0 if span_f else 0.0)
        mf, mo, mp, mh = map(np.array, (mf, mo, mp, mh))
        res['attn'] = dict(on_tracked_sentence_when_present=float(np.nanmean(mf)), on_each_other_sentence=float(mo.mean()),
                           on_placeholders_when_tracked_absent=float(mp[mh == 0].mean()) if (mh == 0).any() else None,
                           on_placeholders_when_tracked_present=float(mp[mh == 1].mean()) if (mh == 1).any() else None)
    # focus transplant: the same chunks with another first instruction (an animal also updated in chunk 0), no switch
    rng2 = np.random.default_rng(99);pairs = []
    while len(pairs) < a.n_pairs:
        st = make_stream(rng2, **dict(task_args(a), p_switch=0.))
        cand = [an for an, c, t in st['chunks'][0][1] if an != st['focus'][0]]
        if not cand:continue
        f2 = int(rng2.choice(cand));tw = make_stream(np.random.default_rng(0), **dict(task_args(a), p_switch=0.), focus=f2, chunk0=st['chunks'][0][1])
        tw['chunks'] = [(f'Keep track of the {ANIMALS[f2]}.', st['chunks'][0][1])]+st['chunks'][1:]
        # recompute the twin's gold from the shared chunks
        col = {};gold = []
        for h, S in tw['chunks']:
            for an, c, t in S:col[an] = c
            gold.append(col[f2])
        tw['gold'] = gold;tw['focus'] = [f2]*a.K
        if st['gold'][-1] == tw['gold'][-1]:continue
        pairs.append((rendered(fm, st), rendered(fm, tw)))
    A_ = [p[0] for p in pairs];B_ = [p[1] for p in pairs];at = a.transplant_at
    RA, RB = run(A_), run(B_)
    zB = torch.cat([r['z'] for r in RB]);qB = [torch.cat([r['q'][k] for r in RB]) if RB[0]['q'][k] is not None else None for k in range(a.K)]
    def by_batch(full):
        st_ = dict(i=0)
        def fn(k, x):
            b = slice(st_['i'], st_['i']+x.shape[0])
            if k == a.K-1:st_['i'] += x.shape[0]
            return full[b, k-1] if k >= at else x
        return fn
    def outcome(R):
        p = torch.cat([r['logits'][:, -1].argmax(-1).cpu() for r in R])
        ga = torch.tensor([x['gold'][-1] for x in A_]);gb = torch.tensor([x['gold'][-1] for x in B_])
        return dict(answers_own_animal=float((p == ga).float().mean()), answers_other_animal=float((p == gb).float().mean()))
    res['transplant'] = dict(at_chunk=at, own=outcome(RA), twin=outcome(RB) if False else None, state=outcome(run(A_, z_fn=by_batch(zB))))
    if loop.reader == 'attn' and loop.query_src in ('state', 'fcst', 'rand'):
        def qfn():
            st_ = dict(i=0)
            def fn(k, q):
                b = slice(st_['i'], st_['i']+q.shape[0])
                if k == a.K-1:st_['i'] += q.shape[0]
                return qB[k][b] if k >= at else q
            return fn
        res['transplant']['query_only'] = outcome(run(A_, q_fn=qfn()))
    # states for probes
    np.savez_compressed(out/f'eval_{tag}.npz', z=torch.cat([r['z'] for r in R0]).float().cpu().numpy(),
                        focus=np.array([x['focus'] for x in sts]), gold=np.array([x['gold'] for x in sts]), switch=np.array([x['switch'] for x in sts]),
                        acc=A0.numpy(), acc_no_memory=An.numpy())
    json.dump(res, open(out/f'eval_{tag}.json', 'w'), indent=1)
    log(tag, 'final acc', round(res['acc_final'], 3), '| by chunk', np.round(res['acc_by_chunk'], 3).tolist(), '| no memory',
        np.round(res['no_memory_acc_by_chunk'], 3).tolist(), '| switch / no switch', res['acc_final_switch'], res['acc_final_noswitch'])
    log(tag, 'transplant', json.dumps(res['transplant']), '| attn', json.dumps(res.get('attn')))


def main():
    p = argparse.ArgumentParser();p.add_argument('stage', choices=['train', 'eval'])
    p.add_argument('--out', required=True);p.add_argument('--model', default='Qwen/Qwen3-1.7B');p.add_argument('--local_only', type=int, default=0)
    p.add_argument('--lw', type=int, default=10);p.add_argument('--lr', type=int, default=18)
    p.add_argument('--reader', default='attn', choices=['attn', 'last']);p.add_argument('--query', default='state', choices=['state', 'fcst', 'const', 'rand'])
    p.add_argument('--heads', default='none', choices=['none', 'state', 'const']);p.add_argument('--lam', type=float, default=0.05)
    p.add_argument('--steps', type=int, default=600);p.add_argument('--bs', type=int, default=32);p.add_argument('--lrate', type=float, default=2e-3)
    p.add_argument('--seed', type=int, default=0);p.add_argument('--tag', default=None)
    p.add_argument('--n_an', type=int, default=10);p.add_argument('--n_col', type=int, default=10);p.add_argument('--K', type=int, default=6)
    p.add_argument('--n_upd', type=int, default=4);p.add_argument('--p_switch', type=float, default=0.3);p.add_argument('--p_focus', type=float, default=0.5)
    p.add_argument('--n_eval', type=int, default=512);p.add_argument('--n_pairs', type=int, default=256);p.add_argument('--transplant_at', type=int, default=2)
    a = p.parse_args()
    if a.stage == 'train':train(a)
    else:evaluate(a)


if __name__ == '__main__':
    main()
