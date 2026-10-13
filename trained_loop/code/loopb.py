"""Phase B (research_v5/PLAN_B.md rev 3): a trained self-describing loop around a frozen Qwen3 thinking its own thoughts.

The model's own thinking traces are cut into K = 8 segments of 64 tokens. Segment k is processed with no access to any
earlier token; its only link to the past is a 32-d state z read from the model itself at the end of segment k-1 (block
L_r output, last token) and written into 4 placeholder positions at its start (block L_w output). Two more placeholder
positions form a channel whose content depends on the arm:
  cut    the state is never passed on (constant z): the floor
  const  the channel carries a constant
  state  the channel carries the previous state again (an extra memory channel: the matched control)
  fcst   the channel carries the self-model's forecast of the state the segment will end in (the strange loop)
A self-model S (z_{k-1} -> z_k) is trained in every arm with its own prediction loss only (detached inputs and targets).
lam is the pressure on the system (reader / writers) to make its own next state predictable by S: the same loss through
a parameter-detached copy of S. The language model stays frozen.
  gen    traces (dolly-15k no-context instructions, thinking mode), natural statistics, readers, reference NLLs
  train  one arm;  gate  stop the queue if the plain loop carries nothing (necessity check on the const arm)
  eval   held-out traces: own / foreign / stale states, channel swap, corruption at segment 3 (identity, fresh vs stale
         forecasts), self-prediction against persistence / segment-mean / linear baselines, word readouts, and free
         generation (the system thinks on its own with only the loop as memory; own vs corrupted vs channel-swapped,
         paired sampling randomness)
    python loopb.py gen   --out /workspace/v5/loopb
    python loopb.py train --out /workspace/v5/loopb --chan fcst --reader learned --lam 1 --steps 600
    python loopb.py eval  --out /workspace/v5/loopb --tag fcst_learned_l1"""
import argparse, json, math, os, pathlib, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

V3 = pathlib.Path(os.environ.get('V3_CODE', pathlib.Path(__file__).resolve().parents[2]/'common'))
sys.path.insert(0, str(V3))
import common                                          # noqa: E402
from common import LM, log, atomic_npz, atomic_json   # noqa: E402

MODEL = os.environ.get('B_MODEL', '8b');KEY = {'8b':'qwen3-8b', '14b':'qwen3-14b', 'cpu':'qwen3-1.7b-smoke'}[MODEL]
LR, LW = {'8b':(29, 22), '14b':(32, 24), 'cpu':(20, 12)}[MODEL]
common.MODELS[KEY]['read'] = LR;common.MODELS[KEY]['write'] = LW
SMOKE = os.environ.get('B_SMOKE') == '1'
KS, NZ, NF, DZ = 8, 4, 2, 32                          # segments, z / channel positions, state dim
SEG = int(os.environ.get('B_SEG', 64))                # tokens per segment (smaller only for CPU smoke tests)
NP = 1+NZ+NF                                          # sink token + placeholders
T = NP+SEG
NB = min(8, SEG)                                      # boundary tokens: the first NB of each segment
CATS = ['open_qa', 'general_qa', 'brainstorming', 'creative_writing', 'classification']
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def sph(x):
    return math.sqrt(DZ)*x/x.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def cos(a, b):
    return F.cosine_similarity(a, b, dim=-1)


def tag_of(chan, reader, lam, ploss='cos', seed=0):
    return f'{chan}_{reader}_l{lam:g}'+{'cos':'', 'nmse':'_n', 'nce':'_c', 'ncex':'_cx'}[ploss]+(f'_s{seed}' if seed else '')


# ---------------------------------------------------------------------------------------------------------------------
# substrate
# ---------------------------------------------------------------------------------------------------------------------
def load_lm():
    lm = LM(KEY, dtype=torch.bfloat16 if DEV == 'cuda' else torch.float32, dev=DEV)
    torch.backends.cuda.matmul.allow_tf32 = True
    s = lm.tok.encode('\n', add_special_tokens=False);p = lm.tok.encode('...', add_special_tokens=False)
    assert len(s) == 1 and len(p) == 1, (s, p)
    lm.ph = torch.tensor([s+p*(NZ+NF)], device=DEV)   # (1, NP): sink + placeholders
    return lm


def edit_fn(W, dtype):
    Wb = W.to(dtype)
    return lambda h, Wb=Wb:torch.cat([h[:, :1], h[:, 1:NP]+Wb, h[:, NP:]], 1) if h.shape[1] >= NP else h


def run_seg(lm, toks_seg, W):
    """One segment: [sink, placeholders, tokens]; W (B, NP-1, d) is added to the placeholders at block L_w.
    Returns per-token NLL (B, n) and the block-L_r state of the last token (B, d)."""
    B, n = toks_seg.shape;ids = torch.cat([lm.ph.expand(B, -1), toks_seg], 1)
    lm.clear();lm.read[LR] = torch.full((B, 1), NP+n-1, device=DEV, dtype=torch.long)
    if W is not None:lm.edit[LW] = edit_fn(W, lm.model.dtype)
    o = lm.model(input_ids=ids, use_cache=False, logits_to_keep=n+1)
    h = lm.store[LR][:, 0];lm.clear()
    nll = F.cross_entropy(o.logits[:, :-1].float().transpose(1, 2), toks_seg, reduction='none')
    return nll, h


class Loop(nn.Module):
    def __init__(self, d, reader, stats, chan):
        super().__init__();self.d = d;self.reader = reader;self.chan = chan
        self.register_buffer('mu', torch.tensor(stats['mu'], dtype=torch.float32))
        self.register_buffer('sd', torch.tensor(stats['sd'], dtype=torch.float32))
        A0 = torch.tensor(stats['A_'+reader], dtype=torch.float32)          # (DZ, d), standardised coordinates
        if reader == 'learned':self.A = nn.Parameter(A0.clone())
        else:self.register_buffer('A', A0.clone())
        self.Wz = nn.Linear(DZ, NZ*d);self.Wf = nn.Linear(DZ, NF*d)
        self.S = nn.Sequential(nn.Linear(DZ, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, DZ))
        self.z0 = nn.Parameter(torch.randn(DZ));self.f0 = nn.Parameter(torch.randn(DZ))
        self.n_w = float(stats['n_w'])

    def read(self, h):
        return sph(((h-self.mu)/self.sd)@self.A.T)

    def forecast(self, z):                              # S with its own parameters (trained by L_S only)
        return sph(self.S(z))

    def forecast_frozen(self, z):                       # same function, parameters detached: gradients reach z only
        return sph(functional_call(self.S, {n:p.detach() for n, p in self.S.named_parameters()}, (z,)))

    def write(self, z, c):
        w = torch.cat([self.Wz(z).view(-1, NZ, self.d), self.Wf(c).view(-1, NF, self.d)], 1)
        return self.n_w*w/w.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def pred_loss(P, T_, kind):
    """Self-prediction loss per sample (B, K-1). cos: 1 - cosine on the sphere (can be satisfied by collapsing every state
    onto one direction); nmse: squared error divided by each dimension's variance over the batch and segments (detached),
    i.e. 1 - R^2 per dimension, which collapse or hiding information in small deviations cannot reduce."""
    if kind == 'cos':return 1-cos(P, T_)
    if kind == 'ncex':                                 # pick the true next state out of the other traces' states at the same moment
        Pn, Tn = F.normalize(P, dim=-1), F.normalize(T_, dim=-1);B_ = P.shape[0]
        lg = torch.einsum('bkd,ckd->kbc', Pn, Tn)/0.1
        return F.cross_entropy(lg.reshape(-1, B_), torch.arange(B_, device=P.device).repeat(P.shape[1]), reduction='none').view(P.shape[1], B_).T
    if kind == 'nce':                                  # pick the true next state out of every state in the batch
        B_, K_ = P.shape[:2];C = F.normalize(T_.reshape(-1, DZ), dim=-1)
        lg = F.normalize(P.reshape(-1, DZ), dim=-1)@C.T/0.1
        return F.cross_entropy(lg, torch.arange(B_*K_, device=P.device), reduction='none').view(B_, K_)
    v = T_.detach().reshape(-1, DZ).var(0)+1e-4*T_.detach().reshape(-1, DZ).var(0).mean()
    return ((P-T_)**2/v).mean(-1)


def episode(lm, loop, toks, zin_fn=None, cin_fn=None, lam=0., ploss='cos'):
    """Closed loop over the K segments. zin_fn(k, zin) / cin_fn(k, cin) may replace the written state / channel content.
    Returns per-token nll (B, K, SEG), states z (B, K, DZ), forecasts zh (B, K, DZ) (S(z_{k-1}); row 0 = S(z_init)),
    channel contents (B, K, DZ), S's own loss and the system's self-predictability loss (both (B, K-1))."""
    B = toks.shape[0];zc = sph(loop.z0.expand(B, DZ));fc = sph(loop.f0.expand(B, DZ));chan = loop.chan
    z_prev = zc;N, Z, ZH, CH, LS, LP = [], [], [], [], [], []
    for k in range(KS):
        zin = zc if (chan == 'cut' or k == 0) else z_prev
        zh = loop.forecast_frozen(z_prev)                # the forecast of the state this segment ends in
        cin = fc if (chan in ('cut', 'const') or k == 0) else (z_prev if chan == 'state' else zh)
        if zin_fn is not None:zin = zin_fn(k, zin)
        if cin_fn is not None:cin = cin_fn(k, cin)
        nll, h = run_seg(lm, toks[:, k*SEG:(k+1)*SEG], loop.write(zin, cin))
        z = loop.read(h)
        if k > 0:
            LS.append((loop.forecast(z_prev.detach()), z.detach()))
            if lam > 0:LP.append((zh, z))
        N.append(nll);Z.append(z);ZH.append(zh);CH.append(cin)
        z_prev = z
    st = lambda L:pred_loss(torch.stack([a for a, _ in L], 1), torch.stack([b for _, b in L], 1), ploss) if L else None
    return torch.stack(N, 1), torch.stack(Z, 1), torch.stack(ZH, 1), torch.stack(CH, 1), st(LS), st(LP)


# ---------------------------------------------------------------------------------------------------------------------
# gen: traces, natural statistics, readers, reference NLLs
# ---------------------------------------------------------------------------------------------------------------------
def dolly(n, seed, cache):
    from huggingface_hub import hf_hub_download
    f = hf_hub_download('databricks/databricks-dolly-15k', 'databricks-dolly-15k.jsonl', repo_type='dataset',
                        local_dir=cache)
    rows = [json.loads(l) for l in open(f, encoding='utf-8')]
    rows = [(r['instruction'].strip(), r['category']) for r in rows
            if not r['context'].strip() and r['category'] in CATS and 15 <= len(r['instruction'].strip()) <= 400]
    rng = np.random.default_rng(seed);by = {c:[r for r in rows if r[1] == c] for c in CATS};out = [];i = 0
    for c in CATS:rng.shuffle(by[c])
    while len(out) < n and i < max(len(v) for v in by.values()):
        for c in CATS:
            if i < len(by[c]):out.append(by[c][i])
        i += 1
    return out[:n]


@torch.no_grad()
def gen_traces(lm, prompts, bs):
    tok = lm.tok;eos = {tok.convert_tokens_to_ids('<|im_end|>'), tok.convert_tokens_to_ids('<|endoftext|>')}
    keep, kidx, t0 = [], [], time.time()
    for b in range(0, len(prompts), bs):
        texts = [lm.chat(p, thinking=True) for p, _ in prompts[b:b+bs]]
        enc = tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to(DEV)
        g = lm.model.generate(**enc, max_new_tokens=KS*SEG, do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                              pad_token_id=tok.pad_token_id)
        new = g[:, enc['input_ids'].shape[1]:].cpu().numpy()
        for j in range(new.shape[0]):
            if new.shape[1] == KS*SEG and not any(int(t) in eos for t in new[j]):
                keep.append(new[j].astype(np.int32));kidx.append(b+j)
        log(f'gen {b+len(texts)}/{len(prompts)} kept {len(keep)} ({time.time()-t0:.0f}s)')
    return np.stack(keep), np.array(kidx)


@torch.no_grad()
def natural(lm, toks, bs=32):
    """Natural segment-end states at L_r (no writes) and residual norms at L_w on token positions."""
    H, norms = [], []
    for b in range(0, len(toks), bs):
        tb = torch.tensor(toks[b:b+bs], device=DEV, dtype=torch.long)
        for k in range(KS):
            ts = tb[:, k*SEG:(k+1)*SEG];ids = torch.cat([lm.ph.expand(ts.shape[0], -1), ts], 1)
            lm.clear();lm.read[LR] = torch.full((ts.shape[0], 1), T-1, device=DEV, dtype=torch.long);lm.read[LW] = None
            lm.model(input_ids=ids, use_cache=False, logits_to_keep=1)
            H.append(lm.store[LR][:, 0].cpu());norms.append(lm.store[LW][:, NP:].norm(dim=-1).flatten().cpu());lm.clear()
    return torch.cat(H).numpy(), torch.cat(norms).numpy()


@torch.no_grad()
def ref_nll(lm, prompts, toks, bs=8):
    """Per-token NLL of the trace (a) in its natural full context (prompt + all earlier thought) and (b) isolated
    (sink + placeholders + the segment, no writes)."""
    full, iso = np.zeros((len(toks), KS, SEG), np.float32), np.zeros((len(toks), KS, SEG), np.float32);tok = lm.tok
    for b in range(0, len(toks), bs):
        pre = [tok.encode(lm.chat(p, thinking=True), add_special_tokens=False) for p in prompts[b:b+bs]]
        L = max(len(x) for x in pre);tb = toks[b:b+bs]
        ids = torch.tensor([[tok.pad_token_id]*(L-len(x))+x+list(tb[j]) for j, x in enumerate(pre)], device=DEV)
        att = torch.tensor([[0]*(L-len(x))+[1]*(len(x)+KS*SEG) for x in pre], device=DEV)
        pos = (att.long().cumsum(-1)-1).clamp_min(0);lm.clear()
        o = lm.model(input_ids=ids, attention_mask=att, position_ids=pos, use_cache=False, logits_to_keep=KS*SEG+1)
        tgt = torch.tensor(tb, device=DEV, dtype=torch.long)
        nll = F.cross_entropy(o.logits[:, :-1].float().transpose(1, 2), tgt, reduction='none')
        full[b:b+bs] = nll.view(-1, KS, SEG).cpu().numpy();del o
        for k in range(KS):
            n, _ = run_seg(lm, tgt[:, k*SEG:(k+1)*SEG], None);iso[b:b+bs, k] = n.cpu().numpy()
    return full, iso


@torch.no_grad()
def stage_gen(a):
    out = pathlib.Path(a.out);out.mkdir(parents=True, exist_ok=True);lm = load_lm()
    n = int(os.environ.get('B_SMOKE_N', 60)) if SMOKE else a.n;prompts = dolly(int(n*1.35), 0, a.cache)
    toks, kidx = gen_traces(lm, prompts, a.gen_bs);prompts = [prompts[i] for i in kidx]
    N = len(toks);rng = np.random.default_rng(1);perm = rng.permutation(N)
    n_te = max(8, int(round(N*400/2400))) if not SMOKE else max(4, N//4)
    n_va = max(8, int(round(N*200/2400))) if not SMOKE else max(2, N//8)
    split = np.zeros(N, np.int8);split[perm[:n_te]] = 2;split[perm[n_te:n_te+n_va]] = 1
    cat = np.array([CATS.index(c) for _, c in prompts], np.int8)
    atomic_npz(out/'data.npz', toks=toks, cat=cat, split=split)
    atomic_json(out/'prompts.json', [p for p, _ in prompts])
    log('traces', N, 'test', int((split == 2).sum()), 'val', int((split == 1).sum()), 'by category',
        np.bincount(cat, minlength=len(CATS)).tolist())
    tr = np.where(split == 0)[0][:64 if SMOKE else 512]
    H, norms = natural(lm, toks[tr]);mu = H.mean(0);sd = H.std(0)+1e-3*H.std(0).mean()
    Hs = (H-mu)/sd;_, sv, Vt = np.linalg.svd(Hs-Hs.mean(0), full_matrices=False)
    A_learned = Vt[:DZ]                                                           # top PCs (standardised coordinates)
    D = lm.lens.D[LR].float()                                                    # J-lens word atoms at L_r
    P = torch.tensor(H-mu, device=DEV)@D.T                                       # word projections of natural states
    _, svj, Vtj = torch.linalg.svd(P-P.mean(0), full_matrices=False);svj = svj.cpu().numpy()
    AJ = (Vtj[:DZ]@D).cpu().numpy()*sd[None]                                     # (DZ, d) in standardised coordinates
    W_ = lm.lens.words;jax = {}
    for i in range(DZ):
        o = torch.argsort(Vtj[i]);jax[i] = dict(pos=[W_[int(j)] for j in o[-12:].flip(0)], neg=[W_[int(j)] for j in o[:12]])
    atomic_json(out/'j_axes.json', jax);del P, Vtj
    Q, _ = np.linalg.qr(np.random.default_rng(2).standard_normal((H.shape[1], DZ)));A_random = Q.T
    white = lambda A:A/((Hs@A.T).std(0)[:, None]+1e-8)                           # unit natural variance per coordinate
    A_learned, AJ, A_random = white(A_learned), white(AJ), white(A_random)
    evr = (sv[:DZ]**2).sum()/(sv**2).sum();evrj = (svj[:DZ]**2).sum()/(svj**2).sum()
    atomic_npz(out/'stats.npz', mu=mu, sd=sd, n_w=np.float32(np.median(norms)), A_learned=A_learned.astype(np.float32),
               A_J=AJ.astype(np.float32), A_random=A_random.astype(np.float32))
    log(f'natural: n_w {np.median(norms):.1f}; top-{DZ} PCs explain {evr:.2f} of standardised variance; J projection '
        f'top-{DZ} {evrj:.2f}')
    ev = np.where(split > 0)[0];full, iso = ref_nll(lm, [prompts[i][0] for i in ev], toks[ev])
    atomic_npz(out/'ref.npz', idx=ev, full=full.mean(-1), iso=iso.mean(-1), full_bnd=full[:, :, :NB].mean(-1),
               iso_bnd=iso[:, :, :NB].mean(-1))
    log('reference NLL (val+test) per segment: full context', full.mean((0, 2)).round(3).tolist(),
        '\n  isolated', iso.mean((0, 2)).round(3).tolist(), '\n  boundary tokens (segments 1-7): full',
        round(float(full[:, 1:, :NB].mean()), 3), 'isolated', round(float(iso[:, 1:, :NB].mean()), 3))


# ---------------------------------------------------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------------------------------------------------
def stage_train(a):
    out = pathlib.Path(a.out)
    if (out/'GATE_FAIL').exists():log('gate failed earlier; not training');return
    tag = tag_of(a.chan, a.reader, a.lam, a.ploss, a.seed);ck = out/'ckpt'/tag;ck.mkdir(parents=True, exist_ok=True)
    lm = load_lm();d = np.load(out/'data.npz');st = dict(np.load(out/'stats.npz'))
    toks = d['toks'];split = d['split'];tr = np.where(split == 0)[0];va = np.where(split == 1)[0][:48]
    torch.manual_seed(a.seed);loop = Loop(lm.d, a.reader, st, a.chan).to(DEV)
    groups = [dict(params=[p for n, p in loop.named_parameters() if p.requires_grad and n != 'A'], lr=a.lr)]
    if a.reader == 'learned':groups.append(dict(params=[loop.A], lr=a.lr/40))
    params = [p for g in groups for p in g['params']]
    opt = torch.optim.AdamW(groups, weight_decay=0.)
    steps = 6 if SMOKE else a.steps;warm = min(30, steps//5+1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s:min(1., (s+1)/warm)*(0.1+0.9*0.5*(1+math.cos(math.pi*min(1., s/steps)))))
    rng = np.random.default_rng(a.seed);hist = [];t0 = time.time()
    for s in range(steps):
        b = rng.choice(tr, a.bs, replace=False);tb = torch.tensor(toks[b], device=DEV, dtype=torch.long)
        nll, Z, ZH, CH, LS, LP = episode(lm, loop, tb, lam=a.lam, ploss=a.ploss)
        lml = nll.mean();loss = lml+LS.mean()+(a.lam*LP.mean() if LP is not None else 0.)
        opt.zero_grad(set_to_none=True);loss.backward()
        gn = {n:round(float(p.grad.norm()), 5) for n, p in loop.named_parameters() if p.grad is not None}
        torch.nn.utils.clip_grad_norm_(params, 1.0);opt.step();sched.step()
        rec = dict(step=s, lm=float(lml), lm_seg=nll.mean((0, 2)).tolist(), ls=float(LS.mean()),
                   lp=float(LP.mean()) if LP is not None else None, gn=gn, t=time.time()-t0,
                   mem=torch.cuda.max_memory_allocated()/2**30 if DEV == 'cuda' else 0.)
        if s % 25 == 0 or s == steps-1:
            with torch.no_grad():
                vb = torch.tensor(toks[va], device=DEV, dtype=torch.long);vn, vZ, vLS = [], [], []
                for i in range(0, len(va), a.bs):
                    n_, Z_, _, _, LS_, _ = episode(lm, loop, vb[i:i+a.bs], ploss=a.ploss);vn.append(n_);vZ.append(Z_);vLS.append(LS_)
                vn = torch.cat(vn);vZ = torch.cat(vZ)
                rec['val_lm'] = float(vn[:, 1:].mean());rec['val_bnd'] = float(vn[:, 1:, :NB].mean())
                rec['val_seg'] = vn.mean((0, 2)).tolist();rec['val_ls'] = float(torch.cat(vLS).mean())
                zz = vZ.reshape(-1, DZ).double();ev = torch.linalg.eigvalsh(torch.cov(zz.T));rec['val_pr'] = float(ev.sum()**2/(ev**2).sum())
                rec['val_collapse'] = float(cos(vZ.reshape(-1, DZ), vZ.reshape(-1, DZ).mean(0, keepdim=True)).mean())
            log(tag, f"step {s} lm {rec['lm']:.3f} ls {rec['ls']:.3f} lp {rec['lp']} | val lm(1-7) {rec['val_lm']:.4f} "
                f"bnd {rec['val_bnd']:.3f} ls {rec['val_ls']:.3f} PR {rec['val_pr']:.1f} coll {rec['val_collapse']:.3f} | mem {rec['mem']:.1f}G {rec['t']:.0f}s")
        hist.append(rec)
        if (s+1) % 200 == 0 or s == steps-1:
            torch.save(loop.state_dict(), ck/'loop.pt.tmp');os.replace(ck/'loop.pt.tmp', ck/'loop.pt')
            atomic_json(ck/'train.json', dict(cfg=vars(a), tag=tag, hist=hist))
    atomic_json(ck/'cfg.json', dict(vars(a), tag=tag, done=True))
    log(tag, 'trained', f'{time.time()-t0:.0f}s')


def stage_gate(a):
    """The bridge must carry something before the other arms are worth running."""
    out = pathlib.Path(a.out);f = out/'eval'/f'{a.tag}.json'
    if not f.exists():log('gate', a.tag, 'no evaluation: FAIL');(out/'GATE_FAIL').write_text('missing eval');return
    r = json.load(open(f));m, se = r['necessity_foreign'];ok = m > max(0.01, 3*se)
    log('gate', a.tag, 'necessity (foreign - own, nats/token)', round(m, 4), '+-', round(se, 4), 'PASS' if ok else 'FAIL')
    if not ok:(out/'GATE_FAIL').write_text(json.dumps(r['necessity_foreign']))


# ---------------------------------------------------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------------------------------------------------
def fit_observer(Ztr, Zte, seed=0, iters=3000):
    """Post-hoc self-model (same MLP as S) fitted on frozen states: z_{k-1} -> z_k. Returns S and test cosines."""
    torch.manual_seed(seed);S = nn.Sequential(nn.Linear(DZ, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                                              nn.Linear(256, DZ)).to(DEV)
    X = torch.tensor(Ztr[:, :-1].reshape(-1, DZ), device=DEV);Y = torch.tensor(Ztr[:, 1:].reshape(-1, DZ), device=DEV)
    opt = torch.optim.Adam(S.parameters(), lr=1e-3, weight_decay=1e-4)
    with torch.enable_grad():
        for i in range(iters):
            j = torch.randint(0, len(X), (512,), device=DEV);l = (1-cos(sph(S(X[j])), Y[j])).mean()
            opt.zero_grad();l.backward();opt.step()
    Xt = torch.tensor(Zte[:, :-1], device=DEV);Yt = torch.tensor(Zte[:, 1:], device=DEV)
    return S, cos(sph(S(Xt)), Yt).cpu().numpy()                                     # (N, K-1): predicts z_1..z_7


def baselines(Ztr, Zte):
    """Self-prediction baselines (cosine with z_k, k = 1..7): persistence, segment mean, linear ridge."""
    pers = (Zte[:, :-1]*Zte[:, 1:]).sum(-1)/DZ
    mk = Ztr.mean(0);mk = math.sqrt(DZ)*mk/np.linalg.norm(mk, axis=-1, keepdims=True)
    segm = (mk[None, 1:]*Zte[:, 1:]).sum(-1)/DZ
    X = np.concatenate([Ztr[:, :-1].reshape(-1, DZ), np.ones((Ztr[:, 1:].size//DZ, 1))], 1);Y = Ztr[:, 1:].reshape(-1, DZ)
    Wl = np.linalg.solve(X.T@X+1e-2*np.eye(DZ+1), X.T@Y)
    P = np.concatenate([Zte[:, :-1], np.ones(Zte[:, :-1].shape[:2]+(1,))], -1)@Wl
    lin = (P*Zte[:, 1:]).sum(-1)/(np.linalg.norm(P, axis=-1)*math.sqrt(DZ)+1e-9)
    return dict(persistence=pers.mean(0).tolist(), segment_mean=segm.mean(0).tolist(), linear=lin.mean(0).tolist())


@torch.no_grad()
def loop_words(lm, loop, Z, n=10):
    """J-lens words of (i) the reader's axes (block L_r), (ii) the mean write and the write along the top-4 principal
    axes of the states (block L_w): what the loop reads from the model and what it tells it."""
    Dr, Dw, W_ = lm.lens.D[LR].float(), lm.lens.D[LW].float(), lm.lens.words
    top = lambda v:[W_[int(j)] for j in torch.argsort(v, descending=True)[:n]]
    axes = (loop.A/loop.sd[None]).float();out = {}
    sc = axes@Dr.T;out['reader_axes'] = {i:dict(pos=top(sc[i]), neg=top(-sc[i])) for i in range(min(8, DZ))}
    zt = torch.tensor(Z.reshape(-1, DZ), device=DEV);zm = sph(zt.mean(0, keepdim=True))
    U = torch.linalg.svd(zt-zt.mean(0), full_matrices=False)[2][:4];fc = sph(loop.f0[None])
    out['write_mean'] = top(loop.write(zm, fc)[0, :NZ].mean(0)@Dw.T)
    for i in range(4):
        hi = loop.write(sph(zm+0.5*math.sqrt(DZ)*U[i:i+1]), fc)[0, :NZ].mean(0)
        lo = loop.write(sph(zm-0.5*math.sqrt(DZ)*U[i:i+1]), fc)[0, :NZ].mean(0)
        sc = (hi-lo)@Dw.T;out[f'write_pc{i}'] = dict(pos=top(sc), neg=top(-sc))
    return out


def sample(logits, gen, temp=0.6, top_p=0.95):
    p = torch.softmax(logits.float()/temp, -1);ps, ix = p.sort(-1, descending=True)
    keep = ps.cumsum(-1)-ps < top_p;ps = ps*keep;ps = ps/ps.sum(-1, keepdim=True)
    return ix.gather(-1, torch.multinomial(ps, 1, generator=gen))[:, 0]


@torch.no_grad()
def gen_seg(lm, W, gen):
    """Free generation of one segment after [sink, placeholders + W] (KV cache within the segment only)."""
    B = W.shape[0];lm.clear();lm.edit[LW] = edit_fn(W, lm.model.dtype)
    o = lm.model(input_ids=lm.ph.expand(B, -1), use_cache=True, logits_to_keep=1);lm.clear()
    cache = o.past_key_values;lg = o.logits[:, -1];out = []
    for t in range(SEG):
        nx = sample(lg, gen);out.append(nx)
        if t == SEG-1:break
        o = lm.model(input_ids=nx[:, None], past_key_values=cache, use_cache=True, logits_to_keep=1,
                     position_ids=torch.full((B, 1), NP+t, device=DEV, dtype=torch.long))
        cache = o.past_key_values;lg = o.logits[:, -1]
    return torch.stack(out, 1)


@torch.no_grad()
def embed(lm, toks):
    """Content embedding of token sequences: mean block-L_r state over the tokens, isolated, no writes."""
    B, n = toks.shape;ids = torch.cat([lm.ph.expand(B, -1), toks], 1);lm.clear();lm.read[LR] = None
    lm.model(input_ids=ids, use_cache=False, logits_to_keep=1);h = lm.store[LR][:, NP:].mean(1);lm.clear()
    return h


@torch.no_grad()
def free_run(lm, loop, toks, seed, swap=None, swap_at=3):
    """The system thinks on its own: segment 0 is its own trace, segments 1..7 are generated with only the loop as
    memory. swap: (zin_donor, cin_donor) tensors (B, DZ) entered at segment swap_at (either may be None)."""
    B = toks.shape[0];gen = torch.Generator(device=DEV);gen.manual_seed(seed)
    zc = sph(loop.z0.expand(B, DZ));fc = sph(loop.f0.expand(B, DZ));chan = loop.chan
    _, h = run_seg(lm, toks[:, :SEG], loop.write(zc, fc));z_prev = loop.read(h)
    segs, Z, ZH = [toks[:, :SEG]], [z_prev], [loop.forecast(zc)]
    for k in range(1, KS):
        zin = zc if chan == 'cut' else z_prev;zh = loop.forecast(z_prev)
        cin = fc if chan in ('cut', 'const') else (z_prev if chan == 'state' else zh)
        if swap is not None and k == swap_at:
            if swap[0] is not None:zin = swap[0]
            if swap[1] is not None:cin = swap[1]
        W = loop.write(zin, cin);g = gen_seg(lm, W, gen);_, h = run_seg(lm, g, W);z_prev = loop.read(h)
        segs.append(g);Z.append(z_prev);ZH.append(zh)
    return torch.stack(segs, 1), torch.stack(Z, 1), torch.stack(ZH, 1)


@torch.no_grad()
def stage_eval(a):
    out = pathlib.Path(a.out)
    if (out/'GATE_FAIL').exists() and not a.tag.startswith('const_learned_l0'):log('gate failed earlier; not evaluating');return
    ck = out/'ckpt'/a.tag;cfg = json.load(open(ck/'cfg.json'));chan = cfg['chan']
    lm = load_lm();d = np.load(out/'data.npz');st = dict(np.load(out/'stats.npz'));toks = d['toks'];split = d['split']
    loop = Loop(lm.d, cfg['reader'], st, chan).to(DEV);loop.load_state_dict(torch.load(ck/'loop.pt', map_location=DEV));loop.eval()
    te = np.where(split == 2)[0][:400];trs = np.where(split == 0)[0][:48 if SMOKE else 600];bs = a.bs
    res = dict(tag=a.tag, chan=chan, reader=cfg['reader'], lam=cfg['lam'], ploss=cfg.get('ploss', 'cos'))

    def run(idx, zin=None, cin=None):
        n_, Z_, ZH_, CH_ = [], [], [], []
        for i in range(0, len(idx), bs):
            tb = torch.tensor(toks[idx[i:i+bs]], device=DEV, dtype=torch.long);sl = slice(i, i+len(tb))
            mk = lambda src, sl=sl:None if src is None else (lambda k, x, s=src:x if not s[1][k] else
                                                             torch.tensor(s[0][sl, k], device=DEV))
            n, Z, ZH, CH, _, _ = episode(lm, loop, tb, zin_fn=mk(zin), cin_fn=mk(cin))
            n_.append(n.cpu().numpy());Z_.append(Z.cpu().numpy());ZH_.append(ZH.cpu().numpy());CH_.append(CH.cpu().numpy())
        return np.concatenate(n_), np.concatenate(Z_), np.concatenate(ZH_), np.concatenate(CH_)

    t0 = time.time();N_own, Z_own, ZH_own, CH_own = run(te);_, Z_tr, _, _ = run(trs)
    rng = np.random.default_rng(5);perm = np.roll(np.arange(len(te)), 1+int(rng.integers(0, len(te)-1)))
    on = [False]+[True]*(KS-1);has_c = chan in ('state', 'fcst')
    Zin_own = np.concatenate([Z_own[:, :1]*0, Z_own[:, :-1]], 1)               # state written into segment k
    Zin_for, CH_for = Zin_own[perm], CH_own[perm]
    st_on = [False, False]+[True]*(KS-2)
    Zin_st = np.concatenate([Z_own[:, :2]*0, Z_own[:, :-2]], 1);CH_st = np.concatenate([CH_own[:, :1], CH_own[:, :-1]], 1)
    N_for, _, _, _ = run(te, zin=(Zin_for, on), cin=(CH_for, on) if has_c else None)
    N_st, _, _, _ = run(te, zin=(Zin_st, st_on), cin=(CH_st, st_on) if has_c else None)
    seg_mean = lambda N:N.mean((0, 2)).tolist()
    res.update(nll_own=seg_mean(N_own), nll_foreign=seg_mean(N_for), nll_stale=seg_mean(N_st),
               bnd_own=float(N_own[:, 1:, :NB].mean()), bnd_foreign=float(N_for[:, 1:, :NB].mean()))
    dfor = (N_for-N_own)[:, 1:].mean((1, 2));dst = (N_st-N_own)[:, 2:].mean((1, 2))
    res['necessity_foreign'] = [float(dfor.mean()), float(dfor.std()/math.sqrt(len(dfor)))]
    res['necessity_stale'] = [float(dst.mean()), float(dst.std()/math.sqrt(len(dst)))]
    arrs = dict(te=te, perm=perm, N_own=N_own.mean(-1), N_bnd=N_own[:, :, :NB].mean(-1), Z_own=Z_own, ZH_own=ZH_own,
                N_for=N_for.mean(-1), N_st=N_st.mean(-1))
    Zt, Zd = torch.tensor(Z_own), torch.tensor(Z_own[perm])
    if has_c:                                                 # channel swap, own state inputs (one step, open loop)
        N_sw, Z_sw, _, _ = run(te, zin=(Zin_own, on), cin=(CH_for, on))
        Zs = torch.tensor(Z_sw);Hd = torch.tensor(ZH_own[perm])
        pull_state = (cos(Zs, Zd)-cos(Zt, Zd))[:, 1:].numpy();pull_fc = (cos(Zs, Hd)-cos(Zt, Hd))[:, 1:].numpy()
        res['swap_pull_state'] = [float(pull_state.mean()), float(pull_state.mean(1).std()/math.sqrt(len(te)))]
        res['swap_pull_forecast'] = [float(pull_fc.mean()), float(pull_fc.mean(1).std()/math.sqrt(len(te)))]
        res['swap_stay'] = float(cos(Zs, Zt)[:, 1:].mean());res['swap_dnll'] = float((N_sw-N_own)[:, 1:].mean())
        arrs.update(N_sw=N_sw.mean(-1), Z_sw=Z_sw)
    res['S_cos'] = cos(torch.tensor(ZH_own[:, 1:]), Zt[:, 1:]).mean(0).tolist()
    Y = Z_own[:, 1:].reshape(-1, DZ);vy = Y.var(0)+1e-9
    r2 = lambda P_:float(1-(((P_.reshape(-1, DZ)-Y)**2).mean(0)/vy).mean())
    mk = Z_tr[:, 1:].mean(0)
    res['R2'] = dict(S=r2(ZH_own[:, 1:]), persistence=r2(Z_own[:, :-1]), segment_mean=r2(np.broadcast_to(mk, Z_own[:, 1:].shape)))
    res['collapse'] = float(cos(Zt.reshape(-1, DZ), Zt.reshape(-1, DZ).mean(0, keepdim=True)).mean())
    def retr(Pk):                                      # top-1 retrieval of z_k among all test states at segment k
        acc = []
        for k in range(1, KS):
            Sm = F.normalize(torch.tensor(Pk[:, k-1]), dim=-1)@F.normalize(Zt[:, k], dim=-1).T
            acc.append(float((Sm.argmax(1) == torch.arange(len(Zt))).float().mean()))
        return float(np.mean(acc))
    res['retrieval'] = dict(S=retr(ZH_own[:, 1:]), persistence=retr(Z_own[:, :-1]), chance=1/len(Zt))
    # corruption at segment 3: the donor's state (and channel) enter once, then the loop runs closed
    c_on = [k == 3 for k in range(KS)]
    N_c, Z_c, ZH_c, _ = run(te, zin=(Zin_for, c_on), cin=(CH_for, c_on) if has_c else None)
    Zc = torch.tensor(Z_c)
    res['identity_own'] = cos(Zc, Zt)[:, 3:].mean(0).tolist();res['identity_donor'] = cos(Zc, Zd)[:, 3:].mean(0).tolist()
    res['nll_corrupt'] = seg_mean(N_c);arrs.update(N_c=N_c.mean(-1), Z_c=Z_c, ZH_c=ZH_c)
    res['track_fresh'] = cos(torch.tensor(ZH_c[:, 4:]), Zc[:, 4:]).mean(0).tolist()       # S(corrupted z_{k-1}) vs z_k
    res['track_stale'] = cos(torch.tensor(ZH_own[:, 4:]), Zc[:, 4:]).mean(0).tolist()     # S(own z_{k-1}) vs z_k
    _, obs = fit_observer(Z_tr, Z_own);res['observer_cos'] = obs.mean(0).tolist();res['baselines'] = baselines(Z_tr, Z_own)
    zz = torch.tensor(Z_own.reshape(-1, DZ)).double();ev = torch.linalg.eigvalsh(torch.cov(zz.T))
    res['pr'] = float(ev.sum()**2/(ev**2).sum());res['words'] = loop_words(lm, loop, Z_own)
    log(a.tag, 'teacher-forced part', f'{time.time()-t0:.0f}s')
    # free generation: own vs corrupted (state + channel from a donor's free run) vs channel-only swap, same randomness
    nf = min(len(te), 16 if SMOKE else a.n_free);fi = te[:nf];fperm = np.roll(np.arange(nf), 1)
    tb = torch.tensor(toks[fi], device=DEV, dtype=torch.long);G, Zg, ZHg = free_run(lm, loop, tb, seed=11)
    don_z = Zg[fperm, 2];don_c = {'state':Zg[fperm, 2], 'fcst':ZHg[fperm, 3]}.get(chan)
    Gc, Zgc, _ = free_run(lm, loop, tb, seed=11, swap=(don_z, don_c))
    runs = dict(own=(G, Zg), corrupt=(Gc, Zgc))
    if has_c:runs['chanswap'] = free_run(lm, loop, tb, seed=11, swap=(None, don_c))[:2]
    Etr = torch.stack([embed(lm, tb[:, k*SEG:(k+1)*SEG]) for k in range(KS)], 1)     # (nf, K, d) the real traces
    mu_e = Etr.reshape(-1, Etr.shape[-1]).mean(0)
    E_own_tr = F.normalize(Etr.mean(1)-mu_e, dim=-1);E_don_tr = E_own_tr[fperm];fr = {}
    for name, (Gx, Zx) in runs.items():
        E = F.normalize(torch.stack([embed(lm, Gx[:, k]) for k in range(KS)], 1)-mu_e, dim=-1)
        To, Td = cos(E, E_own_tr[:, None]), cos(E, E_don_tr[:, None])
        fr[name] = dict(topic_own=To.mean(0).tolist(), topic_donor=Td.mean(0).tolist(),
                        z_vs_own_run=cos(Zx, Zg).mean(0).tolist(), z_vs_donor_run=cos(Zx, Zg[fperm]).mean(0).tolist())
        arrs[f'topic_{name}_own'] = To.cpu().numpy();arrs[f'topic_{name}_don'] = Td.cpu().numpy();arrs[f'G_{name}'] = Gx.cpu().numpy()
    fr['S_cos_free'] = cos(ZHg[:, 1:], Zg[:, 1:]).mean(0).tolist()
    tok = lm.tok;fr['examples'] = []
    for i in range(min(3, nf)):
        ex = dict(trace_start=tok.decode(toks[fi[i], :SEG]), donor_start=tok.decode(toks[fi[fperm[i]], :SEG]))
        for name, (Gx, _) in runs.items():ex[name] = [tok.decode(Gx[i, k].tolist()) for k in (1, 3, 4, 7)]
        fr['examples'].append(ex)
    res['free'] = fr;arrs.update(Zg=Zg.cpu().numpy(), ZHg=ZHg.cpu().numpy(), Zgc=Zgc.cpu().numpy())
    (out/'eval').mkdir(exist_ok=True);atomic_npz(out/'eval'/f'{a.tag}.npz', **arrs);atomic_json(out/'eval'/f'{a.tag}.json', res)
    short = {k:(np.round(v, 3).tolist() if isinstance(v, list) and v and not isinstance(v[0], (list, dict, str)) else v)
             for k, v in res.items() if k not in ('words', 'free', 'baselines')}
    log(a.tag, json.dumps(short), '\n  baselines', json.dumps({k:np.round(v, 3).tolist() for k, v in res['baselines'].items()}),
        '\n  free', json.dumps({k:{kk:np.round(vv, 3).tolist() for kk, vv in v.items()} for k, v in fr.items() if isinstance(v, dict)}),
        f'\n  {time.time()-t0:.0f}s')


@torch.no_grad()
def stage_words(a):
    """J-lens words of the reader axes: the PCA initialisation and every trained learned reader, with the subspace
    overlap between trained and initial readers (1 = same 32-d subspace)."""
    out = pathlib.Path(a.out);lm = load_lm();st = dict(np.load(out/'stats.npz'))
    Dr, W_ = lm.lens.D[LR].float(), lm.lens.words;sdv = torch.tensor(st['sd'], device=DEV)
    top = lambda v:[W_[int(j)] for j in torch.argsort(v, descending=True)[:12]]

    def axes(A):
        ax = torch.as_tensor(A, device=DEV).float()/sdv[None];sc = ax@Dr.T
        return {i:dict(pos=top(sc[i]), neg=top(-sc[i])) for i in range(min(8, DZ))}

    def overlap(A, B):
        Qa, _ = torch.linalg.qr(torch.as_tensor(A, device=DEV).float().T);Qb, _ = torch.linalg.qr(torch.as_tensor(B, device=DEV).float().T)
        return float((Qa.T@Qb).pow(2).sum()/DZ)
    res = dict(init_learned=axes(st['A_learned']), J=axes(st['A_J']))
    for ck in sorted((out/'ckpt').glob('*/loop.pt')):
        sd = torch.load(ck, map_location='cpu')
        if ck.parent.name.split('_')[1] == 'learned' and 'A' in sd:
            res[ck.parent.name] = dict(axes=axes(sd['A']), overlap_with_init=overlap(sd['A'], st['A_learned']),
                                       overlap_with_J=overlap(sd['A'], st['A_J']))
            log(ck.parent.name, 'overlap with init', round(res[ck.parent.name]['overlap_with_init'], 3), 'with J',
                round(res[ck.parent.name]['overlap_with_J'], 3))
    # the clock plane in words: for every learned-reader arm with an evaluation, the plane spanned by the top-2 principal
    # axes of the segment-mean states, mapped back to residual directions through the reader and read with the J lens at
    # the phase of each segment
    clocks = {}
    for ev in sorted((out/'eval').glob('*.npz')):
        tag = ev.stem;ckp = out/'ckpt'/tag/'loop.pt'
        if tag.split('_')[1] != 'learned' or not ckp.exists():continue
        Zo = np.load(ev)['Z_own'];M = Zo.mean(0);C = M-M.mean(0);U = np.linalg.svd(C, full_matrices=False)[2][:2]
        Ph = C@U.T;A = torch.load(ckp, map_location='cpu')['A'].float().to(DEV)
        Ainv = torch.linalg.pinv(A)                                                   # (d, DZ): state -> standardised residual
        rows = []
        for k in range(KS):
            vz = torch.tensor(Ph[k, 0]*U[0]+Ph[k, 1]*U[1], device=DEV, dtype=torch.float32)
            dres = (Ainv@vz)*sdv;sc = dres@Dr.T
            rows.append(dict(segment=k, angle=float(np.degrees(np.arctan2(Ph[k, 1], Ph[k, 0]))), words=top(sc)[:8]))
        clocks[tag] = rows
        log('clock words', tag, ' | '.join(f"{r['segment']}:{int(r['angle'])} {','.join(r['words'][:4])}" for r in rows))
    res['clock_words'] = clocks
    atomic_json(out/'reader_words.json', res)


def main():
    p = argparse.ArgumentParser();p.add_argument('stage', choices=['gen', 'train', 'eval', 'gate', 'words'])
    p.add_argument('--out', required=True);p.add_argument('--cache', default='/workspace/data')
    p.add_argument('--n', type=int, default=2400);p.add_argument('--gen_bs', type=int, default=128)
    p.add_argument('--chan', default='const', choices=['cut', 'const', 'state', 'fcst'])
    p.add_argument('--reader', default='learned', choices=['learned', 'J', 'random'])
    p.add_argument('--lam', type=float, default=0.);p.add_argument('--steps', type=int, default=600)
    p.add_argument('--bs', type=int, default=10);p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--seed', type=int, default=0);p.add_argument('--tag', default=None)
    p.add_argument('--n_free', type=int, default=128);p.add_argument('--ploss', default='cos', choices=['cos', 'nmse', 'nce', 'ncex'])
    a = p.parse_args()
    {'gen':stage_gen, 'train':stage_train, 'eval':stage_eval, 'gate':stage_gate, 'words':stage_words}[a.stage](a)


if __name__ == '__main__':
    main()
