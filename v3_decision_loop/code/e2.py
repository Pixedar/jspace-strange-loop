"""Experiment 2 (+ the trajectories analysed in Experiment 3): the J-space loop (protocol section 4).
Loop map at the decision position D of a running context:  h_Lw <- b + write(code_t);  code_{t+1} = decompose(h_Lr - mu).
One step recomputes position D from the bottom with the prompt's KV cache fixed (b is therefore constant), so the fed-back
code is the only thing that changes between steps.
    python e2.py --model qwen3-8b --stages calib,A,BC,pert [--n 100] [--T 64]"""
import argparse, json, math, pathlib, time, random
import numpy as np, torch
from common import LM, OUT, log, atomic_npz, atomic_json, seed_all, top_tokens
from tasks import build_all, user_text, gen_f3

GRID = (.25, .5, 1., 2., 4.)
COND_A1 = [('nofb', 0.)]+[('J', g) for g in GRID]
CTRL = ('J', 'Jpb', 'dense', 'pca', 'rand', 'shuf', 'foreign', 'tshuf')


def cond_a2(g1, g2):
    return [('nofb', 0.)]+[(c, g) for c in CTRL for g in (g1, g2)]


def select_gains(out):
    """Pre-declared rule (pilot revision, before 8B loop data): g1 = smallest grid gain at which the J loop's final code
    shares < 80 % of its atoms with the natural (no-feedback) code on average over std contexts; g2 = next grid gain."""
    ov = {}
    for f in sorted((out/'A1').glob('A1_std_*.npz')):
        z = np.load(f);rows = json.loads(f.with_suffix('.json').read_text())['rows'];T = z['idx'].shape[0]-1
        nat = {r['unit']:set(z['idx'][1, i].tolist()) for i, r in enumerate(rows) if r['cond'] == 'nofb'}
        for i, r in enumerate(rows):
            if r['cond'] == 'J':ov.setdefault(r['g'], []).append(len(set(z['idx'][T, i].tolist())&nat[r['unit']])/25)
    means = {g:float(np.mean(v)) for g, v in sorted(ov.items())};gs = list(means)
    q = [g for g in gs if means[g] < .8]
    g1 = q[0] if q else gs[-2];g2 = gs[min(gs.index(g1)+1, len(gs)-1)]
    atomic_json(out/'gains.json', dict(g1=g1, g2=g2, final_overlap_with_natural=means))
    return g1, g2
COND_BC = ['B', 'C1', 'C2', 'R', 'F']
N_ACT = 9                       # 0 none, 1-4 amplify the i-th strongest concept, 5-8 suppress it (norm preserved)
WARM = 8                        # random actions while the self-model starts learning


# ----------------------------------------------------------------------------------------------------------------
# contexts
# ----------------------------------------------------------------------------------------------------------------
def make_contexts(n, split='test', seed=20261011):
    """Per scenario three contexts A / B / N (evidence for reading A, for reading B, none) for F2 and F3, plus F5."""
    data = build_all('main', seed)
    ctx = []
    f2 = [it for it in data['F2'] if it['split'] == split and it['challenge'] == 'none']
    sids = list(dict.fromkeys(it['sid'] for it in f2))[:n]
    for s in sids:
        its = {(it['evidence'], it['edir']):it for it in f2 if it['sid'] == s}
        for tag, k in (('A', ('strong', 0)), ('B', ('strong', 1)), ('N', ('none', None))):
            it = its[k];ctx.append(dict(fam='F2', sid=s, sub=it['sub'], tag=tag, text=user_text(it)[0], cands=it['cands'],
                                        truth={'A':0, 'B':1, 'N':-1}[tag]))
    f3 = [it for it in data['F3'] if it['split'] == split and it['challenge'] == 'none' and it['binding'] == 0
          and it['qkind'] == 'belief']
    sids = list(dict.fromkeys(it['sid'] for it in f3))[:n]
    for s in sids:
        its = {it['evidence']:it for it in f3 if it['sid'] == s}
        for tag, ev in (('A', 'none'), ('B', 'tb')):
            it = its[ev];ctx.append(dict(fam='F3', sid=s, sub='tomi', tag=tag, text=user_text(it)[0], cands=it['cands'],
                                         truth={'A':0, 'B':1}[tag]))
        it = its['none'];f = it['fillers']            # ambiguous: no sentence about whether the character saw the move
        amb = dict(it, core=it['core'].replace(f"{f['n1']} left the room. ", ''), evid='')
        ctx.append(dict(fam='F3', sid=s, sub='tomi', tag='N', text=user_text(amb)[0], cands=it['cands'], truth=-1))
    # F6 concept memory: the remembered item is a verbalizable concept (contrast with F2/F3, where the answer is a
    # binding / pointer that pilot H1 found outside the J component)
    CAT = {'color':'red blue green yellow black white brown pink gray purple'.split(),
           'animal':'cat dog horse bird fish lion tiger bear wolf mouse rabbit snake'.split(),
           'fruit':'apple banana grape lemon cherry peach pear mango'.split(),
           'object':'box ball book chair table bag key hat pen coin ring clock'.split()}
    INTRO = ['Here is a note for you.', 'Keep this in mind.', 'A friend left you a message.', 'Read the card carefully.',
             'Someone whispered a secret.', 'There is a label on the jar.']
    rng = random.Random(seed+(0 if split == 'test' else 1 if split == 'val' else 2));cats = list(CAT)
    for s_ in range(n):
        c = cats[s_ % 4];w1, w2 = rng.sample(CAT[c], 2);intro = INTRO[(s_ % 2)+(4 if split == 'test' else 0) if split == 'test' else s_ % 4]
        sid = f'F6-{split}-{s_}'
        for tag, body, tr in (('A', f'{intro} The secret {c} is {w1}.', 0), ('B', f'{intro} The secret {c} is {w2}.', 1),
                              ('N', f'{intro} The secret {c} is hidden.', -1)):
            ctx.append(dict(fam='F6', sid=sid, sub=c, tag=tag, text=body+'\n'+f'Question: What is the secret {c}? Answer with one word.',
                            cands=[w1, w2], truth=tr))
    f5 = [it for it in data['F5'] if it['split'] == split][:n]
    for it in f5:
        ctx.append(dict(fam='F5', sid=it['sid'], sub=it['sub'], tag='S', text=user_text(it)[0], cands=it['cands'],
                        truth=int(it['answer'])))
    return ctx


# ----------------------------------------------------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------------------------------------------------
class Calib:
    def __init__(self, d):
        for k, v in d.items():setattr(self, k, v)


@torch.no_grad()
def full_reads(lm, texts, layers, bs=48):
    out = {l:[] for l in layers};logits = []
    for i in range(0, len(texts), bs):
        ids, att, _ = lm.encode([lm.chat(t) for t in texts[i:i+bs]])
        lm.clear()
        for l in layers:lm.read[l] = torch.full((ids.shape[0], 1), ids.shape[1]-1, device=lm.dev)
        lg = lm.forward(ids, att)[:, -1];logits.append(lg)
        for l in layers:out[l].append(lm.store[l][:, 0])
        lm.clear()
    return {l:torch.cat(v) for l, v in out.items()}, torch.cat(logits)


def stage_calib(lm, out, Lr, Lw, n_cal=192):
    seed_all(0)
    cal = make_contexts(n_cal//3, split='train')
    random.Random(0).shuffle(cal);cal = cal[:n_cal]
    R, _ = full_reads(lm, [c['text'] for c in cal], [Lr, Lw])
    mu_r, mu_w = R[Lr].mean(0), R[Lw].mean(0)
    xr, xw = R[Lr]-mu_r, R[Lw]-mu_w
    _, _, Jr = lm.lens.decompose(Lr, xr, 25);_, _, Jw = lm.lens.decompose(Lw, xw, 25)
    m_J = Jr.norm(dim=-1).median();n_ref = Jw.norm(dim=-1).median();m_dense = xr.norm(dim=-1).median()
    U, S, V = torch.linalg.svd(xr-xr.mean(0), full_matrices=False);V25 = V[:25]
    m_pca = (xr@V25.T).norm(dim=-1).median()
    U2, S2, V2 = torch.linalg.svd(Jr-Jr.mean(0), full_matrices=False);Z = V2[:32];zmu = Jr.mean(0)
    zs = ((Jr-zmu)@Z.T).std(0).clamp_min(1e-6)
    Jw_, Jr_ = lm.lens.J[Lw].float(), lm.lens.J[Lr].float()
    A = Jw_.T@Jw_;lam = 1e-2*A.diagonal().mean()
    P = torch.linalg.solve(A+lam*torch.eye(A.shape[0], device=lm.dev), Jw_.T@Jr_)      # u = P v (column convention)
    m_w = xw.norm(dim=-1).median();nJw = n_ref
    n_ref = m_J            # gain unit (pilot revision): g = 1 re-broadcasts the workspace at the strength it was read
    C = dict(Lr=Lr, Lw=Lw, mu_r=mu_r, mu_w=mu_w, m_J=m_J, n_ref=n_ref, m_dense=m_dense, V25=V25, m_pca=m_pca, Z=Z, zmu=zmu,
             zs=zs, P=P, m_w=m_w, nJw=nJw)
    atomic_npz(out/'calib.npz', **{k:(v.cpu().numpy() if torch.is_tensor(v) else np.array(v)) for k, v in C.items()})
    info = dict(n_cal=len(cal), m_J=float(m_J), n_ref=float(n_ref), m_dense=float(m_dense), m_pca=float(m_pca),
                m_w=float(m_w), nJw=float(nJw), write_to_stream_at_g1=float(m_J/m_w),
                pca25_var=float((S[:25]**2).sum()/(S**2).sum()), z32_var=float((S2[:32]**2).sum()/(S2**2).sum()),
                j_share_r=float((Jr.norm(dim=-1)**2/xr.norm(dim=-1)**2).median()))
    log('calib', info)
    return Calib(C), info


def load_calib(lm, path):
    z = np.load(path)
    return Calib({k:(torch.tensor(z[k], device=lm.dev) if z[k].ndim else (int(z[k]) if k in ('Lr', 'Lw') else
                                                                             torch.tensor(float(z[k]), device=lm.dev)))
                  for k in z.files})


# ----------------------------------------------------------------------------------------------------------------
# self-model (one per independent replicate system = M-group)
# ----------------------------------------------------------------------------------------------------------------
class SelfModel:
    """Predicts y_{t+1} = [z_{t+1}; logodds_{t+1}] from z_t and the action: y = A [z;1] + b_a (shared dynamics,
    action-specific offsets), exponentially-forgetting ridge (0.98). Error models (fast 0.9 / slow 0.99 decayed
    regressions of the squared normalized error on [z;1;onehot(a)]) give expected error and learning progress."""

    def __init__(self, dz, dev, lam=0.98, rho=1.0):
        self.dz = dz;self.dev = dev;self.lam = lam;self.rho = rho;p = dz+1+N_ACT;self.p = p
        self.Sxx = torch.zeros(p, p, device=dev, dtype=torch.float64);self.Sxy = torch.zeros(p, dz+1, device=dev, dtype=torch.float64)
        self.W = torch.zeros(p, dz+1, device=dev, dtype=torch.float64)
        self.E = {k:[torch.zeros(p, p, device=dev, dtype=torch.float64), torch.zeros(p, device=dev, dtype=torch.float64)]
                  for k in ('fast', 'slow')}
        self.frozen = False

    def feats(self, z, a):
        oh = torch.nn.functional.one_hot(a, N_ACT).double()
        return torch.cat([z.double(), torch.ones(z.shape[0], 1, device=self.dev, dtype=torch.float64), oh], 1)

    def predict_all(self, z):
        """(B, N_ACT, dz+1) predictions for every action."""
        B = z.shape[0]
        X = torch.stack([self.feats(z, torch.full((B,), a, device=self.dev, dtype=torch.long)) for a in range(N_ACT)], 1)
        return X@self.W, X

    def expected_err(self, X, which):
        Sxx, Sxy = self.E[which]
        w = torch.linalg.solve(Sxx+self.rho*torch.eye(self.p, device=self.dev, dtype=torch.float64), Sxy)
        return X@w

    def update(self, z, a, y):
        if self.frozen:return None
        X = self.feats(z, a);Y = y.double()
        err = ((X@self.W-Y)**2).mean(1)
        self.Sxx = self.lam*self.Sxx+X.T@X;self.Sxy = self.lam*self.Sxy+X.T@Y
        self.W = torch.linalg.solve(self.Sxx+self.rho*torch.eye(self.p, device=self.dev, dtype=torch.float64), self.Sxy)
        for k, lam in (('fast', .9), ('slow', .99)):
            S = self.E[k];S[0] = lam*S[0]+X.T@X;S[1] = lam*S[1]+X.T@err
        return err


# ----------------------------------------------------------------------------------------------------------------
# the loop runner
# ----------------------------------------------------------------------------------------------------------------
class Loop:
    def __init__(self, lm, C, T=64):
        self.lm = lm;self.C = C;self.T = T;self.Lr, self.Lw = C.Lr, C.Lw
        self.D = lm.lens.D[self.Lr];self.nD = self.D.shape[0]
        self.word_idx = lm.lens.word_index

    def decomp(self, h):
        x = h-self.C.mu_r;idx, coef, rec = self.lm.lens.decompose(self.Lr, x, 25)
        return idx, coef, rec, x

    def zof(self, rec):
        return ((rec-self.C.zmu)@self.C.Z.T)/self.C.zs

    def synth_w(self, idx, coef):
        """Concept transport with a constant-norm broadcast: the loop carries content (which concepts, in which
        proportions), never runaway amplitude (pilot: amplitude-proportional writes diverged for g >= 0.5)."""
        u = self.lm.lens.synth(self.Lw, idx, coef)
        return u/u.norm(dim=-1, keepdim=True).clamp_min(1e-6)*self.C.n_ref

    def pull(self, v, m):
        u = v@self.C.P.T
        return u/u.norm(dim=-1, keepdim=True).clamp_min(1e-6)*self.C.n_ref

    @torch.no_grad()
    def xcode(self, h, logits, kinds):
        """Exploratory arm X (amendment 11): codes built from the system's own linear self-forecast. The J-lens readout
        at the read layer is the workspace's linear forecast of the output; 'inc' broadcasts the dictionary words the
        network outputs beyond that forecast (log p_out - log p_lens), 'out' the network's top output words (plain
        output feedback, the ordinary-explanation control), 'lens' the forecast's own top words."""
        lm = self.lm;ids = lm.lens.ids
        lp_out = torch.log_softmax(logits.float(), -1)[:, ids]
        lj = lm.model.lm_head(lm.model.model.norm((h.float()@lm.lens.J[self.Lr].T).to(lm.model.dtype))).float()
        lp_J = torch.log_softmax(lj, -1)[:, ids]
        idx = torch.zeros(h.shape[0], 25, dtype=torch.long, device=h.device);coef = torch.zeros(h.shape[0], 25, device=h.device)
        for k in set(kinds):
            r = torch.tensor([i for i, kk in enumerate(kinds) if kk == k], device=h.device)
            sc = {'inc':lp_out-lp_J, 'out':lp_out, 'lens':lp_J}[k][r]
            v, i_ = sc.topk(25, -1);idx[r] = i_;coef[r] = torch.softmax(v, -1)
        return idx, coef

    @torch.no_grad()
    def init_codes(self, texts, bs=96):
        """Natural state of each context computed through the same prefix-cache + single-step path as the loop."""
        lm = self.lm;H, L = [], []
        for i in range(0, len(texts), bs):
            ids, att, _ = lm.encode([lm.chat(t) for t in texts[i:i+bs]])
            pre_ids, pre_att, last = ids[:, :-1], att[:, :-1], ids[:, -1:]
            posid = (pre_att.long().cumsum(-1)-1).clamp_min(0);lm.clear()
            o = lm.model(input_ids=pre_ids, attention_mask=pre_att, position_ids=posid, use_cache=True, logits_to_keep=1)
            cache = o.past_key_values;lm.read[self.Lr] = None
            o = lm.model(input_ids=last, attention_mask=torch.cat([pre_att, torch.ones_like(last)], 1),
                         position_ids=pre_att.long().sum(-1, keepdim=True), past_key_values=cache, use_cache=True, logits_to_keep=1)
            H.append(lm.store[self.Lr][:, -1]);L.append(o.logits[:, -1].float());lm.clear();del cache
        idx, coef, rec, x = self.decomp(torch.cat(H))
        return dict(idx=idx, coef=coef, rec=rec, x=x, logits=torch.cat(L))

    def cand_ids(self, cands, fam):
        if fam == 'F5':return None
        return self.lm.variant_ids(cands[0]), self.lm.variant_ids(cands[1])

    @torch.no_grad()
    def run(self, rows, ctxs, save_h=None, pert=None, yoke=None, seed=0):
        """rows: list of dicts with keys ctx (run context index), init (init context index), cond, g, mg (M-group id or
        None), partner (row index for 'foreign'/'trans'). Returns per-step records."""
        lm, T = self.lm, self.T;B = len(rows);dev = lm.dev
        gen = torch.Generator(device=dev);gen.manual_seed(seed);rng = np.random.default_rng(seed)
        run_texts = [ctxs[r['ctx']]['text'] for r in rows]
        # initial codes from the init contexts (unique)
        uinit = sorted(set(r['init'] for r in rows));pos_of = {c:i for i, c in enumerate(uinit)}
        I = self.init_codes([ctxs[c]['text'] for c in uinit]);ii = torch.tensor([pos_of[r['init']] for r in rows], device=dev)
        idx, coef, rec, x = I['idx'][ii].clone(), I['coef'][ii].clone(), I['rec'][ii].clone(), I['x'][ii].clone()
        # prefix caches of the running contexts (prompt minus its last token)
        ids, att, _ = lm.encode([lm.chat(t) for t in run_texts])
        pre_ids, pre_att, last = ids[:, :-1], att[:, :-1], ids[:, -1:]
        posid = (pre_att.long().cumsum(-1)-1).clamp_min(0)
        lm.clear()
        out = lm.model(input_ids=pre_ids, attention_mask=pre_att, position_ids=posid, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values;plen = pre_ids.shape[1];del out
        step_att = torch.cat([pre_att, torch.ones_like(last)], 1);step_pos = pre_att.long().sum(-1, keepdim=True)
        conds = [r['cond'] for r in rows];g = torch.tensor([r['g'] for r in rows], device=dev)[:, None]
        cm = {c:torch.tensor([cc == c for cc in conds], device=dev) for c in set(conds)}
        partner = torch.tensor([r.get('partner', i) if r.get('partner') is not None else i for i, r in enumerate(rows)], device=dev)
        # M-groups
        mgs = sorted(set(r['mg'] for r in rows if r.get('mg') is not None))
        M = {m:SelfModel(32, dev) for m in mgs}
        mrows = {m:torch.tensor([i for i, r in enumerate(rows) if r.get('mg') == m], device=dev) for m in mgs}
        mcond = {m:rows[int(mrows[m][0])]['cond'] for m in mgs}
        cid = [self.cand_ids(ctxs[r['ctx']]['cands'], ctxs[r['ctx']]['fam']) for r in rows]
        digit_ids = [lm.tok.encode(str(d), add_special_tokens=False)[0] for d in range(10)]
        hist_idx, hist_coef = [idx.clone()], [coef.clone()]
        rec_ = dict(idx=np.zeros((T+1, B, 25), np.int32), coef=np.zeros((T+1, B, 25), np.float16),
                    z=np.zeros((T+1, B, 32), np.float16), lo=np.zeros((T+1, B), np.float32),
                    digits=np.zeros((T+1, B, 10), np.float16), top=np.zeros((T+1, B, 5), np.int32),
                    wnorm=np.zeros((T, B), np.float32), act=np.full((T, B), -1, np.int8),
                    perr=np.full((T, B), np.nan, np.float32), pred=np.full((T, B, 33), np.nan, np.float16),
                    recnorm=np.zeros((T+1, B), np.float32), xnorm=np.zeros((T+1, B), np.float32))
        H = np.zeros((T+1, len(save_h), lm.d), np.float16) if save_h else None

        def record(t, idx, coef, rec, x, logits):
            rec_['idx'][t] = idx.cpu().numpy();rec_['coef'][t] = coef.cpu().numpy()
            rec_['z'][t] = self.zof(rec).cpu().numpy();rec_['recnorm'][t] = rec.norm(dim=-1).cpu().numpy()
            rec_['xnorm'][t] = x.norm(dim=-1).cpu().numpy()
            lp = torch.log_softmax(logits.float(), -1)
            rec_['digits'][t] = lp[:, digit_ids].cpu().numpy();rec_['top'][t] = lp.topk(5, -1).indices.cpu().numpy()
            lo = np.zeros(B, np.float32)
            for b in range(B):
                if cid[b] is not None:
                    a_, b_ = cid[b];sa, sb = set(a_), set(b_)
                    a_ = [i for i in a_ if i not in sb];b_ = [i for i in b_ if i not in sa]
                    lo[b] = (torch.logsumexp(lp[b, a_], 0)-torch.logsumexp(lp[b, b_], 0)).item()
            rec_['lo'][t] = lo
            if save_h:H[t] = (x+self.C.mu_r)[save_h].cpu().numpy()
        record(0, idx, coef, rec, x, I['logits'][ii])
        z = self.zof(rec);lo_t = torch.tensor(rec_['lo'][0], device=dev)
        xkinds = [c if c in ('inc', 'out', 'lens') else 'out' for c in conds];has_x = any(c in ('inc', 'out', 'lens') for c in conds)
        if has_x:
            xi, xc = self.xcode(x+self.C.mu_r, I['logits'][ii], xkinds);rec_['widx'] = np.zeros((T+1, B, 25), np.int32)
            rec_['widx'][0] = xi.cpu().numpy()
        for t in range(T):
            W = torch.zeros(B, lm.d, device=dev)
            c_use = coef.clone();i_use = idx.clone()
            # --- actions for self-model conditions
            act = torch.zeros(B, dtype=torch.long, device=dev)
            if mgs:
                for m in mgs:
                    rr = mrows[m];cond = mcond[m];Mm = M[m]
                    preds, X = Mm.predict_all(z[rr])
                    if cond == 'Y':a = torch.tensor(yoke[m][:, t], device=dev, dtype=torch.long)
                    elif t < WARM or cond == 'R':a = torch.randint(0, N_ACT, (len(rr),), device=dev, generator=gen)
                    else:
                        if cond == 'F' and not Mm.frozen:Mm.frozen = True
                        if cond == 'B':a = Mm.expected_err(X, 'slow').argmin(1)
                        elif cond in ('C1', 'F'):a = Mm.expected_err(X, 'slow').argmax(1)
                        elif cond == 'C2':a = (Mm.expected_err(X, 'slow')-Mm.expected_err(X, 'fast')).argmax(1)
                    if pert is not None and pert['kind'] == 'self' and t == pert['t']:
                        e = Mm.expected_err(X, 'slow');a = e.argmin(1) if cond in ('C1', 'F', 'C2') else e.argmax(1)
                    act[rr] = a
                    rec_['pred'][t, rr.cpu().numpy()] = preds[torch.arange(len(rr), device=dev), a].float().cpu().numpy()
                o = torch.argsort(-coef, 1)
                for k in range(4):
                    sel = act == 1+k;r = torch.nonzero(sel)[:, 0]
                    if len(r):c_use[r, o[r, k]] *= 2.
                    sel = act == 5+k;r = torch.nonzero(sel)[:, 0]
                    if len(r):c_use[r, o[r, k]] = 0.
                chg = act > 0
                if chg.any():
                    c_use[chg] *= (coef[chg].norm(dim=-1, keepdim=True)/c_use[chg].norm(dim=-1, keepdim=True).clamp_min(1e-6))
            rec_['act'][t] = act.cpu().numpy()
            # --- perturbations of the code at step pert['t']
            if pert is not None and t == pert['t'] and pert['kind'] in ('sem', 'trans', 'rand'):
                pm = pert['rows']
                if pert['kind'] == 'rand':
                    i_use[pm] = torch.randint(0, self.nD, i_use[pm].shape, device=dev, generator=gen)
                elif pert['kind'] == 'trans':
                    i_use[pm] = idx[partner[pm]];c_use[pm] = coef[partner[pm]]
                else:
                    for b in pm.tolist():
                        ca, cb = ctxs[rows[b]['ctx']]['cands'];wa, wb = self.word_idx.get(ca), self.word_idx.get(cb)
                        if wa is None or wb is None:continue
                        ib = i_use[b].tolist()
                        if wa in ib or wb in ib:
                            i_use[b] = torch.tensor([wb if v == wa else wa if v == wb else v for v in ib], device=dev)
                        else:
                            fav = wb if rec_['lo'][t, b] > 0 else wa;j = int(torch.argmin(c_use[b]))
                            i_use[b, j] = fav;c_use[b, j] = c_use[b].max()
            # --- writes per condition
            for c, m in cm.items():
                if c == 'nofb' or not m.any():continue
                r = torch.nonzero(m)[:, 0]
                if c in ('J', 'B', 'C1', 'C2', 'R', 'F', 'Y'):W[r] = self.synth_w(i_use[r], c_use[r])
                elif c == 'rand':W[r] = self.synth_w(torch.randint(0, self.nD, i_use[r].shape, device=dev, generator=gen), c_use[r])
                elif c == 'shuf':
                    p_ = torch.argsort(torch.rand(c_use[r].shape, device=dev, generator=gen), 1)
                    W[r] = self.synth_w(i_use[r], torch.gather(c_use[r], 1, p_))
                elif c == 'foreign':W[r] = self.synth_w(idx[partner[r]], coef[partner[r]])
                elif c == 'tshuf':
                    s = [int(rng.integers(0, t)) if t > 0 else 0 for _ in range(len(r))]
                    W[r] = self.synth_w(torch.stack([hist_idx[s_][rr] for s_, rr in zip(s, r.tolist())]),
                                        torch.stack([hist_coef[s_][rr] for s_, rr in zip(s, r.tolist())]))
                elif c == 'Jpb':W[r] = self.pull(rec[r], self.C.m_J)
                elif c == 'dense':W[r] = self.pull(x[r], self.C.m_dense)
                elif c == 'pca':W[r] = self.pull((x[r]@self.C.V25.T)@self.C.V25, self.C.m_pca)
                elif c in ('inc', 'out', 'lens'):W[r] = self.synth_w(xi[r], xc[r])
            W = W*g
            if pert is not None and t == pert['t'] and pert['kind'] == 'eps':
                pm = pert['rows'];nz = torch.randn(len(pm), lm.d, device=dev, generator=gen)
                W[pm] = W[pm]+nz/nz.norm(dim=-1, keepdim=True)*0.01*self.C.n_ref
            rec_['wnorm'][t] = W.norm(dim=-1).cpu().numpy()
            # --- one recurrent step: write at Lw, read at Lr
            lm.clear()
            lm.edit[self.Lw] = lambda h, W=W:h+W[:, None, :].to(h.dtype)
            lm.read[self.Lr] = None
            o_ = lm.model(input_ids=last, attention_mask=step_att, position_ids=step_pos, past_key_values=cache,
                          use_cache=True, logits_to_keep=1)
            hr = lm.store[self.Lr][:, -1];logits = o_.logits[:, -1].float();lm.clear()
            cache.crop(plen)
            idx, coef, rec, x = self.decomp(hr)
            record(t+1, idx, coef, rec, x, logits)
            hist_idx.append(idx.clone());hist_coef.append(coef.clone())
            if has_x:
                xi, xc = self.xcode(hr, logits, xkinds);rec_['widx'][t+1] = xi.cpu().numpy()
            z_new = self.zof(rec);lo_new = torch.tensor(rec_['lo'][t+1], device=dev)
            for m in mgs:
                rr = mrows[m];y = torch.cat([z_new[rr], lo_new[rr, None]/4.], 1)
                e = M[m].update(z[rr], act[rr], y)
                if e is None:e = ((torch.tensor(rec_['pred'][t, rr.cpu().numpy()], device=dev).double()-y.double())**2).mean(1)
                rec_['perr'][t, rr.cpu().numpy()] = e.float().cpu().numpy()
            z, lo_t = z_new, lo_new
        del cache;torch.cuda.empty_cache()
        if save_h:rec_['H'] = H
        return rec_


# ----------------------------------------------------------------------------------------------------------------
def chunked(xs, n):
    return [xs[i:i+n] for i in range(0, len(xs), n)]


def ctx_units(ctxs, kind):
    """kind 'mem': (run=N, init=A|B) per F2/F3 scenario; 'std': run = init for every context."""
    units = []
    if kind == 'mem':
        by = {}
        for i, c in enumerate(ctxs):by.setdefault((c['fam'], c['sid']), {})[c['tag']] = i
        for (fam, sid), d in by.items():
            if 'N' in d:
                units.append(dict(ctx=d['N'], init=d['A'], other=d['B'], sid=sid, fam=fam, tag='A'))
                units.append(dict(ctx=d['N'], init=d['B'], other=d['A'], sid=sid, fam=fam, tag='B'))
    else:
        by = {}
        for i, c in enumerate(ctxs):by.setdefault((c['fam'], c['sid']), {})[c['tag']] = i
        for i, c in enumerate(ctxs):
            d = by[(c['fam'], c['sid'])];oth = d.get({'A':'B', 'B':'A'}.get(c['tag'], ''), None)
            units.append(dict(ctx=i, init=i, other=oth, sid=c['sid'], fam=c['fam'], tag=c['tag']))
    return units


def save_run(path, rec, rows, units, extra=None):
    meta = dict(rows=rows, units=units, **(extra or {}))
    atomic_npz(path.with_suffix('.npz'), **rec);atomic_json(path.with_suffix('.json'), meta)


def stage_A(loop, ctxs, out, per_pass=32, conds=None, name='A1', kinds=('std', 'mem')):
    for kind in kinds:
        units = ctx_units(ctxs, kind)
        for pi, uu in enumerate(chunked(units, per_pass)):
            f = out/f'{name}_{kind}_{pi:03d}'
            if f.with_suffix('.npz').exists():continue
            rows = [];jrow = {}
            for ui, u in enumerate(uu):
                for c, g in conds:
                    if c == 'J':jrow[(ui, g)] = len(rows)
                    rows.append(dict(ctx=u['ctx'], init=u['init'], cond=c, g=g, unit=ui))
            for r in rows:                                   # foreign: J row of another scenario, same gain
                if r['cond'] == 'foreign':
                    cands = [ui for ui in range(len(uu)) if uu[ui]['sid'] != uu[r['unit']]['sid']]
                    r['partner'] = jrow[(cands[(r['unit']*7) % len(cands)], r['g'])] if cands else None
            save_h = [i for i, r in enumerate(rows) if r['cond'] in ('J', 'nofb', 'dense') and (name == 'A1' or r['g'] == conds[1][1])]
            t0 = time.time();rec = loop.run(rows, ctxs, save_h=save_h, seed=pi)
            save_run(f, rec, rows, uu, dict(save_h=save_h, kind=kind))
            log(f.name, len(rows), 'rows', f'{time.time()-t0:.0f}s')


def stage_BC(loop, ctxs, out, per_pass=32, seeds=(0, 1, 2), gains=(1., 2.), kinds=('std', 'mem')):
    """Each chunk of 32 contexts is one pass holding every seed x condition x gain; every (seed, condition, gain) has its
    own self-model (an independent replicate system of 32 concurrently running contexts). F5 is benchmark-only."""
    g_lo = gains[0]
    for kind in kinds:
        units = [u for u in ctx_units(ctxs, kind) if ctxs[u['ctx']]['fam'] != 'F5']
        chunks = chunked(units, per_pass)
        for pi, uu in enumerate(chunks):
            f = out/f'BC_{kind}_{pi:03d}'
            if f.with_suffix('.npz').exists():continue
            rows = [];mg = 0
            for sd in seeds:
                for c in COND_BC:
                    for g in gains:
                        for ui, u in enumerate(uu):rows.append(dict(ctx=u['ctx'], init=u['init'], cond=c, g=g, mg=mg, unit=ui, seed=sd))
                        mg += 1
            save_h = [i for i, r in enumerate(rows) if r['g'] == g_lo and r['cond'] in ('B', 'C1', 'C2', 'R') and r['seed'] == seeds[0]]
            t0 = time.time();rec = loop.run(rows, ctxs, save_h=save_h, seed=1000*pi+(0 if kind == 'std' else 500))
            save_run(f, rec, rows, uu, dict(save_h=save_h, kind=kind))
            log(f.name, len(rows), 'rows', f'{time.time()-t0:.0f}s')
        # yoked: C1 action sequences from the next chunk (other contexts), same seed and gain, replayed without a model
        for pi, uu in enumerate(chunks):
            f = out/f'BCY_{kind}_{pi:03d}'
            if f.with_suffix('.npz').exists():continue
            src = out/f'BC_{kind}_{(pi+1) % len(chunks):03d}'
            z = np.load(src.with_suffix('.npz'));meta = json.loads(src.with_suffix('.json').read_text())
            rows = [];yoke = {};mg = 0
            for sd in seeds:
                for g in gains:
                    sr = [i for i, r in enumerate(meta['rows']) if r['cond'] == 'C1' and r['g'] == g and r['seed'] == sd]
                    acts = z['act'][:, sr].T
                    for ui, u in enumerate(uu):rows.append(dict(ctx=u['ctx'], init=u['init'], cond='Y', g=g, mg=mg, unit=ui, seed=sd))
                    yoke[mg] = np.stack([acts[ui % len(acts)] for ui in range(len(uu))]);mg += 1
            t0 = time.time();rec = loop.run(rows, ctxs, yoke=yoke, seed=5000*pi+(0 if kind == 'std' else 500))
            save_run(f, rec, rows, uu, dict(kind=kind, yoke_src=src.name))
            log(f.name, len(rows), 'rows', f'{time.time()-t0:.0f}s')


def stage_pert(loop, ctxs, out, per_pass=32, t_p=16, gains=(1., 2.)):
    units = [u for u in ctx_units(ctxs, 'std') if u['other'] is not None]     # A/B contexts with an opposite partner
    for pi, uu in enumerate(chunked(units, per_pass)):
        f = out/f'P_{pi:03d}'
        if not f.with_suffix('.npz').exists():
            rows = [];base = {}
            for kind in ('base', 'eps', 'sem', 'trans', 'rand'):
                for g in gains:
                    for ui, u in enumerate(uu):
                        base.setdefault((kind, g), []).append(len(rows))
                        rows.append(dict(ctx=u['ctx'], init=u['init'], cond='J', g=g, unit=ui, pert=kind))
            # partner for 'trans': the base row of the opposite-evidence context if it is in this chunk, else itself
            uidx = {(u['ctx']):ui for ui, u in enumerate(uu)}
            for r in rows:
                o = uu[r['unit']]['other'];r['partner'] = base[('base', r['g'])][uidx[o]] if o in uidx else None
            rec_all = None
            # one run per perturbation kind keeps the perturbation step exact; base rows are rerun as a replay check
            for kind in ('eps', 'sem', 'trans', 'rand'):
                pm = torch.tensor([i for i, r in enumerate(rows) if r['pert'] == kind], device=loop.lm.dev)
                rec = loop.run(rows, ctxs, pert=dict(kind=kind, t=t_p, rows=pm), seed=7000+pi)
                if rec_all is None:rec_all = {k:v.copy() for k, v in rec.items()}
                sel = pm.cpu().numpy()
                for k in rec:
                    if k == 'H':continue
                    if rec[k].ndim >= 2:rec_all[k][:, sel] = rec[k][:, sel]
            save_run(f, rec_all, rows, uu, dict(t_p=t_p))
            log(f.name, len(rows), 'rows')
    # system-level perturbations of self-model conditions: twin passes with every row perturbed
    units_bc = units[:per_pass*4]
    for pi, uu in enumerate(chunked(units_bc, per_pass)):
        for kind in ('none', 'eps', 'self', 'trans'):
            f = out/f'PBC_{pi:03d}_{kind}'
            if f.with_suffix('.npz').exists():continue
            rows = [];mg = 0
            for c in ('B', 'C1', 'C2'):
                for ui, u in enumerate(uu):rows.append(dict(ctx=u['ctx'], init=u['init'], cond=c, g=gains[0], mg=mg, unit=ui))
                mg += 1
            uidx = {u['ctx']:ui for ui, u in enumerate(uu)}
            for i, r in enumerate(rows):
                o = uu[r['unit']]['other'];r['partner'] = (i-r['unit']+uidx[o]) if o in uidx else None
            pm = torch.arange(len(rows), device=loop.lm.dev)
            rec = loop.run(rows, ctxs, pert=None if kind == 'none' else dict(kind=kind, t=t_p, rows=pm), seed=9000+pi)
            save_run(f, rec, rows, uu, dict(t_p=t_p, kind=kind))
            log(f.name, len(rows), 'rows')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='qwen3-8b');p.add_argument('--stages', default='calib,A1,A2,BC,pert')
    p.add_argument('--n', type=int, default=100);p.add_argument('--T', type=int, default=64)
    p.add_argument('--per_pass', type=int, default=32);p.add_argument('--tag', default='main')
    p.add_argument('--dev', default='cuda');p.add_argument('--fp32', action='store_true');p.add_argument('--lens_path', default=None)
    p.add_argument('--Lr', type=int, default=None);p.add_argument('--Lw', type=int, default=None)
    p.add_argument('--split', default='test')
    a = p.parse_args()
    out = OUT/'e2'/a.model/a.tag;out.mkdir(parents=True, exist_ok=True)
    from common import MODELS
    if a.Lr is not None:MODELS[a.model]['read'] = a.Lr
    if a.Lw is not None:MODELS[a.model]['write'] = a.Lw
    lm = LM(a.model, dev=a.dev, dtype=torch.float32 if a.fp32 else torch.bfloat16, lens_path=a.lens_path)
    Lr, Lw = lm.cfg['read'], lm.cfg['write']
    ctxs = make_contexts(a.n, split=a.split)
    atomic_json(out/'contexts.json', ctxs);atomic_json(out/'provenance.json', dict(lm.provenance, Lr=Lr, Lw=Lw, T=a.T, n=a.n))
    for st in a.stages.split(','):
        so = out/st;so.mkdir(exist_ok=True)
        if (so/'DONE').exists():log('skip', st);continue
        t0 = time.time();log('stage', st)
        if st == 'calib':
            C, info = stage_calib(lm, so, Lr, Lw);atomic_json(so/'calib.json', info)
            # transport fidelity: one write of the own code at g=1, read-after-write overlap with the written code
            loop = Loop(lm, C, T=1);units = ctx_units(ctxs, 'std')[:64];fid = {}
            for c in ('J', 'Jpb', 'dense', 'pca'):
                rows = [dict(ctx=u['ctx'], init=u['ctx'], cond=c, g=1.) for u in units]
                rows0 = [dict(ctx=u['ctx'], init=u['ctx'], cond='nofb', g=0.) for u in units]
                r1 = loop.run(rows+rows0, ctxs);n = len(rows)
                jac = [len(set(r1['idx'][1, i])&set(r1['idx'][0, i]))/25 for i in range(n)]
                jac0 = [len(set(r1['idx'][1, n+i])&set(r1['idx'][0, n+i]))/25 for i in range(n)]
                dz = r1['z'][1, :n].astype(np.float32)-r1['z'][1, n:].astype(np.float32);z0 = r1['z'][0, :n].astype(np.float32)
                cos = (dz*z0).sum(1)/(np.linalg.norm(dz, axis=1)*np.linalg.norm(z0, axis=1)+1e-9)
                fid[c] = dict(atom_overlap_written=float(np.mean(jac)), atom_overlap_nofb=float(np.mean(jac0)),
                              z_delta_cos_with_code=float(np.mean(cos)), lo_shift=float(np.mean(np.abs(r1['lo'][1, :n]-r1['lo'][1, n:]))))
            atomic_json(so/'fidelity.json', fid);log('fidelity', fid)
        else:
            C = load_calib(lm, out/'calib'/'calib.npz');loop = Loop(lm, C, T=a.T)
            if st == 'A1':stage_A(loop, ctxs, so, a.per_pass, conds=COND_A1, name='A1')
            else:
                g1, g2 = select_gains(out);log('selected gains', g1, g2)
                if st == 'A2':stage_A(loop, ctxs, so, a.per_pass, conds=cond_a2(g1, g2), name='A2')
                elif st == 'BC':stage_BC(loop, ctxs, so, a.per_pass, gains=(g1, g2))
                elif st == 'pert':stage_pert(loop, ctxs, so, a.per_pass, t_p=16 if a.T >= 32 else max(1, a.T//2), gains=(g1, g2))
                elif st == 'BCs':stage_BC(loop, ctxs, so, a.per_pass, gains=(1., 2.), kinds=('std',))   # exploratory strong coupling
                elif st == 'BC1':stage_BC(loop, ctxs, so, a.per_pass, seeds=(0,), gains=(g1, g2), kinds=('std',))  # 14B replication (amendment 16)
                elif st == 'X':stage_A(loop, ctxs, so, a.per_pass, name='X',
                                       conds=[('nofb', 0.)]+[(c, g) for c in ('J', 'inc', 'out', 'lens') for g in (g1, g2)])
        (so/'DONE').write_text(f'{time.time()-t0:.0f}s');log('stage done', st, f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
