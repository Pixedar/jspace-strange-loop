"""Research v5 Phase Q (PROTOCOL_V5 §5, §8): qualification and timing of the self-forecast loop. Qwen3-8B, one GPU.

    python qual.py --stage calib  --out /workspace/v5/q     natural statistics, readers, writers, hidden-mode direction
    python qual.py --stage g0     --out /workspace/v5/q     replay determinism, cached vs full, branch order
    python qual.py --stage fp32   --out /workspace/v5/q     G0(d): the same passes in an fp32 model (separate process)
    python qual.py --stage gain   --out /workspace/v5/q     gain rule (g*), G1(a)
    python qual.py --stage g2     --out /workspace/v5/q     G2 state-specificity, G1(c)
    python qual.py --stage g3     --out /workspace/v5/q     G3 positive control (hidden mode)
    python qual.py --stage pilot  --out /workspace/v5/q     donor coverage, manipulation size, power (provisional F_q)
    python qual.py --stage report --out /workspace/v5/q     gate table, timing, Phase P cost projection

Loop (protocol §4): decision position of a fixed prompt, prefix KV fixed; write at L_w = 22 (replacing the previous
write), read at L_r = 29; observables lo and c^ch = lo(prompt + challenge ch, same write) - lo(prompt, write)."""
import argparse, json, math, os, pathlib, sys, time
import numpy as np
import torch

V3 = pathlib.Path(os.environ.get('V3_CODE', pathlib.Path(__file__).resolve().parents[2]/'v3_decision_loop'/'code'))
sys.path.insert(0, str(V3))
from common import LM, log, atomic_npz, atomic_json  # noqa: E402
from tasks import build_all, user_text               # noqa: E402

LR, LW, T, KZ = 29, 22, 6, 32
CH = ['contra', 'auth', 'hint', 'irrel']               # c^ch order; contra is the primary observable
NF = 6                                                  # forecast vector: (lo, c_contra) x horizons 1..3
GAINS = [.25, .5, 1., 2.]
BS = int(os.environ.get('Q_BS', 64))
DEV = 'cuda'
CNT = dict(passes=0, t_gpu=0.)                          # decision-token passes and their wall time (timing gate)


# ----------------------------------------------------------------------------------------------------------------
# scenarios
# ----------------------------------------------------------------------------------------------------------------
SMOKE = int(os.environ.get('Q_SMOKE', 0))


def f2_items(split, n_scen=None):
    """One item per (scenario, evidence type, direction): its plain prompt and the four challenged prompts."""
    if SMOKE:n_scen = min(n_scen or 10**9, {'train':12, 'val':10, 'test':10}[split])
    data = build_all('main', 20261011)['F2'];by = {}
    for it in data:
        if it['split'] == split:by.setdefault((it['sid'], it['evidence'], it['edir']), {})[it['challenge']] = it
    sids = list(dict.fromkeys(k[0] for k in by))[:n_scen] if n_scen else list(dict.fromkeys(k[0] for k in by))
    items = []
    for (sid, ev, d), chs in by.items():
        if sid not in sids:continue
        b = chs['none']
        items.append(dict(sid=sid, sub=b['sub'], template=b['template'], evidence=ev, edir=-1 if d is None else int(d),
                          cands=b['cands'], text={c:user_text(chs[c])[0] for c in ['none']+CH}))
    return items


SUBS, EVS = ['attach', 'pronoun', 'cause'], ['none', 'weak', 'strong', 'hearsay', 'unrel', 'resolved']


def content_onehot(items):
    X = np.zeros((len(items), 3+6+3+6), np.float32)
    for i, it in enumerate(items):
        X[i, SUBS.index(it['sub'])] = 1;X[i, 3+EVS.index(it['evidence'])] = 1;X[i, 9+it['edir']+1] = 1
        X[i, 12+it['template']] = 1
    return X


# ----------------------------------------------------------------------------------------------------------------
# engine
# ----------------------------------------------------------------------------------------------------------------
class Eng:
    def __init__(self, lm):
        self.lm = lm

    @torch.no_grad()
    def prefix(self, texts):
        lm = self.lm;ids, att, _ = lm.encode([lm.chat(t) for t in texts])
        pre_ids, pre_att, last = ids[:, :-1], att[:, :-1], ids[:, -1:]
        posid = (pre_att.long().cumsum(-1)-1).clamp_min(0);lm.clear()
        o = lm.model(input_ids=pre_ids, attention_mask=pre_att, position_ids=posid, use_cache=True, logits_to_keep=1)
        return dict(cache=o.past_key_values, att=torch.cat([pre_att, torch.ones_like(last)], 1), last=last,
                    pos=pre_att.long().sum(-1, keepdim=True), L=pre_ids.shape[1], ids=ids, full_att=att)

    @torch.no_grad()
    def step(self, P, w, read=(LR,)):
        """The decision token on top of the fixed prefix with w (B, d) added at L_w; the cache is cropped back."""
        lm = self.lm;lm.clear()
        for l in read:lm.read[l] = None
        if w is not None:
            ww = w.to(lm.model.dtype);lm.edit[LW] = lambda h, ww=ww:h+ww[:, None, :]
        if DEV == 'cuda':torch.cuda.synchronize()
        t0 = time.time()
        o = lm.model(input_ids=P['last'], attention_mask=P['att'], position_ids=P['pos'], past_key_values=P['cache'],
                     use_cache=True, logits_to_keep=1)
        P['cache'].crop(P['L'])
        out = {l:lm.store[l][:, -1].float() for l in read};out['logits'] = o.logits[:, -1].float()
        if DEV == 'cuda':torch.cuda.synchronize()
        CNT['passes'] += P['last'].shape[0];CNT['t_gpu'] += time.time()-t0
        lm.clear();return out

    @torch.no_grad()
    def full(self, P, w):
        """Whole-sequence recomputation with w added at L_w of the last position only (G0(b))."""
        lm = self.lm;lm.clear();ids, att = P['ids'], P['full_att']
        if w is not None:
            ww = w.to(lm.model.dtype)

            def ed(h, ww=ww):
                h = h.clone();h[:, -1] = h[:, -1]+ww;return h
            lm.edit[LW] = ed
        posid = (att.long().cumsum(-1)-1).clamp_min(0)
        o = lm.model(input_ids=ids, attention_mask=att, position_ids=posid, use_cache=False, logits_to_keep=1)
        lm.clear();return o.logits[:, -1].float()


def cand_sets(lm, items):
    out = []
    for it in items:
        a, b = it['cands'];ia, ib = set(lm.variant_ids(a)), set(lm.variant_ids(b))
        out.append((torch.tensor(sorted(ia-ib), device=DEV, dtype=torch.long), torch.tensor(sorted(ib-ia), device=DEV, dtype=torch.long)))
    return out


def lo_mass(logits, sets):
    lp = torch.log_softmax(logits.float(), -1)
    lo = torch.stack([torch.logsumexp(lp[i, a], 0)-torch.logsumexp(lp[i, b], 0) for i, (a, b) in enumerate(sets)])
    m = torch.stack([torch.logsumexp(lp[i, torch.cat([a, b])], 0).exp() for i, (a, b) in enumerate(sets)])
    return lo, m


# ----------------------------------------------------------------------------------------------------------------
# readers and writers (from calibration)
# ----------------------------------------------------------------------------------------------------------------
class Calib:
    def __init__(self, path):
        z = np.load(path);self.np = {k:z[k] for k in z.files}
        self.t = {k:torch.tensor(v, device=DEV) for k, v in self.np.items() if v.dtype.kind == 'f'}

    def readers(self, lm, hr):
        """hr (B, d) residual at L_r -> (z_R, z_J), each (B, 32) standardised."""
        c = self.t;zR = (((hr-c['mu_r'])/c['sd_r'])@c['V_R'].T)/c['zs_R']
        _, _, rec = lm.lens.decompose(LR, hr-c['mu_r'], 25)
        zJ = ((rec-c['rec_mu'])@c['V_J'].T)/c['zs_J']
        return zR, zJ, rec

    def write(self, u, g):
        n = u.norm(dim=-1, keepdim=True);w = g*float(self.np['n_J'])*u/n.clamp_min(1e-6)
        return torch.where(n > 1e-6, w, torch.zeros_like(w))

    def M(self, seed):
        return self.t[f'Mz{seed}'], self.t[f'Mf{seed}']


# ----------------------------------------------------------------------------------------------------------------
# rollouts
# ----------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def rollout(eng, C, items, mode='none', g=0., seed=0, writer=1, F=None, cand='R', hidden=None, keep_h=False,
            corrupt=None, forced_f=None, Tn=T):
    """mode: 'none' (no write), 'random' (fresh N(0, I) code each step through the writer), 'closed' (W(z, F(z, t))).
    hidden: (sigma (B,), amplitude) adds sigma * a * v_h at L_w at every step (G3).
    corrupt: dict(k=step, f=(B, NF) tensor) replaces the forecast used at step k (only rows where f is finite).
    forced_f: (Tn, B, NF) forecasts to use instead of F's (NaN = use own)."""
    lm = eng.lm;B = len(items);sets = cand_sets(lm, items)
    P = {ch:eng.prefix([it['text'][ch] for it in items]) for ch in ['none']+CH}
    gen = torch.Generator(device=DEV);gen.manual_seed(seed)
    Mz, Mf = C.M(writer);vh = C.t['v_h']
    R = dict(zR=np.zeros((Tn+1, B, KZ), np.float32), zJ=np.zeros((Tn+1, B, KZ), np.float32), lo=np.zeros((Tn+1, B), np.float32),
             c=np.zeros((Tn+1, B, len(CH)), np.float32), mass=np.zeros((Tn+1, B), np.float32),
             eps=np.zeros((Tn, B, KZ+NF), np.float32), f=np.full((Tn, B, NF), np.nan, np.float32), wn=np.zeros((Tn, B), np.float32))
    if keep_h:R['h'] = np.zeros((Tn+1, B, lm.d), np.float32)
    hid = None
    if hidden is not None:
        sig = torch.tensor(hidden[0], device=DEV, dtype=torch.float32);hid = sig[:, None]*hidden[1]*float(C.np['n_J'])*vh[None]
    w = hid
    for t in range(Tn+1):
        o = eng.step(P['none'], w);lo, m = lo_mass(o['logits'], sets)
        R['lo'][t] = lo.cpu().numpy();R['mass'][t] = m.cpu().numpy()
        for j, ch in enumerate(CH):
            oc = eng.step(P[ch], w, read=());loc, _ = lo_mass(oc['logits'], sets);R['c'][t, :, j] = (loc-lo).cpu().numpy()
        zR, zJ, _ = C.readers(lm, o[LR]);R['zR'][t] = zR.cpu().numpy();R['zJ'][t] = zJ.cpu().numpy()
        if keep_h:R['h'][t] = o[LR].cpu().numpy()
        if t == Tn:break
        z = zR if cand == 'R' else zJ
        if mode == 'none':u = None
        elif mode == 'random':
            e = torch.randn(B, KZ+NF, device=DEV, generator=gen);R['eps'][t] = e.cpu().numpy()
            u = e[:, :KZ]@Mz.T+e[:, KZ:]@Mf.T
        else:
            f = F(z, t)
            if forced_f is not None:
                ff = torch.tensor(forced_f[t], device=DEV);ok = torch.isfinite(ff).all(-1);f = torch.where(ok[:, None], ff, f)
            if corrupt is not None and t == corrupt['k']:
                ff = corrupt['f'];ok = torch.isfinite(ff).all(-1);f = torch.where(ok[:, None], ff, f)
            R['f'][t] = f.cpu().numpy();u = z@Mz.T+f@Mf.T
        ww = C.write(u, g) if u is not None else None
        R['wn'][t] = ww.norm(dim=-1).cpu().numpy() if ww is not None else 0.
        w = ww if hid is None else (hid if ww is None else ww+hid)
    del P;torch.cuda.empty_cache()
    return R


def batched(eng, C, items, **kw):
    out = None;hidden = kw.pop('hidden', None);corrupt = kw.pop('corrupt', None);forced = kw.pop('forced_f', None)
    for s in range(0, len(items), BS):
        sl = slice(s, s+BS);kk = dict(kw)
        if hidden is not None:kk['hidden'] = (hidden[0][sl], hidden[1])
        if corrupt is not None:kk['corrupt'] = dict(k=corrupt['k'], f=torch.tensor(corrupt['f'][sl], device=DEV))
        if forced is not None:kk['forced_f'] = forced[:, sl]
        kk['seed'] = kw.get('seed', 0)*1000+s
        r = rollout(eng, C, items[sl], **kk)
        out = r if out is None else {k:np.concatenate([out[k], r[k]], 1) for k in r}
    return out


# ----------------------------------------------------------------------------------------------------------------
# small predictors (G2, G3, provisional F_q)
# ----------------------------------------------------------------------------------------------------------------
class MLP(torch.nn.Module):
    def __init__(self, i, o, h=128):
        super().__init__();self.net = torch.nn.Sequential(torch.nn.Linear(i, h), torch.nn.GELU(), torch.nn.Linear(h, h),
                                                          torch.nn.GELU(), torch.nn.Linear(h, o))

    def forward(self, x):
        return self.net(x)


def fit_mlp(X, Y, Xv=None, Yv=None, seed=0, epochs=300, lr=1e-3, wd=1e-4):
    torch.manual_seed(seed);X = torch.tensor(X, device=DEV);Y = torch.tensor(Y, device=DEV)
    mx, sx = X.mean(0), X.std(0);sx = torch.where(sx < 1e-3, torch.ones_like(sx), sx)   # constant columns stay unscaled
    my, sy = Y.mean(0), Y.std(0).clamp_min(1e-6)
    m = MLP(X.shape[1], Y.shape[1]).to(DEV);opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    n = len(X);perm_gen = torch.Generator(device=DEV);perm_gen.manual_seed(seed)
    # inner early stopping on 15 % of rows (rows of the same scenario were already kept together by the caller)
    nv = max(1, int(.15*n));idx = torch.randperm(n, device=DEV, generator=perm_gen);vi, ti = idx[:nv], idx[nv:]
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        for b in torch.split(ti[torch.randperm(len(ti), device=DEV, generator=perm_gen)], 256):
            loss = (((m((X[b]-mx)/sx)-(Y[b]-my)/sy))**2).mean();opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():v = float((((m((X[vi]-mx)/sx)-(Y[vi]-my)/sy))**2).mean())
        if v < best-1e-5:best, bad = v, 0;best_state = {k:x.clone() for k, x in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 20:break
    m.load_state_dict(best_state)

    def pred(Z):
        with torch.no_grad():return (m((torch.tensor(Z, device=DEV)-mx)/sx)*sy+my).cpu().numpy()
    return pred


def grouped_cv(X, Y, groups, k=5, seed=0):
    """Out-of-fold predictions with folds by scenario."""
    us = np.unique(groups);rng = np.random.default_rng(seed);folds = np.array_split(rng.permutation(us), k)
    P = np.zeros_like(Y)
    for fi, fo in enumerate(folds):
        te = np.isin(groups, fo);pr = fit_mlp(X[~te], Y[~te], seed=seed+fi);P[te] = pr(X[te])
    return P


def r2(y, p):
    return float(1-((y-p)**2).sum()/max(((y-y.mean())**2).sum(), 1e-12))


def boot_scen(groups, fn, n=10000, seed=0):
    """Grouped bootstrap: fn(index array) -> statistic; resamples scenarios."""
    rng = np.random.default_rng(seed);us = np.unique(groups);rows = {u:np.where(groups == u)[0] for u in us}
    vals = []
    for _ in range(n):
        pick = rng.choice(us, len(us));vals.append(fn(np.concatenate([rows[u] for u in pick])))
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


# ----------------------------------------------------------------------------------------------------------------
def deadline_left():
    try:return int(open('/workspace/deadline').read())-time.time()
    except Exception:return 1e9


def load_lm(dtype=torch.bfloat16):
    lm = LM('qwen3-8b', dtype=dtype);lm.model.config.use_cache = True
    return lm


def stage_calib(out):
    lm = load_lm();eng = Eng(lm);t0 = time.time()
    tr = f2_items('train');va = f2_items('val')
    HR, HW, LO = [], [], []
    for s in range(0, len(tr), BS):
        it = tr[s:s+BS];P = eng.prefix([x['text']['none'] for x in it]);o = eng.step(P, None, read=(LR, LW))
        HR.append(o[LR].cpu());HW.append(o[LW].cpu());LO.append(lo_mass(o['logits'], cand_sets(lm, it))[0].cpu());del P
    HR, HW = torch.cat(HR).to(DEV), torch.cat(HW).to(DEV)
    mu_r = HR.mean(0);sd = HR.std(0);sd_r = torch.maximum(sd, .1*sd.median())
    Y = (HR-mu_r)/sd_r;_, _, V = torch.linalg.svd(Y-Y.mean(0), full_matrices=False);V_R = V[:KZ];zs_R = (Y@V_R.T).std(0).clamp_min(1e-6)
    recs = torch.cat([lm.lens.decompose(LR, HR[s:s+256]-mu_r, 25)[2] for s in range(0, len(HR), 256)])
    rec_mu = recs.mean(0);_, _, V2 = torch.linalg.svd(recs-rec_mu, full_matrices=False);V_J = V2[:KZ]
    zs_J = ((recs-rec_mu)@V_J.T).std(0).clamp_min(1e-6);n_J = recs.norm(dim=-1).median()
    mu_w = HW.mean(0);sd_w = HW.std(0);massive = sd_w > 20*sd_w.median()
    Xw = (HW-mu_w);Xw[:, massive] = 0;_, Sw, Vw = torch.linalg.svd(Xw, full_matrices=False);P_w = Vw[:256]
    C = dict(mu_r=mu_r, sd_r=sd_r, V_R=V_R, zs_R=zs_R, rec_mu=rec_mu, V_J=V_J, zs_J=zs_J, n_J=n_J, mu_w=mu_w, sd_w=sd_w,
             massive=massive.float(), P_w=P_w, pw_var=(Sw[:256]**2).sum()/(Sw**2).sum())
    for sd_ in (1, 2, 3):
        g = torch.Generator(device=DEV);g.manual_seed(1000+sd_)
        npw = P_w.shape[0]                                       # 256 (fewer only in smoke mode)
        Mz = P_w.T@torch.randn(npw, KZ, device=DEV, generator=g);Mz = Mz/Mz.norm(dim=0, keepdim=True)
        Mf = P_w.T@torch.randn(npw, NF, device=DEV, generator=g);Mf = Mf/Mf.norm(dim=0, keepdim=True)*math.sqrt(KZ/NF)
        C[f'Mz{sd_}'], C[f'Mf{sd_}'] = Mz, Mf
    # validation: natural observables and the hidden-mode direction (L_w states of the extreme contra-response quartiles)
    lo_v, c_v, m_v, hw_v = [], [], [], []
    for s in range(0, len(va), BS):
        it = va[s:s+BS];sets = cand_sets(lm, it);P = {ch:eng.prefix([x['text'][ch] for x in it]) for ch in ['none']+CH}
        o = eng.step(P['none'], None, read=(LR, LW));lo, m = lo_mass(o['logits'], sets);cc = []
        for ch in CH:cc.append(lo_mass(eng.step(P[ch], None, read=())['logits'], sets)[0]-lo)
        lo_v.append(lo.cpu());m_v.append(m.cpu());c_v.append(torch.stack(cc, 1).cpu());hw_v.append(o[LW].cpu());del P
    lo_v, m_v, c_v, hw_v = torch.cat(lo_v), torch.cat(m_v), torch.cat(c_v), torch.cat(hw_v).to(DEV)
    q1, q3 = torch.quantile(c_v[:, 0], .25), torch.quantile(c_v[:, 0], .75)
    X = hw_v-mu_w;X[:, massive] = 0;vh = X[(c_v[:, 0] >= q3).to(DEV)].mean(0)-X[(c_v[:, 0] <= q1).to(DEV)].mean(0);C['v_h'] = vh/vh.norm()
    atomic_npz(out/'calib.npz', **{k:(v.float().cpu().numpy() if torch.is_tensor(v) else np.array(v, np.float32)) for k, v in C.items()},
               nat_val_lo=lo_v.numpy(), nat_val_c=c_v.numpy(), nat_val_mass=m_v.numpy(), nat_train_lo=torch.cat(LO).numpy())
    info = dict(n_train=len(tr), n_val=len(va), n_J=float(n_J), massive_dims=int(massive.sum()), pw_var=float(C['pw_var']),
                nat_val=dict(mass_frac_ge_0_5=float((m_v >= .5).float().mean()), mean_abs_c={ch:float(c_v[:, j].abs().mean()) for j, ch in enumerate(CH)},
                             G1b_pass=bool(c_v[:, 0].abs().mean() >= 1 and c_v[:, 3].abs().mean() <= .25*c_v[:, 0].abs().mean())),
                seconds=time.time()-t0, provenance=lm.provenance)
    atomic_json(out/'calib.json', info);log('calib', info)


def stage_g0(out):
    lm = load_lm();eng = Eng(lm);C = Calib(out/'calib.npz');va = f2_items('val')[:64];res = {}
    a = batched(eng, C, va, mode='random', g=.5, seed=3);b = batched(eng, C, va, mode='random', g=.5, seed=3)
    res['a_replay_bit_identical'] = bool(all(np.array_equal(a[k], b[k], equal_nan=True) for k in a))
    # branch order: the same prompts with the batch reversed (no-write path: no random draws depend on row position)
    n0 = batched(eng, C, va, mode='none');n1 = batched(eng, C, va[::-1], mode='none')
    res['c_branch_order_maxdiff'] = float(max(np.abs(n0[k]-n1[k][:, ::-1]).max() for k in ('lo', 'c', 'zR', 'zJ')))
    # cached prefix + decision token vs whole-sequence recomputation, with a write
    sets = cand_sets(lm, va[:32]);P = {ch:eng.prefix([x['text'][ch] for x in va[:32]]) for ch in ['none', 'contra']}
    g = torch.Generator(device=DEV);g.manual_seed(5);Mz, Mf = C.M(1)
    diffs = dict(lo=[], c=[]);saved = []
    for rep in range(2):
        e = torch.randn(32, KZ+NF, device=DEV, generator=g);w = C.write(e[:, :KZ]@Mz.T+e[:, KZ:]@Mf.T, 1.)
        lc, _ = lo_mass(eng.step(P['none'], w, read=())['logits'], sets);lf, _ = lo_mass(eng.full(P['none'], w), sets)
        cc, _ = lo_mass(eng.step(P['contra'], w, read=())['logits'], sets);cf, _ = lo_mass(eng.full(P['contra'], w), sets)
        diffs['lo'] += (lc-lf).abs().tolist();diffs['c'] += ((cc-lc)-(cf-lf)).abs().tolist()
        saved.append(dict(w=w.cpu().numpy(), lo=lc.cpu().numpy(), c=(cc-lc).cpu().numpy()))
    res['b_cached_vs_full_q95'] = {k:float(np.quantile(v, .95)) for k, v in diffs.items()}
    res['b_pass'] = bool(max(res['b_cached_vs_full_q95'].values()) <= .02)
    # inputs for G0(d): the same prompts with no write and with the two writes (fp32 model, separate process)
    lc0, _ = lo_mass(eng.step(P['none'], None, read=())['logits'], sets);cc0, _ = lo_mass(eng.step(P['contra'], None, read=())['logits'], sets)
    atomic_npz(out/'g0_fp32_inputs.npz', w1=saved[0]['w'], w2=saved[1]['w'], lo1=saved[0]['lo'], lo2=saved[1]['lo'], c1=saved[0]['c'],
               c2=saved[1]['c'], lo0=lc0.cpu().numpy(), c0=(cc0-lc0).cpu().numpy())
    res['a_pass'] = res['a_replay_bit_identical'];res['c_pass'] = res['c_branch_order_maxdiff'] <= 1e-3
    atomic_json(out/'g0.json', res);log('g0', res)


def stage_fp32(out):
    lm = load_lm(torch.float32);eng = Eng(lm);va = f2_items('val')[:32];I = np.load(out/'g0_fp32_inputs.npz')
    sets = cand_sets(lm, va);P = {ch:eng.prefix([x['text'][ch] for x in va]) for ch in ['none', 'contra']};res = {}
    d_lo, d_c, sign_bad = [], [], 0;n_sign = 0
    for key, w in (('0', None), ('1', I['w1']), ('2', I['w2'])):
        ww = None if w is None else torch.tensor(w, device=DEV)
        l, _ = lo_mass(eng.step(P['none'], ww, read=())['logits'], sets);c, _ = lo_mass(eng.step(P['contra'], ww, read=())['logits'], sets)
        l, c = l.cpu().numpy(), (c-l).cpu().numpy();lb, cb = I['lo'+key], I['c'+key]
        d_lo += np.abs(l-lb).tolist();d_c += np.abs(c-cb).tolist()
        big = np.abs(cb) > .5;n_sign += int(big.sum());sign_bad += int((np.sign(c[big]) != np.sign(cb[big])).sum())
    res = dict(q95_abs_dlo=float(np.quantile(d_lo, .95)), q95_abs_dc=float(np.quantile(d_c, .95)), sign_flips=sign_bad, n_sign=n_sign)
    res['d_pass'] = bool(res['q95_abs_dlo'] <= .15 and res['q95_abs_dc'] <= .15 and sign_bad == 0)
    atomic_json(out/'g0_fp32.json', res);log('fp32', res)


def stage_gain(out):
    lm = load_lm();eng = Eng(lm);C = Calib(out/'calib.npz');it = f2_items('val', 50);res = {}
    nat = batched(eng, C, it, mode='none', Tn=0);c_nat = nat['c'][0, :, 0]
    for g in GAINS:
        r = batched(eng, C, it, mode='random', g=g, seed=11)
        fin = np.isfinite(r['lo']).all(0)&np.isfinite(r['c']).all((0, 2))
        rms = float(np.sqrt(np.mean((r['c'][1:, fin, 0]-c_nat[None, fin])**2)))
        res[g] = dict(rms_dc_contra=rms, mass_frac_ge_0_5=float(np.mean(r['mass'][1:, fin] >= .5)), diverged=int((~fin).sum()),
                      rms_dlo=float(np.sqrt(np.mean((r['lo'][1:, fin]-nat['lo'][0, fin][None])**2))))
        log('gain', g, res[g])
    nf = noise_floor(out);thr = max(1., 5*nf)
    ok = [g for g in GAINS if res[g]['rms_dc_contra'] >= thr and res[g]['mass_frac_ge_0_5'] >= .9 and res[g]['diverged'] == 0]
    out_ = dict(per_gain=res, g_star=ok[0] if ok else None, noise_floor_c=nf, threshold=thr,
                rule='smallest g: RMS change of c_contra >= max(1 nat, 5 x G0 path-noise q95), mass >= 0.5 in >= 90 %, no divergence (Amendment Q-2)')
    atomic_json(out/'gain.json', out_);log('g*', out_['g_star'])


def natural(eng, C, items):
    """Unwritten observables per item (the prompt's own lo and c): the baseline of every delta (Amendment Q-1)."""
    r = batched(eng, C, items, mode='none', Tn=0);return r['lo'][0], r['c'][0]


def deltas(r, nat):
    """Write-induced change of the observables relative to the prompt's natural values."""
    lo0, c0 = nat;return r['lo']-lo0[None], r['c']-c0[None]


def noise_floor(out):
    try:return float(json.loads((out/'g0.json').read_text())['b_cached_vs_full_q95']['c'])
    except Exception:return .2


def gstar(out):
    g = json.loads((out/'gain.json').read_text())['g_star']
    if g is None:raise SystemExit('no gain qualifies (protocol §5): Phase Q stops here')
    return g


def swap_index(items, lo, t, rng):
    """Content-matched donor row for each row at step t: other scenario, same subtype / evidence / direction, |dlo| <= 0.5."""
    key = [(it['sub'], it['evidence'], it['edir']) for it in items];sid = [it['sid'] for it in items];n = len(items)
    out = np.full(n, -1)
    for i in range(n):
        cand = [j for j in range(n) if key[j] == key[i] and sid[j] != sid[i] and abs(lo[t, j]-lo[t, i]) <= .5]
        if cand:out[i] = cand[int(np.argmin([abs(lo[t, j]-lo[t, i]) for j in cand]))]      # nearest output, deterministic
    return out


def g2_design(items, r, cand, nat):
    """Rows (item, t): features without the state code, the own code, the swap code; targets: the write-induced change of
    c_contra and lo at t+1 relative to the prompt's natural values (Amendment Q-1)."""
    rng = np.random.default_rng(0);Xc = content_onehot(items);n = len(items);rows = []
    Z = r['z'+cand];dlo, dc = deltas(r, nat)
    for t in range(T):
        sw = swap_index(items, r['lo'], t, rng)
        for i in range(n):
            if sw[i] < 0:continue
            base = np.concatenate([Xc[i], np.eye(T, dtype=np.float32)[t], [r['lo'][t, i]], r['eps'][t, i],
                                   r['eps'][t-1, i] if t > 0 else np.zeros(KZ+NF, np.float32)])
            rows.append((items[i]['sid'], base, Z[t, i], Z[t, sw[i]], dc[t+1, i, 0], dlo[t+1, i]))
    g = np.array([x[0] for x in rows]);B = np.stack([x[1] for x in rows]);Zo = np.stack([x[2] for x in rows]);Zs = np.stack([x[3] for x in rows])
    Y = np.stack([[x[4], x[5]] for x in rows]).astype(np.float32)
    return g, B, Zo, Zs, Y


def g2_contrast(items, r, cand, tag, nat):
    g, B, Zo, Zs, Y = g2_design(items, r, cand, nat)
    Po = grouped_cv(np.concatenate([B, Zo], 1).astype(np.float32), Y, g);Ps = grouped_cv(np.concatenate([B, Zs], 1).astype(np.float32), Y, g)
    res = {}
    for j, nm in enumerate(('dc_contra', 'dlo')):
        y, po, ps = Y[:, j], Po[:, j], Ps[:, j]
        d = r2(y, po)-r2(y, ps);ci = boot_scen(g, lambda ix:r2(y[ix], po[ix])-r2(y[ix], ps[ix]), n=2000)
        res[nm] = dict(r2_own=r2(y, po), r2_swap=r2(y, ps), delta=d, ci=ci, pass_=bool(d >= .05 and ci[0] > 0))
    res['n_rows'] = int(len(g));res['n_scen'] = int(len(np.unique(g)))
    log(tag, cand, res);return res


def stage_g2(out):
    lm = load_lm();eng = Eng(lm);C = Calib(out/'calib.npz');g = gstar(out);it = f2_items('val')
    nat = natural(eng, C, it);t0 = time.time();r = batched(eng, C, it, mode='random', g=g, seed=21);tr = time.time()-t0
    atomic_npz(out/'g2_rollouts.npz', nat_lo=nat[0], nat_c=nat[1], **r)
    # G1(c) (Amendment Q-2): the write-induced change of c_contra must stand clear of the bf16 path noise
    _, dc = deltas(r, nat);dcc = dc[1:, :, 0];nf = noise_floor(out);snr = float(np.sqrt(np.mean(dcc**2))/max(nf, 1e-6))
    within = float(np.mean(r['c'][1:, :, 0].var(0))/max(r['c'][1:, :, 0].var(), 1e-12))
    res = dict(g_star=g, G1c_snr=snr, G1c_pass=snr >= 5, G1c_within_share_raw=within, noise_floor_c=nf,
               G1a_mass_frac=float(np.mean(r['mass'][1:] >= .5)), rollout_seconds=tr)
    for cand in ('R', 'J'):res[cand] = g2_contrast(it, r, cand, 'G2', nat)
    res['pass_any'] = bool(res['R']['dc_contra']['pass_'] or res['J']['dc_contra']['pass_'])
    atomic_json(out/'g2.json', res);log('g2', {k:v for k, v in res.items() if k not in ('R', 'J')})


def stage_g3(out):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    lm = load_lm();eng = Eng(lm);C = Calib(out/'calib.npz');g = gstar(out);it = f2_items('val')
    sig = np.where(np.random.default_rng(31).random(len(it)) < .5, -1., 1.)
    nat = natural(eng, C, it)
    r = batched(eng, C, it, mode='random', g=g, seed=31, hidden=(sig, .5));res = dict(g_star=g, hidden_amplitude='0.5 n_J (Amendment Q-3)')
    groups = np.array([x['sid'] for x in it])
    for cand in ('R', 'J'):
        Z = r['z'+cand][1:].transpose(1, 0, 2).reshape(len(it), -1);acc = []
        for tr_, te in GroupKFold(5).split(Z, sig, groups):
            m = LogisticRegression(C=1., max_iter=3000).fit(Z[tr_], sig[tr_]);acc.append(float((m.predict(Z[te]) == sig[te]).mean()))
        res[cand] = dict(sigma_decode=float(np.mean(acc)), contrast=g2_contrast(it, r, cand, 'G3', nat))
        res[cand]['pass_'] = bool(res[cand]['sigma_decode'] >= .9 and res[cand]['contrast']['dc_contra']['delta'] >= .05)
    res['pass_any'] = bool(res['R']['pass_'] or res['J']['pass_'])
    atomic_json(out/'g3.json', res);log('g3', res)


def stage_pilot(out):
    """Provisional forecaster F_q (z_t, t -> six standardised forecasts of the write-induced changes dlo, dc_contra at
    horizons 1-3) fitted on the G2 rollouts; closed loop with it on validation items: own vs one-step-corrupted
    (restored) branch; donor coverage (rule on deltas, Amendment Q-1), manipulation size, power."""
    lm = load_lm();eng = Eng(lm);C = Calib(out/'calib.npz');g = gstar(out);it = f2_items('val')
    r = dict(np.load(out/'g2_rollouts.npz'));nat = (r['nat_lo'], r['nat_c']);dlo, dc = deltas(r, nat);res = dict(g_star=g)
    for cand in ('R', 'J'):
        Z = r['z'+cand];X1, Y1, X3, Y3 = [], [], [], []
        for t in range(T):
            x = np.concatenate([Z[t], np.tile(np.eye(T, dtype=np.float32)[t], (len(it), 1))], 1)
            X1.append(x);Y1.append(np.stack([dlo[t+1], dc[t+1, :, 0]], 1))
            if t+3 <= T:X3.append(x);Y3.append(np.concatenate([np.stack([dlo[t+h], dc[t+h, :, 0]], 1) for h in (2, 3)], 1))
        X1, Y1, X3, Y3 = map(np.concatenate, (X1, Y1, X3, Y3))
        my, sy = np.concatenate([Y1.mean(0), Y3.mean(0)]), np.concatenate([Y1.std(0), Y3.std(0)])+1e-6
        p1 = fit_mlp(X1, (Y1-my[:2])/sy[:2], seed=7);p3 = fit_mlp(X3, (Y3-my[2:])/sy[2:], seed=8)

        def F(z, t, p1=p1, p3=p3, my=my, sy=sy):
            x = torch.cat([z, torch.eye(T, device=DEV)[t].expand(len(z), T)], 1).cpu().numpy();a_ = p1(x)
            if t+3 <= T:b_ = p3(x)
            else:b_ = (np.concatenate([a_, a_], 1)*np.tile(sy[:2], 2)+np.tile(my[:2], 2)-my[2:])/sy[2:]   # beyond the horizon: repeat h=1
            return torch.tensor(np.concatenate([a_, b_], 1), device=DEV, dtype=torch.float32)
        own = batched(eng, C, it, mode='closed', g=g, writer=1, F=F, cand=cand, seed=41)
        odlo, odc = deltas(own, nat)
        # donor rule at k = 2 on deltas: same subtype/evidence/direction, other scenario, |dlo_k| and |f_dlo| within 0.5
        # nats, |f_dc| at least 1 SD apart; <= 3 uses per donor
        k = 2;f = own['f'][k];sd_c = float(np.std(own['f'][:, :, 1]));fl = f[:, 0]*sy[0]+my[0];fc = f[:, 1]
        key = [(x['sub'], x['evidence'], x['edir']) for x in it];sid = [x['sid'] for x in it];uses = {};corr = np.full((len(it), NF), np.nan, np.float32)
        for i in range(len(it)):
            el = [j for j in range(len(it)) if key[j] == key[i] and sid[j] != sid[i] and abs(odlo[k, j]-odlo[k, i]) <= .5
                  and abs(fl[j]-fl[i]) <= .5 and abs(fc[j]-fc[i]) >= sd_c and uses.get(j, 0) < 3]
            if el:
                j = max(el, key=lambda q:(abs(fc[q]-fc[i]), -uses.get(q, 0)));uses[j] = uses.get(j, 0)+1
                corr[i] = f[i];corr[i, 1::2] = own['f'][k, j, 1::2]          # replace the dc forecasts (all horizons)
        m = np.isfinite(corr).all(1);cov = float(m.mean());out_c = dict(coverage=cov, coverage_pass=cov >= .7)
        if m.sum() >= 2:
            rest = batched(eng, C, it, mode='closed', g=g, writer=1, F=F, cand=cand, seed=41, corrupt=dict(k=k, f=corr))
            rdlo, rdc = deltas(rest, nat)
            d_ = rdc[k+1:, m, 0]-odc[k+1:, m, 0];manip = float(np.sqrt(np.mean(d_**2)))
            fr = rest['f'][k+1:T, m, 1]*sy[1]+my[1];st = own['f'][k+1:T, m, 1]*sy[1]+my[1];y = rdc[k+2:T+1, m, 0]
            ow = own['f'][k+1:T, m, 1]*sy[1]+my[1];yo = odc[k+2:T+1, m, 0]
            gs = np.array(sid)[m];per = []
            for s_ in np.unique(gs):
                q = gs == s_;per.append((np.mean((fr[:, q]-y[:, q])**2), np.mean((st[:, q]-y[:, q])**2), np.mean((ow[:, q]-yo[:, q])**2)))
            P_ = np.array(per);tr_ratio = P_[:, 0].mean()/max(P_[:, 1].mean(), 1e-12);ni = P_[:, 0].mean()/max(P_[:, 2].mean(), 1e-12)
            lr_tr = np.log(P_[:, 0]+1e-9)-np.log(P_[:, 1]+1e-9);lr_ni = np.log(P_[:, 0]+1e-9)-np.log(P_[:, 2]+1e-9)
            zz = 2.73+0.84
            n_tr = int(math.ceil((zz*lr_tr.std()/(math.log(.9)-math.log(.75)))**2));n_ni = int(math.ceil((zz*lr_ni.std()/(math.log(1.25)-math.log(1.)))**2))
            out_c.update(manip_rms_dc=manip, manip_pass=manip >= .5, tracking_ratio=float(tr_ratio), noninferiority_ratio=float(ni),
                         sd_log_tr=float(lr_tr.std()), sd_log_ni=float(lr_ni.std()), n_scen_used=int(len(P_)),
                         n_required=dict(TR=n_tr, NI=n_ni), n_test=int(max(200, n_tr, n_ni)), power_ok=bool(max(n_tr, n_ni) <= 400))
        yy = odc[1:, :, 0];ff = own['f'][:T, :, 1]*sy[1]+my[1]
        out_c['fq_selfskill_dc_h1'] = float(1-np.mean((ff-yy)**2)/max(np.var(yy), 1e-12))
        res[cand] = out_c;log('pilot', cand, out_c)
    atomic_json(out/'pilot.json', res)


def stage_report(out):
    J = {n:json.loads((out/f'{n}.json').read_text()) for n in ('calib', 'g0', 'g0_fp32', 'gain', 'g2', 'g3', 'pilot')
         if (out/f'{n}.json').exists()}
    g_ = J.get('gain', {});gs_ = g_.get('g_star');rms = g_.get('per_gain', {}).get(str(gs_), g_.get('per_gain', {}).get(gs_, {})).get('rms_dc_contra') if gs_ else None
    nf = J.get('g0', {}).get('b_cached_vs_full_q95', {}).get('c')
    G0b_rel = bool(nf is not None and rms and nf <= .2*rms)                 # Amendment Q-2: path noise <= 20 % of the effect
    gates = dict(G0a=J.get('g0', {}).get('a_pass'), G0b_abs=J.get('g0', {}).get('b_pass'), G0b_rel=G0b_rel, G0c=J.get('g0', {}).get('c_pass'),
                 G0d=J.get('g0_fp32', {}).get('d_pass'), G1b=J.get('calib', {}).get('nat_val', {}).get('G1b_pass'),
                 g_star=J.get('gain', {}).get('g_star'), G1c=J.get('g2', {}).get('G1c_pass'), G2=J.get('g2', {}).get('pass_any'),
                 G3=J.get('g3', {}).get('pass_any'))
    rep = dict(gates=gates, files=sorted(J), cnt=CNT)
    atomic_json(out/'q_report.json', rep);log('report', rep)


def main():
    p = argparse.ArgumentParser();p.add_argument('--stage', required=True);p.add_argument('--out', required=True)
    a = p.parse_args();out = pathlib.Path(a.out);out.mkdir(parents=True, exist_ok=True)
    if deadline_left() < 300 and a.stage != 'report':raise SystemExit('deadline: not starting ' + a.stage)
    t0 = time.time()
    {'calib':stage_calib, 'g0':stage_g0, 'fp32':stage_fp32, 'gain':stage_gain, 'g2':stage_g2, 'g3':stage_g3,
     'pilot':stage_pilot, 'report':stage_report}[a.stage](out)
    tm = dict(stage=a.stage, seconds=time.time()-t0, passes=CNT['passes'], gpu_seconds_in_passes=CNT['t_gpu'],
              ms_per_pass=1000*CNT['t_gpu']/max(CNT['passes'], 1),
              peak_gb=float(torch.cuda.max_memory_allocated()/1e9) if torch.cuda.is_available() else 0.)
    with open(out/'timing.jsonl', 'a') as f:f.write(json.dumps(tm)+'\n')
    log('timing', tm)


if __name__ == '__main__':
    main()
