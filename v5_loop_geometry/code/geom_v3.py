"""Geometry of what the v3 loops built (research_v5/GEOMETRY_PLAN.md, retrospective phase). CPU only.

    python geom_v3.py --root <.../v3/e2/qwen3-8b/main> --model Qwen/Qwen3-8B --rev <sha> --out <dir> [--quick 1]

Objects O1-O7 and measures M1-M9 of the plan, from the v3 E2 arrays: A1 (J loop at five gains, residual trajectories),
A2 (control feedbacks), BC/BCY and BCs or BC1 (self-model regimes), P/PBC (perturbations at t = 16), contexts.json.
Writes geom.json and figures into --out."""
import argparse, collections, json, math, os, pathlib, time
import numpy as np

TS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
FR_T = [1, 2, 3, 4, 8, 16, 32, 64]            # loop steps that enter the pooled PCA frame
K = 128                                       # frame dimension
RNG = np.random.default_rng(20261012)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ----------------------------------------------------------------------------------------------------------------
# generic measures
# ----------------------------------------------------------------------------------------------------------------
def dedup(X, dec=5):
    return np.unique(np.round(np.asarray(X, np.float64), dec), axis=0)


def pr(X):
    """Participation ratio of the covariance (M1)."""
    X = np.asarray(X, np.float64);X = X-X.mean(0)
    s = np.linalg.svd(X, compute_uv=False)**2
    return float(s.sum()**2/max((s**2).sum(), 1e-30))


def spectrum(X, n=10):
    X = np.asarray(X, np.float64);X = X-X.mean(0);s = np.linalg.svd(X, compute_uv=False)**2
    return (s[:n]/s.sum()).round(4).tolist()


def twonn(X, frac=.9, scales=True, max_n=8000):
    """TwoNN intrinsic dimension (Facco et al. 2017): -log(1-F(mu)) = d log(mu), first 90 % of sorted mu (M2).
    Computed on de-duplicated points; scale stability from random halves and quarters."""
    from sklearn.neighbors import NearestNeighbors
    X = dedup(X)
    if len(X) > max_n:X = X[RNG.choice(len(X), max_n, replace=False)]

    def one(Y):
        if len(Y) < 20:return float('nan')
        d, _ = NearestNeighbors(n_neighbors=3).fit(Y).kneighbors(Y)
        ok = d[:, 1] > 1e-12;mu = np.sort(d[ok, 2]/d[ok, 1]);N = len(mu)
        if N < 20:return float('nan')
        k = int(N*frac);x = np.log(mu[:k]);y = -np.log(1-np.arange(1, N+1)[:k]/N)
        return float((x*y).sum()/max((x*x).sum(), 1e-30))
    out = dict(n=len(X), d=one(X))
    if scales and len(X) >= 80:
        out['d_half'] = float(np.nanmean([one(X[RNG.choice(len(X), len(X)//2, replace=False)]) for _ in range(3)]))
        out['d_quarter'] = float(np.nanmean([one(X[RNG.choice(len(X), len(X)//4, replace=False)]) for _ in range(3)]))
    return out


def spread(X, m=600):
    from scipy.spatial.distance import pdist
    X = np.asarray(X, np.float64)
    if len(X) > m:X = X[RNG.choice(len(X), m, replace=False)]
    return float(np.median(pdist(X))) if len(X) > 2 else float('nan')


def rsa(X0, X1):
    """Spearman correlation of the two context-by-context distance matrices (M5)."""
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    a, b = pdist(np.asarray(X0, np.float64)), pdist(np.asarray(X1, np.float64))
    if np.std(b) < 1e-12 or np.std(a) < 1e-12:return 0.
    return float(spearmanr(a, b).correlation)


def decode(X, y, groups, n_pc=50):
    """Grouped 5-fold logistic decoding accuracy (M5); groups = scenario ids, so A and B of one scenario share a fold."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(X, np.float64);y = np.asarray(y);groups = np.asarray(groups)
    if len(set(y)) < 2 or len(set(groups)) < 5:return float('nan')
    acc = []
    for tr, te in GroupKFold(5).split(X, y, groups):
        if len(set(y[tr])) < 2:continue
        if np.std(X[tr]) < 1e-12:acc.append(float(np.mean(y[te] == collections.Counter(y[tr]).most_common(1)[0][0])));continue
        m = make_pipeline(StandardScaler(), PCA(min(n_pc, len(tr)-1, X.shape[1])), LogisticRegression(C=1., max_iter=3000))
        m.fit(X[tr], y[tr]);acc.append(float((m.predict(X[te]) == y[te]).mean()))
    return float(np.mean(acc)) if acc else float('nan')


def fps(X, n, seed=0):
    """Farthest-point subsample."""
    X = np.asarray(X, np.float64);N = len(X)
    if N <= n:return X
    rng = np.random.default_rng(seed);sel = [int(rng.integers(N))];d = np.linalg.norm(X-X[sel[0]], axis=1)
    for _ in range(n-1):
        i = int(d.argmax());sel.append(i);d = np.minimum(d, np.linalg.norm(X-X[i], axis=1))
    return X[sel]


def _ph(Y):
    from ripser import ripser          # pip install ripser (installed on the analysis instance)
    dg = ripser(Y, maxdim=1)['dgms']
    h0 = dg[0][np.isfinite(dg[0][:, 1])];h1 = dg[1]
    p0 = np.sort(h0[:, 1]-h0[:, 0])[::-1] if len(h0) else np.zeros(0)
    p1 = np.sort(h1[:, 1]-h1[:, 0])[::-1] if len(h1) else np.zeros(0)
    return p0, p1


def ph_test(args):
    """Persistent homology with a matched Gaussian null (M3). Returns significant loops / clusters."""
    name, X, dims, n, nnull, seed = args
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    X = dedup(X)
    if len(X) < 30:return name, dict(n_unique=int(len(X)), skipped=True)
    Xc = X-X.mean(0);_, S, Vt = np.linalg.svd(Xc, full_matrices=False);dims = min(dims, Vt.shape[0])
    Y = Xc@Vt[:dims].T;Y = Y/np.sqrt((Y**2).sum(1).mean());cov = np.cov(Y.T)
    p0, p1 = _ph(fps(Y, n, seed))
    rng = np.random.default_rng(seed+1);m0, m1 = [], []
    for k in range(nnull):
        G = rng.multivariate_normal(np.zeros(dims), cov, size=len(Y));q0, q1 = _ph(fps(G, n, seed+2+k))
        m0.append(float(q0[0]) if len(q0) else 0.);m1.append(float(q1[0]) if len(q1) else 0.)
    top1 = float(p1[0]) if len(p1) else 0.
    return name, dict(n_unique=int(len(X)), dims=int(dims), n_used=int(min(n, len(Y))), h1_top=p1[:5].round(4).tolist(),
                      h0_top=p0[:8].round(4).tolist(), null_h1_max=float(np.max(m1)), null_h1_q95=float(np.quantile(m1, .95)),
                      null_h0_max=float(np.max(m0)), p_loop=float((1+sum(m >= top1 for m in m1))/(1+nnull)),
                      n_sig_loops=int((p1 > np.max(m1)).sum()), n_sig_clusters=int((p0 > np.max(m0)).sum()))


def lz76(s):
    """Lempel-Ziv (1976) complexity of a symbol sequence, normalised by n / log_A(n)."""
    s = list(s);n = len(s)
    if n < 2:return float('nan')
    i, k, l, c, kmax = 0, 1, 1, 1, 1
    while True:
        if s[i+k-1] == s[l+k-1]:
            k += 1
            if l+k > n:c += 1;break
        else:
            kmax = max(k, kmax);i += 1
            if i == l:
                c += 1;l += kmax
                if l+1 > n:break
                i, k, kmax = 0, 1, 1
            else:k = 1
    A = max(len(set(s)), 2)
    return float(c*math.log(n, A)/n)


def entropy(lab):
    c = np.array(list(collections.Counter(lab).values()), float);p = c/c.sum()
    return float(-(p*np.log2(p)).sum())


def nmi_perm(a, b, nperm=199):
    from sklearn.metrics import normalized_mutual_info_score as nmi
    a, b = np.asarray(a), np.asarray(b);obs = nmi(a, b)
    null = [nmi(a, RNG.permutation(b)) for _ in range(nperm)]
    return dict(nmi=float(obs), null_q95=float(np.quantile(null, .95)), p=float((1+sum(x >= obs for x in null))/(1+nperm)))


def eta2(score, lab):
    """Share of the variance of a coordinate explained by a categorical variable."""
    score = np.asarray(score, float);lab = np.asarray(lab);m = score.mean();ss = ((score-m)**2).sum()
    if ss < 1e-12:return 0.
    return float(sum((lab == g).sum()*(score[lab == g].mean()-m)**2 for g in set(lab.tolist()))/ss)


# ----------------------------------------------------------------------------------------------------------------
# loop-specific measures
# ----------------------------------------------------------------------------------------------------------------
def period_census(z, late=32, pmax=16):
    """Per trajectory: 1 = exact fixed point, p = exact cycle of period p in the last `late` steps, 0 = none (M6/M7);
    t_star = first step from which the state never changes (fixed points only)."""
    T = z.shape[0]-1;n = z.shape[1];per = np.zeros(n, int);tstar = np.full(n, -1)
    same = (z[1:] == z[:-1]).all(-1)                        # (T, n)
    for i in range(n):
        for p in range(1, pmax+1):
            if all((z[t, i] == z[t-p, i]).all() for t in range(T-late+1, T+1)):per[i] = p;break
        if per[i] == 1:
            s = same[:, i];t = T
            while t > 0 and s[t-1]:t -= 1
            tstar[i] = t
    return per, tstar


def flow_stats(S, tstar_hint=None):
    """Flow geometry of trajectories S (T+1, n, D) in a frame (M4). Deviations from each trajectory's final state."""
    S = np.asarray(S, np.float64);T = S.shape[0]-1;X = S-S[-1][None];nrm = np.linalg.norm(X, axis=-1)
    eps = 1e-6*max(np.median(nrm[0]), 1e-12)
    conv = nrm < eps
    tst = np.array([next((t for t in range(T+1) if conv[t:, i].all()), T+1) for i in range(S.shape[1])])
    ratios, angs, sangs, plan, wind, longT = [], [], [], [], [], []
    for i in range(S.shape[1]):
        ts = tst[i]
        for t in range(min(ts, T)-1):
            if nrm[t, i] > eps and nrm[t+1, i] > eps:
                ratios.append(nrm[t+1, i]/nrm[t, i])
                c = X[t, i]@X[t+1, i]/(nrm[t, i]*nrm[t+1, i]);angs.append(math.degrees(math.acos(max(-1., min(1., c)))))
        D = S[1:, i]-S[:-1, i];dn = np.linalg.norm(D, axis=-1)
        for t in range(min(ts, T)-2):
            if dn[t] > eps and dn[t+1] > eps:
                c = D[t]@D[t+1]/(dn[t]*dn[t+1]);sangs.append(math.degrees(math.acos(max(-1., min(1., c)))))
        if ts >= 4:
            Y = X[:ts, i];_, s, Vt = np.linalg.svd(Y, full_matrices=False)
            plan.append(float((s[:2]**2).sum()/max((s**2).sum(), 1e-30)))
            if ts >= 6:
                P = Y@Vt[:2].T;a = np.unwrap(np.arctan2(P[:, 1], P[:, 0]));wind.append(float(abs(a[-1]-a[0])));longT.append(i)
    out = dict(n=int(S.shape[1]), transient_median=float(np.median(tst)), transient_q90=float(np.quantile(tst, .9)),
               frac_converged=float(np.mean(tst <= T)),
               contraction_median=float(np.median(ratios)) if ratios else None,
               dev_angle_median=float(np.median(angs)) if angs else None, dev_angle_q=[float(np.quantile(angs, q)) for q in (.1, .5, .9)] if angs else None,
               step_angle_median=float(np.median(sangs)) if sangs else None,
               frac_step_reversal=float(np.mean(np.array(sangs) > 120)) if sangs else None,
               planarity_median=float(np.median(plan)) if plan else None,
               n_long=len(wind), frac_winding_gt_pi=float(np.mean(np.array(wind) > math.pi)) if wind else None,
               winding_median_deg=float(math.degrees(np.median(wind))) if wind else None)
    out['pooled'] = pooled_map(X, tst, eps)
    return out


def pooled_map(X, tst, eps, k=64, lam=1e-3, nboot=100):
    """Shared linear map of deviation transitions x_{t+1} ~ A x_t (ridge in the PCA of pooled deviations)."""
    T = X.shape[0]-1;pairs = [(t, i) for i in range(X.shape[1]) for t in range(min(tst[i], T)-1)
                             if np.linalg.norm(X[t, i]) > eps]
    if len(pairs) < 3*k:k = max(2, len(pairs)//4)
    if len(pairs) < 12:return None
    A0 = np.stack([X[t, i] for t, i in pairs]);A1 = np.stack([X[t+1, i] for t, i in pairs])
    _, _, Vt = np.linalg.svd(A0, full_matrices=False);k = min(k, Vt.shape[0]);V = Vt[:k].T
    def fit(sel):
        P0, P1 = A0[sel]@V, A1[sel]@V
        M = np.linalg.solve(P0.T@P0+lam*np.trace(P0.T@P0)/k*np.eye(k), P0.T@P1).T
        ev, W = np.linalg.eig(M);c = P0@np.linalg.pinv(W).T          # mode coordinates of the data
        w = (np.abs(c)**2).mean(0);o = np.argsort(-w)
        return ev[o], w[o]/w.sum(), M, P0, P1
    ev, w, M, P0, P1 = fit(np.arange(len(pairs)))
    r2 = 1-((P1-P0@M.T)**2).sum()/max(((P1-P1.mean(0))**2).sum(), 1e-30)
    cplx = np.abs(ev.imag) > 1e-6
    rot = [float(abs(math.degrees(np.angle(e)))) for e in ev[:8]]
    lead_c = next((j for j in range(len(ev)) if cplx[j]), None)
    boot = []
    groups = collections.defaultdict(list)
    for j, (t, i) in enumerate(pairs):groups[i].append(j)
    keys = list(groups)
    for _ in range(nboot):
        sel = np.concatenate([groups[keys[q]] for q in RNG.integers(0, len(keys), len(keys))])
        e2, w2, *_ = fit(sel);boot.append(float(abs(math.degrees(np.angle(e2[0])))))
    return dict(n_pairs=len(pairs), k=int(k), r2=float(r2), eig_top=[(float(e.real), float(e.imag)) for e in ev[:8]],
                weight_top=w[:8].round(4).tolist(), rot_deg_top=rot, weight_complex=float(w[cplx].sum()),
                lead_complex_rank=lead_c, lead_complex_rot=float(abs(math.degrees(np.angle(ev[lead_c])))) if lead_c is not None else None,
                lead_complex_weight=float(w[lead_c]) if lead_c is not None else None,
                top_mode_rot_ci=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))])


# ----------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------
class Data:
    def __init__(self, root, quick=0):
        self.root = pathlib.Path(root);self.ctxs = json.loads((self.root/'contexts.json').read_text())
        self.prov = json.loads((self.root/'provenance.json').read_text());self.quick = quick
        g = json.loads((self.root/'gains.json').read_text());self.g1, self.g2 = g['g1'], g['g2']

    def files(self, stage, pat):
        fs = sorted((self.root/stage).glob(pat))
        return fs[:self.quick] if self.quick else fs

    def meta(self, ci, key):
        return self.ctxs[ci][key]


def load_rows(f, keys=('z', 'lo', 'idx', 'coef', 'act', 'perr')):
    meta = json.loads(f.with_suffix('.json').read_text());a = np.load(f)
    arr = {k:a[k] for k in keys if k in a.files}
    hpos = {r:j for j, r in enumerate(meta.get('save_h', []))}
    return meta, arr, hpos, a


def finalize(d, D):
    """Stack per-row lists; drop rows whose residual states or codes are not finite (diverged); add context labels."""
    d = {f:(np.stack(v, 1) if f != 'ctx' else np.array(v)) for f, v in d.items()}
    ok = np.isfinite(d['z'].astype(np.float32)).all(axis=(0, 2))
    for f in ('S_std', 'S_raw'):
        if f in d:ok &= np.isfinite(d[f]).all(axis=(0, 2))
    n_div = int((~ok).sum())
    for f in list(d):
        d[f] = d[f][ok] if f == 'ctx' else d[f][:, ok]
    for key in ('fam', 'tag', 'sub', 'sid'):d[key] = np.array([D.meta(c, key) for c in d['ctx']])
    d['n_diverged'] = n_div
    return d


class Frame:
    """Per-dimension standardisation by natural-state statistics (std floored at 10 % of its median) or plain centring,
    then a pooled PCA frame of K dimensions. `dims`: residual dimensions used (finite in every natural state)."""

    def __init__(self, nat, sample, standardise=True, dims=None):
        self.dims = dims;self.mu = nat.mean(0)
        sd = nat.std(0);self.sd = np.maximum(sd, .1*np.median(sd)) if standardise else np.ones_like(sd)
        Y = (sample-self.mu)/self.sd;self.c = Y.mean(0);Y = Y-self.c
        C = (Y.T@Y)/len(Y);ev, V = np.linalg.eigh(C);o = np.argsort(-ev)
        self.V = V[:, o[:K]].astype(np.float32);self.captured = float(ev[o[:K]].sum()/ev.sum())

    def __call__(self, H):
        if self.dims is not None:H = H[..., self.dims]
        sh = H.shape
        with np.errstate(over='ignore', invalid='ignore'):Y = ((H.reshape(-1, sh[-1]).astype(np.float32)-self.mu)/self.sd-self.c)@self.V
        return Y.reshape(*sh[:-1], K)


# ----------------------------------------------------------------------------------------------------------------
def words_of_model(name, rev):
    """The English J-lens dictionary words (same rule as research_v3/code/common.py: english_vocab)."""
    import re, torch
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from wordfreq import zipf_frequency
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    idx = json.loads(open(hf_hub_download(name, 'model.safetensors.index.json', revision=rev)).read())['weight_map']
    key = 'lm_head.weight' if 'lm_head.weight' in idx else 'model.embed_tokens.weight'
    with safe_open(hf_hub_download(name, idx[key], revision=rev), 'pt') as h:W = h.get_tensor(key)
    n = min(W.shape[0], len(tok));toks = tok.convert_ids_to_tokens(list(range(n)))
    keep = [i for i, s in enumerate(toks) if s and re.fullmatch(r'Ġ[A-Za-z]{3,}', s)]
    nrm = W[keep].float().norm(dim=-1);lo = float(torch.quantile(nrm, .02))
    keep = [i for i, q in zip(keep, nrm.tolist()) if q >= lo and zipf_frequency(toks[i][1:].lower(), 'en') >= 3.
            and re.fullmatch(r'[A-Z]?[a-z]+', toks[i][1:])]
    return [toks[i][1:] for i in keep]


# ----------------------------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True);p.add_argument('--out', required=True)
    p.add_argument('--model', default=None);p.add_argument('--rev', default=None)
    p.add_argument('--quick', type=int, default=0);p.add_argument('--procs', type=int, default=8)
    p.add_argument('--nnull', type=int, default=49);p.add_argument('--words', type=int, default=1)
    p.add_argument('--bc', default='BC');p.add_argument('--bcs', default='BCs')
    a = p.parse_args();out = pathlib.Path(a.out);(out/'figs').mkdir(parents=True, exist_ok=True)
    D = Data(a.root, a.quick);R = dict(provenance=D.prov, g1=D.g1, g2=D.g2, plan='research_v5/GEOMETRY_PLAN.md')
    model = a.model or D.prov['model'];rev = a.rev or D.prov['revision']
    words = None
    if a.words:
        try:words = words_of_model(model, rev);log('dictionary words', len(words))
        except Exception as e:log('words unavailable', repr(e)[:200])
    W = (lambda i:words[int(i)] if words is not None and int(i) < len(words) else f'#{int(i)}')
    ph_jobs = [];keep_arrays = {}

    # ---------------- pass 1: natural states and frame sample from A1 std ----------------
    # Residuals were stored in fp16: dimensions that overflow in any natural state are dropped (massive activations),
    # and loop rows whose states are not finite are counted as diverged and excluded from geometry.
    fA1 = D.files('A1', 'A1_std_*.npz');nat, samp = [], []
    for f in fA1:
        meta, arr, hpos, a_ = load_rows(f, ('z',));H = a_['H']
        for i, r in enumerate(meta['rows']):
            if i in hpos and r['cond'] == 'nofb':nat.append(H[0, hpos[i]].astype(np.float32))
    nat = np.stack(nat);dims = np.where(np.isfinite(nat).all(0))[0];nat = nat[:, dims]
    for f in fA1:
        meta, arr, hpos, a_ = load_rows(f, ('z',));H = a_['H']
        for i, r in enumerate(meta['rows']):
            if i in hpos and r['cond'] == 'J':
                h = H[FR_T, hpos[i]][:, dims].astype(np.float32)
                if np.isfinite(h).all():samp.append(h)
        log('pass1', f.name)
    samp = np.concatenate([nat]+samp)
    if len(samp) > 60000:samp = samp[RNG.choice(len(samp), 60000, replace=False)]
    frames = {'std':Frame(nat, samp, True, dims), 'raw':Frame(nat, samp, False, dims)};del samp
    R['frame'] = {k:dict(K=K, captured=v.captured, dims_used=int(len(dims))) for k, v in frames.items()}
    log('frames', R['frame'])

    # ---------------- pass 2: project A1 trajectories ----------------
    A = collections.defaultdict(lambda:collections.defaultdict(list))     # (cond, g) -> field -> list
    for f in fA1:
        meta, arr, hpos, a_ = load_rows(f, ('z', 'lo', 'idx', 'coef'));H = a_['H']
        for i, r in enumerate(meta['rows']):
            d = A[(r['cond'], r['g'])];d['ctx'].append(r['ctx'])
            for k in ('z', 'lo', 'idx', 'coef'):d[k].append(arr[k][:, i])
            if i in hpos:
                h = H[:, hpos[i]]
                for fk, fr in frames.items():d['S_'+fk].append(fr(h))
        log('pass2', f.name)
    A = {k:finalize(d, D) for k, d in A.items()}
    keysA = sorted(A, key=lambda k:(k[0] != 'nofb', k[1]))
    R['diverged'] = {f'{k[0]}@{k[1]}':A[k]['n_diverged'] for k in keysA}
    log('diverged rows', R['diverged'])

    # ---------------- O1 natural manifold ----------------
    n0 = A[('nofb', 0.)];N0 = n0['S_std'][0];fam, tag, sub = n0['fam'], n0['tag'], n0['sub']
    R['O1_natural'] = dict(n=int(len(N0)), pr=pr(N0), twonn=twonn(N0), spectrum=spectrum(N0),
                           pr_raw=pr(n0['S_raw'][0]), twonn_raw=twonn(n0['S_raw'][0]))
    ph_jobs.append(('O1_natural', N0, 10, 400, a.nnull, 1))
    for fm in sorted(set(fam)):
        m = fam == fm;ph_jobs.append((f'O1_natural_{fm}', N0[m], 10, 400, a.nnull, 2))
    R['O1_natural']['by_family'] = {fm:dict(n=int((fam == fm).sum()), pr=pr(N0[fam == fm]), twonn=twonn(N0[fam == fm], scales=False))
                                    for fm in sorted(set(fam))}
    R['O1_natural']['meaning'] = meaning(N0, n0['idx'][0], n0['coef'][0], fam, tag, sub, n0['lo'][0], W)
    R['O1_natural']['isomap'] = isomap_order(N0, dict(logodds=n0['lo'][0], family=fam, tag=tag, subtype=sub))

    # ---------------- O2/O4 flows and self-reproducing sets per gain ----------------
    R['O2_flow'] = {};R['O4_attractors'] = {};R['M5_retention'] = {}
    for k in keysA:
        c, g = k;d = A[k];name = f'{c}@{g}'
        if 'S_std' not in d or d['S_std'].shape[1] < 20:continue
        S, Sr, fam_k, tag_k, sid_k = d['S_std'], d['S_raw'], d['fam'], d['tag'], d['sid']
        pop = {}
        for t in TS:
            Xt = S[t];pop[t] = dict(pr=pr(Xt), pr_raw=pr(Sr[t]), spread=spread(Xt), rsa0=rsa(S[0], Xt) if t else 1.,
                                    rsa0_raw=rsa(Sr[0], Sr[t]) if t else 1.)
            for fm in ('F2', 'F3', 'F6'):
                m = (fam_k == fm)&np.isin(tag_k, ['A', 'B'])
                if m.sum() >= 20:pop[t][f'decode_{fm}'] = decode(Xt[m], tag_k[m], sid_k[m])
            if t in (0, 8, 64):pop[t]['twonn'] = twonn(Xt)
        R['M5_retention'][name] = pop
        if c == 'nofb':continue
        per, tst = period_census(d['z'])
        R['O2_flow'][name] = dict(census=census_summary(per, tst, d['z']), flow_std=flow_stats(S), flow_raw=flow_stats(Sr),
                                  flow_z=flow_stats(d['z'].astype(np.float32)))
        R['O4_attractors'][name] = attractors(d, per, fam_k, tag_k, d['sub'], sid_k, S, W)
        R['O4_attractors'][name]['meaning_t64'] = meaning(S[-1], d['idx'][-1], d['coef'][-1], fam_k, tag_k, d['sub'], d['lo'][-1], W)
        ph_jobs.append((f'O4_selfrep_{name}', S[-1], 10, 400, a.nnull, 3))
        tt = sorted(set(list(range(0, 9))+[12, 16, 24, 32, 48, 64]))
        ph_jobs.append((f'O2_flowcloud_{name}', S[tt].reshape(-1, K), 10, 400, a.nnull, 4))
        nn_ = S.shape[1];R['O2_flow'][name]['isomap_cloud'] = isomap_order(
            S[tt].reshape(-1, K), dict(step=np.repeat(np.array(tt, float), nn_), logodds=d['lo'][tt].reshape(-1),
                                       family=np.tile(fam_k, len(tt)), tag=np.tile(tag_k, len(tt))))
        R['O2_flow'][name]['twonn_vs_pr_t64'] = dict(twonn=twonn(S[-1]), pr=pr(S[-1]))
        keep_arrays[name] = dict(traj=S[:, :120, :3].astype(np.float32), tag=tag_k[:120], fam=fam_k[:120], end=S[-1, :, :3].astype(np.float32),
                                 end_tag=tag_k, end_fam=fam_k)
        log('A1', name, R['O2_flow'][name]['census'], 'flow', {kk:R['O2_flow'][name]['flow_std'][kk] for kk in ('transient_median', 'dev_angle_median', 'frac_step_reversal')})

    # ---------------- O3 control flows (A2): z-space census and flow; H for J / dense at g1 ----------------
    R['O3_controls'] = {}
    A2 = collections.defaultdict(lambda:collections.defaultdict(list))
    for f in D.files('A2', 'A2_std_*.npz'):
        meta, arr, hpos, a_ = load_rows(f, ('z', 'lo', 'idx', 'coef'));H = a_['H'] if 'H' in a_.files else None
        for i, r in enumerate(meta['rows']):
            dd = A2[(r['cond'], r['g'])];dd['ctx'].append(r['ctx'])
            for kk in ('z', 'lo', 'idx', 'coef'):dd[kk].append(arr[kk][:, i])
            if i in hpos:dd['S_std'].append(frames['std'](H[:, hpos[i]]))
    for k2 in sorted(A2, key=str):
        c, g = k2
        if c == 'nofb':continue
        dd = finalize(A2[k2], D)
        per, tst = period_census(dd['z'])
        o = dict(n_diverged=dd['n_diverged'], census=census_summary(per, tst, dd['z']), flow_z=flow_stats(dd['z'].astype(np.float32)),
                 late_z_twonn=twonn(dd['z'][33:].reshape(-1, 32)), late_z_pr=pr(dd['z'][33:].reshape(-1, 32).astype(np.float64)))
        if 'S_std' in dd and dd['S_std'].shape[1] >= 20:
            Sd, fm2, tg2, sd2 = dd['S_std'], dd['fam'], dd['tag'], dd['sid']
            o['flow_std'] = flow_stats(Sd)
            o['retention'] = {t:dict(pr=pr(Sd[t]), rsa0=rsa(Sd[0], Sd[t]) if t else 1.,
                                     decode_F2=decode(Sd[t][(fm2 == 'F2')&np.isin(tg2, ['A', 'B'])], tg2[(fm2 == 'F2')&np.isin(tg2, ['A', 'B'])],
                                                      sd2[(fm2 == 'F2')&np.isin(tg2, ['A', 'B'])])) for t in (0, 4, 16, 64)}
            ph_jobs.append((f'O3_selfrep_{c}@{g}', Sd[-1], 10, 400, a.nnull, 5))
        R['O3_controls'][f'{c}@{g}'] = o;log('A2', c, g, o['census'])
    del A2

    # ---------------- O5 self-model regimes ----------------
    R['O5_selfmodel'] = {}
    for stage in [s for s in (a.bc, a.bcs) if (D.root/s).exists()]:
        R['O5_selfmodel'][stage] = selfmodel(D, stage, frames['std'], ph_jobs, a.nnull)

    # ---------------- O6 perturbations ----------------
    if (D.root/'pert').exists():R['O6_perturbation'] = perturbations(D)

    # ---------------- M3 persistent homology (parallel) ----------------
    log('persistent homology:', len(ph_jobs), 'objects')
    try:
        import ripser  # noqa: F401
        from multiprocessing import Pool
        with Pool(a.procs) as pool:R['M3_topology'] = dict(pool.imap_unordered(ph_test, ph_jobs))
    except ImportError:
        log('ripser missing: topology skipped');R['M3_topology'] = {}
    for k, v in sorted(R['M3_topology'].items()):log('PH', k, {kk:v.get(kk) for kk in ('n_unique', 'n_sig_loops', 'p_loop', 'n_sig_clusters')})

    (out/'geom.json').write_text(json.dumps(R, indent=1, default=float))
    log('written', out/'geom.json')
    flat = {'natural':N0[:, :3].astype(np.float32), 'natural_fam':fam, 'natural_tag':tag}
    for nm, dct in keep_arrays.items():
        for kk, vv in dct.items():flat[f'{nm}|{kk}'] = vv
    np.savez_compressed(out/'geom_arrays.npz', **flat)
    try:figures(out, A, keysA, fam, tag, R, D)
    except Exception as e:log('figures failed', repr(e)[:300])


def census_summary(per, tst, z=None):
    n = len(per);c = collections.Counter(per.tolist())
    out = dict(n=n, frac_fixed=c.get(1, 0)/n, frac_cycle=sum(v for k, v in c.items() if k > 1)/n,
               frac_none=c.get(0, 0)/n, periods={int(k):v for k, v in sorted(c.items()) if k > 1},
               t_star_median=float(np.median(tst[tst >= 0])) if (tst >= 0).any() else None)
    if z is not None:                         # near-fixed: largest late step below 1e-3 standardised units
        zz = z.astype(np.float32);step = np.abs(np.diff(zz[-33:], axis=0)).max(axis=(0, 2))
        out['frac_near_fixed_1e-3'] = float(np.mean(step < 1e-3));out['late_maxstep_median'] = float(np.median(step))
    return out


def attractors(d, per, fam, tag, sub, sid, S, W):
    """M6: identity of the self-reproducing state = the five strongest code words at t = 64."""
    idx, coef = d['idx'][-1], d['coef'][-1].astype(np.float32)
    keys = [tuple(sorted(idx[i][np.argsort(-coef[i])[:5]].tolist())) for i in range(len(idx))]
    cnt = collections.Counter(keys);ids = {k:j for j, (k, _) in enumerate(cnt.most_common())}
    lab = np.array([ids[k] for k in keys])
    top = []
    for k, v in cnt.most_common(10):
        m = lab == ids[k]
        top.append(dict(n=v, words=[W(i) for i in k], fam=dict(collections.Counter(fam[m].tolist())),
                        tag=dict(collections.Counter(tag[m].tolist()))))
    out = dict(n=int(len(keys)), n_attractors=len(cnt), n_singletons=sum(1 for v in cnt.values() if v == 1),
               occupancy_entropy_bits=entropy(keys), top=top, nmi_family=nmi_perm(lab, fam), nmi_tag=nmi_perm(lab, tag),
               nmi_subtype=nmi_perm(lab, sub))
    # whole-code overlap between contexts (weighted Jaccard of the 25-word codes) at t = 64 vs natural
    out['code_overlap_t64'] = code_overlap(idx, coef);out['code_overlap_t0'] = code_overlap(d['idx'][0], d['coef'][0].astype(np.float32))
    # basin predictability from the natural state (attractors with >= 10 members, grouped CV)
    big = [ids[k] for k, v in cnt.items() if v >= 10]
    if len(big) >= 2:
        m = np.isin(lab, big);out['basin_decode'] = dict(n=int(m.sum()), classes=len(big), acc=decode(S[0][m], lab[m], sid[m]),
                                                          majority=float(collections.Counter(lab[m].tolist()).most_common(1)[0][1]/m.sum()))
    return out


def code_overlap(idx, coef, m=400):
    n = len(idx);sel = RNG.choice(n, min(n, m), replace=False);vals = []
    for a_ in range(len(sel)):
        da = dict(zip(idx[sel[a_]].tolist(), coef[sel[a_]].tolist()))
        for b_ in range(a_+1, len(sel)):
            db = dict(zip(idx[sel[b_]].tolist(), coef[sel[b_]].tolist()))
            ks = set(da)|set(db);num = sum(min(da.get(k, 0), db.get(k, 0)) for k in ks);den = sum(max(da.get(k, 0), db.get(k, 0)) for k in ks)
            vals.append(num/max(den, 1e-12))
    return dict(mean=float(np.mean(vals)), q90=float(np.quantile(vals, .9)))


def isomap_order(X, vars_, n=1500, k=10):
    """Curved low-dimensional manifold rule: first Isomap coordinate vs variables (Spearman / eta2)."""
    from sklearn.manifold import Isomap
    from scipy.stats import spearmanr
    X = np.asarray(X, np.float64);sel = RNG.choice(len(X), min(n, len(X)), replace=False)
    try:Y = Isomap(n_neighbors=k, n_components=2).fit_transform(X[sel])
    except Exception as e:return dict(error=repr(e)[:120])
    out = dict(n=int(len(sel)))
    for name, v in vars_.items():
        v = np.asarray(v)[sel]
        if v.dtype.kind in 'fiu':out[f'rho_{name}'] = [float(abs(spearmanr(Y[:, j], v).correlation)) if np.std(v) > 0 else None for j in range(2)]
        else:out[f'eta2_{name}'] = [eta2(Y[:, j], v) for j in range(2)]
    return out


def meaning(X, idx, coef, fam, tag, sub, lo, W, ncomp=3):
    """M9: what the leading coordinates of a point set correlate with (descriptive)."""
    from scipy.stats import spearmanr
    X = np.asarray(X, np.float64);Xc = X-X.mean(0);_, s, Vt = np.linalg.svd(Xc, full_matrices=False);P = Xc@Vt[:ncomp].T
    pres = collections.defaultdict(lambda:np.zeros(len(X), bool))
    for i in range(len(X)):
        for w in idx[i][coef[i].astype(np.float32) > 0]:pres[int(w)][i] = True
    common = {w:v for w, v in pres.items() if v.mean() >= .05}
    out = []
    for j in range(P.shape[1]):
        sc = P[:, j];o = dict(var=float(s[j]**2/(s**2).sum()), eta2_family=eta2(sc, fam), eta2_tag=eta2(sc, tag),
                              eta2_subtype=eta2(sc, sub), rho_logodds=float(spearmanr(sc, lo).correlation) if np.std(lo) > 0 else None)
        if common:
            r = {w:float(np.corrcoef(sc, v.astype(float))[0, 1]) for w, v in common.items() if 0 < v.mean() < 1}
            o['words_pos'] = [(W(w), round(x, 3)) for w, x in sorted(r.items(), key=lambda q:-q[1])[:6]]
            o['words_neg'] = [(W(w), round(x, 3)) for w, x in sorted(r.items(), key=lambda q:q[1])[:6]]
        out.append(o)
    return out


def selfmodel(D, stage, frame, ph_jobs, nnull):
    """M7: per-context late behaviour of the self-model regimes, paired by replicate system against R and Y."""
    rows_all = collections.defaultdict(list);H = collections.defaultdict(list)
    for f in D.files(stage, 'BC*_std_*.npz'):
        meta, arr, hpos, a_ = load_rows(f, ('z', 'act', 'perr', 'lo'));chunk = f.stem.split('_')[-1]
        Hf = a_['H'] if (hpos and 'H' in a_.files) else None
        for i, r in enumerate(meta['rows']):
            z = arr['z'][:, i];act = arr['act'][:, i];T = z.shape[0]-1
            per, _ = period_census(z[:, None]);late = z[T-31:]
            rec = dict(cond=r['cond'], g=r['g'], seed=r.get('seed', 0), chunk=chunk, ctx=r['ctx'], period=int(per[0]),
                       n_distinct_late=int(len(np.unique(late, axis=0))),
                       late_step=float(np.linalg.norm(np.diff(late.astype(np.float32), axis=0), axis=-1).mean()),
                       act_entropy=entropy(act[8:].tolist()), act_lz=lz76(act[8:].tolist()),
                       perr_late=float(np.nanmean(arr['perr'][48:, i])) if 'perr' in arr else None)
            rows_all[(r['cond'], r['g'])].append((rec, z))
            if Hf is not None and i in hpos:H[(r['cond'], r['g'])].append((r['ctx'], frame(Hf[:, hpos[i]])))
        log(stage, f.name)
    out = {}
    for k, lst in sorted(rows_all.items(), key=str):
        c, g = k;recs = [r for r, _ in lst];per = np.array([r['period'] for r in recs])
        Z = np.stack([z for _, z in lst], 1).astype(np.float32);late = Z[33:].reshape(-1, Z.shape[-1])
        o = dict(n=len(recs), frac_fixed=float(np.mean(per == 1)), frac_cycle=float(np.mean(per > 1)), frac_none=float(np.mean(per == 0)),
                 periods=dict(collections.Counter(per[per > 1].tolist())),
                 n_distinct_late_mean=float(np.mean([r['n_distinct_late'] for r in recs])),
                 late_step_mean=float(np.mean([r['late_step'] for r in recs])), act_entropy_mean=float(np.mean([r['act_entropy'] for r in recs])),
                 act_lz_mean=float(np.nanmean([r['act_lz'] for r in recs])),
                 perr_late_mean=float(np.nanmean([r['perr_late'] for r in recs if r['perr_late'] is not None])) if recs[0]['perr_late'] is not None else None,
                 late_pr=pr(late), late_twonn=twonn(late), flow_z=flow_stats(Z))
        ph_jobs.append((f'O5_{stage}_late_{c}@{g}', late, 10, 400, nnull, 6))
        if k in H:
            keep = [(cc, s_) for cc, s_ in H[k] if np.isfinite(s_).all()];o['residual_diverged'] = len(H[k])-len(keep)
        if k in H and len(keep) >= 20:
            ctxs = [cc for cc, _ in keep];S = np.stack([s_ for _, s_ in keep], 1)
            fm = np.array([D.meta(cc, 'fam') for cc in ctxs]);tg = np.array([D.meta(cc, 'tag') for cc in ctxs]);sd = np.array([D.meta(cc, 'sid') for cc in ctxs])
            m = (fm == 'F2')&np.isin(tg, ['A', 'B'])
            o['residual'] = dict(n=len(ctxs), late_pr=pr(S[33:].reshape(-1, K)), late_twonn=twonn(S[33:].reshape(-1, K)),
                                 rsa_0_64=rsa(S[0], S[-1]), decode_F2_t64=decode(S[-1][m], tg[m], sd[m]) if m.sum() >= 20 else None,
                                 flow=flow_stats(S))
            ph_jobs.append((f'O5_{stage}_residual_late_{c}@{g}', S[33:].reshape(-1, K), 10, 400, nnull, 7))
        out[f'{c}@{g}'] = o
        log(stage, c, g, {kk:o[kk] for kk in ('frac_fixed', 'frac_cycle', 'frac_none', 'n_distinct_late_mean', 'act_lz_mean')})
    # paired contrasts by replicate system (chunk, seed, gain) against R and Y
    sysm = collections.defaultdict(dict)
    for (c, g), lst in rows_all.items():
        for r, _ in lst:
            key = (r['chunk'], r['seed'], g);sysm[key].setdefault(c, []).append(r)
    con = {}
    for ref in ('R', 'Y'):
        for c in ('B', 'C1', 'C2', 'F', 'R', 'Y'):
            if c == ref:continue
            for g in sorted({k[2] for k in sysm}):
                diffs = collections.defaultdict(list)
                for key, d in sysm.items():
                    if key[2] != g or c not in d or ref not in d:continue
                    for mtr in ('n_distinct_late', 'late_step', 'act_lz'):
                        diffs[mtr].append(np.nanmean([r[mtr] for r in d[c]])-np.nanmean([r[mtr] for r in d[ref]]))
                    diffs['frac_nonfixed'].append(np.mean([r['period'] != 1 for r in d[c]])-np.mean([r['period'] != 1 for r in d[ref]]))
                if diffs:
                    con[f'{c}-{ref}@{g}'] = {m:dict(n=len(v), mean=float(np.mean(v)), ci=boot_ci(v)) for m, v in diffs.items()}
    return dict(policies=out, paired=con)


def boot_ci(v, n=4000):
    v = np.asarray(v, float);bs = [np.mean(v[RNG.integers(0, len(v), len(v))]) for _ in range(n)]
    return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


def perturbations(D):
    """M8: twin-distance curves and fates after a perturbation at t_p; slow-manifold alignment."""
    out = {}
    P = collections.defaultdict(list)
    for f in D.files('pert', 'P_*.npz'):
        meta, arr, hpos, a_ = load_rows(f, ('z',));z = arr['z'];tp = meta['t_p']
        base = {(r['g'], r['unit']):i for i, r in enumerate(meta['rows']) if r['pert'] == 'base'}
        zT = {g:np.stack([z[-1, i] for (gg, u), i in base.items() if gg == g]).astype(np.float32) for g in {k[0] for k in base}}
        ref = {g:float(np.median(np.linalg.norm(v[:, None]-v[None], axis=-1)[np.triu_indices(len(v), 1)])) for g, v in zT.items()}
        for i, r in enumerate(meta['rows']):
            if r['pert'] == 'base':continue
            b = base[(r['g'], r['unit'])];zb, zp = z[:, b].astype(np.float32), z[:, i].astype(np.float32)
            dist = np.linalg.norm(zp-zb, axis=-1)/max(ref[r['g']], 1e-9)
            part = r.get('partner');dpart = None
            if part is not None:dpart = float(np.linalg.norm(zp[-1]-z[-1, part].astype(np.float32))/max(ref[r['g']], 1e-9))
            # slow-manifold test: final approach directions of the perturbed and the unperturbed run (same fixed point)
            cosv = None
            if dist[-1] == 0:
                def approach(zz, t0):
                    dv = np.linalg.norm(zz-zz[-1], axis=-1);nz = [t for t in range(t0, len(zz)) if dv[t] > 0]
                    return (zz[nz[-1]]-zz[-1])/dv[nz[-1]] if nz else None
                ua, ub = approach(zp, tp+1), approach(zb, 0)
                if ua is not None and ub is not None:cosv = float(ua@ub)
            P[(r['pert'], r['g'])].append(dict(dist=dist[tp:].tolist(), dT=float(dist[-1]), dpart=dpart, cos=cosv, ctx=r['ctx']))
        for g in {k[0] for k in base}:
            U = []
            for (gg, u), i in base.items():
                if gg != g:continue
                zz = z[:, i].astype(np.float32);dv = np.linalg.norm(zz-zz[-1], axis=-1);nz = [t for t in range(len(zz)) if dv[t] > 0]
                if nz:U.append((zz[nz[-1]]-zz[-1])/dv[nz[-1]])
            if len(U) > 2:
                U = np.stack(U);C = np.abs(U@U.T)[np.triu_indices(len(U), 1)];P[('between_contexts', g)].append(dict(cosabs=C.tolist()))
    for (kind, g), lst in sorted(P.items(), key=str):
        if kind == 'between_contexts':
            c = np.concatenate([x['cosabs'] for x in lst]);out[f'approach_cos_between_contexts@{g}'] = dict(n=int(len(c)), median_abs=float(np.median(c)),
                                                                                                         frac_gt_0_9=float(np.mean(c > .9)))
            continue
        dist = np.array([x['dist'] for x in lst]);dT = np.array([x['dT'] for x in lst])
        o = dict(n=len(lst), median_curve=np.median(dist, 0)[[0, 1, 2, 4, 8, 16, 32, -1]].round(4).tolist(),
                 frac_exact_return=float(np.mean(dT == 0)), frac_near_return=float(np.mean((dT > 0)&(dT < .1))),
                 frac_offset=float(np.mean(dT >= .1)))
        dp = [x['dpart'] for x in lst if x['dpart'] is not None]
        if dp and kind == 'trans':o['frac_switch_to_partner'] = float(np.mean([(x['dpart'] < .1) and (x['dT'] >= .1) for x in lst if x['dpart'] is not None]))
        cs = [x['cos'] for x in lst if x['cos'] is not None]
        if cs:o['approach_cos'] = dict(n=len(cs), median_abs=float(np.median(np.abs(cs))), frac_gt_0_9=float(np.mean(np.array(cs) > .9)))
        out[f'{kind}@{g}'] = o
    # baseline: alignment of final approach directions between different contexts (unperturbed)
    # self-model twins
    PB = collections.defaultdict(list)
    groups = collections.defaultdict(dict)
    for f in D.files('pert', 'PBC_*.npz'):
        meta = json.loads(f.with_suffix('.json').read_text());chunk = f.stem.split('_')[1]
        groups[chunk][meta['kind']] = (meta, np.load(f)['z'])
    for chunk, d in groups.items():
        if 'none' not in d:continue
        mn, zn = d['none']
        for kind, (m, z) in d.items():
            if kind == 'none':continue
            tp = m['t_p']
            for i, r in enumerate(m['rows']):
                dist = np.linalg.norm(z[:, i].astype(np.float32)-zn[:, i].astype(np.float32), axis=-1)
                PB[(kind, r['cond'])].append(dist[tp:])
    for (kind, c), lst in sorted(PB.items()):
        dist = np.array(lst);out[f'selfmodel_{kind}|{c}'] = dict(n=len(lst), median_curve=np.median(dist, 0)[[0, 1, 2, 4, 8, 16, 32, -1]].round(4).tolist(),
                                                                 frac_zero_at_end=float(np.mean(dist[:, -1] == 0)))
    return out


# ----------------------------------------------------------------------------------------------------------------
def figures(out, A, keysA, fam, tag, R, D):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mod = D.prov['model'].split('/')[-1]
    # 1. flows in the pooled frame
    gains = [k for k in keysA if k[0] == 'J']
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5));ax = ax.ravel()
    N0 = A[('nofb', 0.)]['S_std'][0];col = {'F2':'C0', 'F3':'C1', 'F5':'C2', 'F6':'C3'}
    ax[0].scatter(N0[:, 0], N0[:, 1], s=6, c=[col.get(f, 'k') for f in fam]);ax[0].set_title('natural states (PC1-2 of pooled frame), colour = family')
    tc = {'A':'C0', 'B':'C3', 'N':'0.5', 'S':'C2'}
    for j, k in enumerate(gains[:5]):
        S = A[k]['S_std'];a_ = ax[j+1];tg = A[k]['tag']
        if S.shape[1] == 0:a_.set_title(f'g = {k[1]}: all rows diverged');continue
        for i in RNG.choice(S.shape[1], min(80, S.shape[1]), replace=False):
            a_.plot(S[:, i, 0], S[:, i, 1], '-', lw=.6, color=tc.get(tg[i], 'k'), alpha=.6)
            a_.plot(S[0, i, 0], S[0, i, 1], 'o', ms=2.5, color=tc.get(tg[i], 'k'));a_.plot(S[-1, i, 0], S[-1, i, 1], 'x', ms=4, color='k')
        a_.set_title(f'J loop g = {k[1]}: 80 trajectories (o start, x t = 64)')
    fig.suptitle(f'{mod}: the J-space loop in the residual stream (read layer), pooled PCA frame');fig.tight_layout()
    fig.savefig(out/'figs'/f'geom_flow_{mod}.png', dpi=110);plt.close(fig)
    # 2. retention and dimension over time
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    for k in keysA:
        name = f'{k[0]}@{k[1]}'
        if name not in R['M5_retention']:continue
        pop = R['M5_retention'][name];ts = sorted(pop)
        ax[0].plot(ts, [pop[t]['pr'] for t in ts], label=name);ax[1].plot(ts, [pop[t]['rsa0'] for t in ts], label=name)
        ax[2].plot(ts, [pop[t].get('decode_F2', np.nan) for t in ts], label=name);ax[3].plot(ts, [pop[t].get('decode_F6', np.nan) for t in ts], label=name)
    for a_, t_ in zip(ax, ['participation ratio', 'RSA with natural (Spearman)', 'decode F2 evidence A/B', 'decode F6 secret A/B']):
        a_.set_title(t_);a_.set_xscale('symlog');a_.set_xlabel('loop step')
    ax[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'figs'/f'geom_time_{mod}.png', dpi=110);plt.close(fig)
    # 3. pooled-map eigenvalues
    fig, ax = plt.subplots(1, len(gains), figsize=(3.6*len(gains), 3.6))
    for a_, k in zip(np.atleast_1d(ax), gains):
        pm = R['O2_flow'].get(f'J@{k[1]}', {}).get('flow_std', {}).get('pooled')
        th = np.linspace(0, 2*np.pi, 200);a_.plot(np.cos(th), np.sin(th), 'k-', lw=.5)
        if pm:
            ev = np.array(pm['eig_top']);w = np.array(pm['weight_top'])
            a_.scatter(ev[:, 0], ev[:, 1], s=20+400*w, alpha=.7)
        a_.set_aspect('equal');a_.set_title(f'g = {k[1]}: transition eigenvalues');a_.set_xlim(-1.1, 1.1);a_.set_ylim(-1.1, 1.1)
    fig.tight_layout();fig.savefig(out/'figs'/f'geom_eig_{mod}.png', dpi=110);plt.close(fig)
    log('figures written')


if __name__ == '__main__':
    main()
