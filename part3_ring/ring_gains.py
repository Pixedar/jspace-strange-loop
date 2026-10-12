"""Exploratory, added after the first results (not pre-registered): the ring across the substrate's transition from
wandering to collapse.

Complexity tends to peak near the edge between order and chaos, and an attractor is not bad in itself: what matters
is what it holds and whether it is trivial. So per gain (the main run is the x1 point) and condition:
  exploration, collapse and memory as in ring_analyze.py;
  beta, the spectral exponent of the trajectory (power ~ 1/f^beta over its top 10 components): about 0 for white
       noise, about 2 for a random walk, about 1 is the classic sign of a critical regime;
  an attractor census: the end state of every seed (mean of its last 50 steps), seeds grouped when their end states
       have cosine > 0.9, and for each group the words it holds and what the mind says about it (the glimpse);
  ring_err against its controls on exploration, and the freeze effect, at every gain;
  synchrony: how alike different minds of one condition are at the same moment, in state and in motion (the step
       c_t - c_{t-1}); if minds move together, another mind's surprise is nearly one's own, and the yoked control
       stops being a stranger.

    python ring_gains.py runs/ring/qwen3-4b        -> runs/ring/qwen3-4b/gains/gains.json
"""
import json, os, sys
import numpy as np

out = sys.argv[1]
KEYS = ('C', 'idx', 'coef', 'step_cos', 'pred_cos')


def load(d):
    parts = [d] if os.path.exists(f'{d}/ring.npz') else sorted(f'{d}/{x}' for x in os.listdir(d) if os.path.exists(f'{d}/{x}/ring.npz'))
    Ms = [json.load(open(f'{p}/meta.json', encoding='utf-8')) for p in parts];Z = {}
    for p in parts:
        with np.load(f'{p}/ring.npz') as z:
            for k in KEYS:Z.setdefault(k, []).append(z[k])
    Z = {k:np.concatenate(v, 1) for k, v in Z.items()}
    rows = sum((m['rows'] for m in Ms), []);glimpse = sum((m.get('glimpse') or ['']*len(m['rows']) for m in Ms), [])
    return Z, rows, glimpse, Ms[0]


def pr(V):
    V = V-V.mean(0);ev = np.clip(np.linalg.eigvalsh(V@V.T), 0, None)
    return float(ev.sum()**2/max((ev**2).sum(), 1e-12))


def beta(X, k=10):
    """Spectral exponent of one mind's trajectory X (n, d): top-k components via the Gram matrix, Hann window."""
    X = X-X.mean(0);ev, ec = np.linalg.eigh(X@X.T);Y = ec[:, -k:]*np.sqrt(np.clip(ev[-k:], 0, None))
    F = np.abs(np.fft.rfft(Y*np.hanning(len(Y))[:, None], axis=0))**2;f = np.fft.rfftfreq(len(Y))
    m = (f > 2/len(Y)) & (f < .25)
    return float(-np.polyfit(np.log(f[m]), np.log(F[m].sum(1)+1e-20), 1)[0])


def paired(x, y):
    dd = np.asarray(x, float)-np.asarray(y, float);n = len(dd);se = dd.std(ddof=1)/np.sqrt(n) if n > 1 else 0.
    return dict(diff=float(dd.mean()), se=float(se), t=float(dd.mean()/se) if se > 0 else 0., pos=int((dd > 0).sum()), n=n)


def analyse(d, words):
    Z, rows, glimpse, M = load(d);T, B = Z['step_cos'].shape;TF = M['args']['t_freeze']
    C = Z['C'].astype(np.float32);U = C/np.maximum(np.linalg.norm(C, axis=-1, keepdims=True), 1e-8)
    W, LT, LG = min(100, max(10, T//6)), min(200, T//3), min(100, T//4)
    conds = list(dict.fromkeys(r['cond'] for r in rows));seeds = sorted({r['seed'] for r in rows})
    row = {(r['cond'], r['seed']):i for i, r in enumerate(rows)};res = {}
    for c in conds:
        bs = [row[c, s] for s in seeds]
        m = dict(explore=[np.mean([pr(U[w:w+W, b]) for w in range(T-LT, T-W+1, max(1, W//2))]) for b in bs],
                 explore_pre=[np.mean([pr(U[w:w+W, b]) for w in range(max(0, TF-LT), TF-W+1, max(1, W//2))]) for b in bs],
                 stuck=[float((Z['step_cos'][T-LT:, b] > .98).mean()) for b in bs],
                 lag1=[float(Z['step_cos'][T-LT:, b].mean()) for b in bs],
                 lag100=[float((U[T-LG:, b]*U[T-2*LG:T-LG, b]).sum(-1).mean()) for b in bs],
                 beta=[beta(U[min(100, T//6):, b]) for b in bs],
                 words=[len(set(Z['idx'][:, b, 0].tolist())) for b in bs])
        # attractor census on the end states
        E = U[T-50:, bs].mean(0);E /= np.maximum(np.linalg.norm(E, axis=-1, keepdims=True), 1e-8)
        left, groups = list(range(len(bs))), []
        while left:
            i = left[0];grp = [j for j in left if float(E[i]@E[j]) > .9];groups.append(grp);left = [j for j in left if j not in grp]
        census = []
        for grp in sorted(groups, key=len, reverse=True):
            acc = np.zeros(len(words))
            for j in grp:np.add.at(acc, Z['idx'][T-50:, bs[j]].ravel(), Z['coef'][T-50:, bs[j]].astype(np.float64).ravel())
            census.append(dict(size=len(grp), seeds=[seeds[j] for j in grp], words=[words[i] for i in np.argsort(-acc)[:8]],
                               glimpse=list(dict.fromkeys(glimpse[bs[j]].strip() for j in grp))[:3]))
        late = U[T-LT:, bs];dV = np.diff(C[T-LT-1:, bs], axis=0);dV /= np.maximum(np.linalg.norm(dV, axis=-1, keepdims=True), 1e-8)
        off = lambda X:float(((np.einsum('tid,tjd->tij', X, X).sum((1, 2))-X.shape[1])/(X.shape[1]*(X.shape[1]-1))).mean())
        res[c] = dict(per_seed=m, mean={k:float(np.mean(v)) for k, v in m.items()}, n_attractors=len(groups), census=census,
                      sync_state=off(late), sync_motion=off(dV))
    tests = {}
    if 'ring_err' in res:
        for ctrl in ('vel', 'yoked_err', 'shuf_err', 'rand_err', 'free', 'content'):
            if ctrl in res:tests[f'ring_err_vs_{ctrl}'] = {k:paired(res['ring_err']['per_seed'][k], res[ctrl]['per_seed'][k])
                                                           for k in ('explore', 'stuck', 'beta')}
        if 'frozen_err' in res:
            f, r = res['frozen_err']['per_seed'], res['ring_err']['per_seed']
            tests['freeze_effect'] = dict(explore=paired(np.subtract(f['explore'], f['explore_pre']), np.subtract(r['explore'], r['explore_pre'])))
    return dict(g=rows[0]['g'], sigma=rows[0]['sigma'], seeds=len(seeds), T=T, conds=res, tests=tests)


runs = {}
M0 = json.load(open(f'{out}/params.json'))
words = None
for d in [f'{out}/main']+sorted(f'{out}/gains/{x}' for x in os.listdir(f'{out}/gains') if x.startswith('g') and os.path.isdir(f'{out}/gains/{x}')):
    if not any(os.path.exists(f'{p}/meta.json') for p in [d]+[f'{d}/{x}' for x in os.listdir(d)] if os.path.isdir(p)):continue
    if words is None:
        first = d if os.path.exists(f'{d}/meta.json') else [f'{d}/{x}' for x in sorted(os.listdir(d)) if os.path.exists(f'{d}/{x}/meta.json')][0]
        words = json.load(open(f'{first}/meta.json', encoding='utf-8'))['words']
    r = analyse(d, words);runs[r['g']] = r;print(f'analysed g {r["g"]} ({d})', flush=True)
gs = sorted(runs)
json.dump(dict(picked_g=M0['g'], gains=[runs[g] for g in gs]), open(f'{out}/gains/gains.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

conds = list(runs[gs[0]]['conds'])
print(f'\n{out}: gains {gs} (picked {M0["g"]}), sigma {runs[gs[0]]["sigma"]}')
for k, fmt in (('explore', '{:7.2f}'), ('stuck', '{:7.3f}'), ('lag100', '{:7.3f}'), ('beta', '{:7.2f}'), ('words', '{:7.1f}')):
    print(f'\n{k:11s}' + ''.join(f'{"g "+str(g):>9s}' for g in gs))
    for c in conds:print(f'{c:11s}' + ''.join(f'{fmt.format(runs[g]["conds"][c]["mean"][k]):>9s}' for g in gs))
print('\nsynchrony between minds of a condition (state / motion, mean cosine at the same step)')
for c in conds:print(f'{c:11s}' + ''.join(f'{runs[g]["conds"][c]["sync_state"]:6.2f}/{runs[g]["conds"][c]["sync_motion"]:4.2f}' for g in gs))
print('\nring_err minus control, exploration (t over seeds)')
for name in runs[gs[0]]['tests']:
    print(f'{name:22s}' + ''.join(f'{runs[g]["tests"][name]["explore"]["diff"]:+7.2f} t{runs[g]["tests"][name]["explore"]["t"]:+5.1f}  '
                                  for g in gs if name in runs[g]['tests']))
print('\nattractors: number of distinct end states among the seeds; the biggest group, its words and what it says')
for c in ('free', 'content', 'ring_pred', 'ring_err', 'yoked_err', 'rand_err'):
    if c not in conds:continue
    for g in gs:
        r = runs[g]['conds'][c];top = r['census'][0]
        print(f'{c:10s} g {g:<6} {r["n_attractors"]:2d} end states | biggest {top["size"]:2d}: {", ".join(top["words"][:6])} | {top["glimpse"][0][:90]!r}')
