"""Analysis of a ring run: is there a loop that is more than feedback, noise or momentum?

    python ring_analyze.py runs/ring/qwen3-4b      -> results.json

Fixed before any real run:
  primary   ring_err against vel, yoked_err, shuf_err and rand_err on (a) exploration: participation ratio of the state
            in 100-step windows, (b) collapse: share of late steps stuck (step-to-step cosine > 0.98), (c) self-model
            skill: how much better the mind's own self-model predicts its next state than "same as now".
            A ring that is more than feedback beats every control, and frozen_err loses the difference after freezing.
          (Windows scale with the run: 100-step windows, the last 200 steps, a 100-step lag at 600 steps.)
  secondary ring_pred against content and yoked_pred (self-fulfilment vs plain feedback); every condition against free;
            privileged self-knowledge: a predictor fitted to a mind's own first 400 steps against one fitted to another
            mind of the same condition, both tested on that mind's last 200 steps.
All comparisons are paired by seed (rows with the same seed share every random draw).
"""
import json, os, sys
import numpy as np

d = sys.argv[1]                                                         # one run, or a folder of chunks (seeds split)
parts = [d] if os.path.exists(os.path.join(d, 'ring.npz')) else sorted(
    os.path.join(d, x) for x in os.listdir(d) if os.path.exists(os.path.join(d, x, 'ring.npz')))
Ms = [json.load(open(os.path.join(p, 'meta.json'), encoding='utf-8')) for p in parts]
Z = {}
for p in parts:
    with np.load(os.path.join(p, 'ring.npz')) as z:
        for k in ('C', 'idx', 'coef', 'pred_cos', 'step_cos', 'loss', 'sig_cos', 'err_rel'):Z.setdefault(k, []).append(z[k])
Z = {k:np.concatenate(v, 1) for k, v in Z.items()}
M = dict(Ms[0], rows=sum((m['rows'] for m in Ms), []), glimpse=sum((m.get('glimpse') or ['']*len(m['rows']) for m in Ms), []))
rows, a = M['rows'], M['args'];T, B = Z['step_cos'].shape;TF = a['t_freeze']
assert len({(r['cond'], r['seed']) for r in rows}) == B, 'chunks share seed labels'
conds = list(dict.fromkeys(r['cond'] for r in rows));seeds = sorted({r['seed'] for r in rows})
row = {(r['cond'], r['seed']):i for i, r in enumerate(rows)}
C = Z['C'].astype(np.float32)                                          # (T, B, d)
U = C/np.maximum(np.linalg.norm(C, axis=-1, keepdims=True), 1e-8)
words = np.array(M['words'])


def paired(x, y):
    dd = np.asarray(x, float)-np.asarray(y, float);n = len(dd);se = dd.std(ddof=1)/np.sqrt(n) if n > 1 else 0.
    return dict(diff=float(dd.mean()), se=float(se), t=float(dd.mean()/se) if se > 0 else 0., pos=int((dd > 0).sum()), n=n)


def pr(V):                                                              # participation ratio of rows of V (n, d)
    V = V-V.mean(0);G = V@V.T;ev = np.clip(np.linalg.eigvalsh(G), 0, None)
    return float(ev.sum()**2/max((ev**2).sum(), 1e-12))


W = min(100, max(10, T//6));LT = min(200, T//3);LG = min(100, T//4)        # window, late phase, long lag (600 steps: 100/200/100)
late = slice(T-LT, T)
per = {}                                                                # per condition: arrays over seeds
for c in conds:
    bs = [row[c, s] for s in seeds]
    out = dict(
        explore=[np.mean([pr(U[w:w+W, b]) for w in range(T-LT, T-W+1, max(1, W//2))]) for b in bs],
        stuck=[float((Z['step_cos'][late, b] > .98).mean()) for b in bs],
        lag1=[float(Z['step_cos'][late, b].mean()) for b in bs],
        lag100=[float((U[T-LG:, b]*U[T-2*LG:T-LG, b]).sum(-1).mean()) for b in bs],
        skill=[float((Z['pred_cos'][late, b]-Z['step_cos'][late, b]).mean()) for b in bs],
        pred=[float(Z['pred_cos'][late, b].mean()) for b in bs],
        selffulfil=[float(Z['sig_cos'][late, b].mean()) for b in bs],
        words=[len(set(Z['idx'][:, b, 0].tolist())) for b in bs],
        explore_pre=[np.mean([pr(U[w:w+W, b]) for w in range(max(0, TF-LT), TF-W+1, max(1, W//2))]) for b in bs],
        stuck_pre=[float((Z['step_cos'][max(1, TF-LT):TF, b] > .98).mean()) for b in bs])
    per[c] = out

tests = {}
P = lambda c1, c2, k:paired(per[c1][k], per[c2][k]) if c1 in per and c2 in per else None
for ctrl in ('vel', 'yoked_err', 'shuf_err', 'rand_err', 'free', 'content'):
    tests[f'ring_err_vs_{ctrl}'] = {k:P('ring_err', ctrl, k) for k in ('explore', 'stuck', 'skill', 'lag100', 'words')}
for ctrl in ('content', 'yoked_pred', 'free', 'vel'):
    tests[f'ring_pred_vs_{ctrl}'] = {k:P('ring_pred', ctrl, k) for k in ('explore', 'stuck', 'skill', 'lag100', 'selffulfil')}
for c in conds:
    if c != 'free':tests[f'{c}_vs_free'] = {k:P(c, 'free', k) for k in ('explore', 'stuck', 'skill')}
# freezing: the change after the freeze, frozen minus its ring_err twin (identical until the freeze)
if 'frozen_err' in per and 'ring_err' in per:
    dE = np.array(per['frozen_err']['explore'])-np.array(per['frozen_err']['explore_pre'])
    dR = np.array(per['ring_err']['explore'])-np.array(per['ring_err']['explore_pre'])
    dS = np.array(per['frozen_err']['stuck'])-np.array(per['frozen_err']['stuck_pre'])
    dSR = np.array(per['ring_err']['stuck'])-np.array(per['ring_err']['stuck_pre'])
    tests['freeze_effect'] = dict(explore=paired(dE, dR), stuck=paired(dS, dSR),
                                  skill=paired(per['frozen_err']['skill'], per['ring_err']['skill']))

# privileged self-knowledge: own-fitted vs other-fitted linear predictor of the next state, in a shared 128-dim PCA space
Xall = C[::3].reshape(-1, C.shape[-1]);mu = Xall.mean(0)
_, _, Vt = np.linalg.svd((Xall-mu)[::max(1, len(Xall)//6000)], full_matrices=False);Pc = Vt[:128]
Y = (C-mu)@Pc.T                                                         # (T, B, 128)


def fit(b, t0, t1, lam=1.):
    X0, X1 = Y[t0:t1-1, b], Y[t0+1:t1, b];A = X0.T@X0+lam*np.eye(X0.shape[1])
    return np.linalg.solve(A, X0.T@X1)


def score(Wm, b, t0, t1):
    pred = Y[t0:t1-1, b]@Wm;tgt = Y[t0+1:t1, b]
    cs = (pred*tgt).sum(-1)/np.maximum(np.linalg.norm(pred, axis=-1)*np.linalg.norm(tgt, axis=-1), 1e-8)
    return float(cs.mean())


priv = {}
for c in [x for x in ('free', 'ring_err', 'ring_pred', 'content', 'vel', 'yoked_err') if x in per]:
    own, oth = [], []
    for i, s in enumerate(seeds):
        b = row[c, s];bo = row[c, seeds[(i+1) % len(seeds)]]
        own.append(score(fit(b, 0, T-LT), b, T-LT, T));oth.append(score(fit(bo, 0, T-LT), b, T-LT, T))
    priv[c] = dict(own=float(np.mean(own)), other=float(np.mean(oth)), test=paired(own, oth))

# what is on the minds' minds: top words at a few moments, and the final glimpse in words
def top(t, b, n=5):
    o = np.argsort(-Z['coef'][t, b].astype(float))[:n]
    return [str(words[Z['idx'][t, b, j]]) for j in o if Z['coef'][t, b, j] > 0]


examples = {c:{'words':[[top(t, row[c, s]) for t in (0, T//4, T//2, 3*T//4, T-1)] for s in seeds[:4]],
               'glimpse':[M['glimpse'][row[c, s]] if M.get('glimpse') else '' for s in seeds[:4]]} for c in conds}
TW = max(1, T//24)
timecourse = {c:dict(skill=[float((Z['pred_cos'][w:w+TW, [row[c, s] for s in seeds]]-Z['step_cos'][w:w+TW, [row[c, s] for s in seeds]]).mean())
                            for w in range(1, T-TW+1, TW)],
                     lag1=[float(Z['step_cos'][w:w+TW, [row[c, s] for s in seeds]].mean()) for w in range(1, T-TW+1, TW)], window=TW)
              for c in conds}
summary = {c:{k:float(np.mean(v)) for k, v in per[c].items()} for c in conds}
res = dict(model=a['model'], T=T, t_freeze=TF, seeds=len(seeds), L_in=M['L_in'], L_read=M['L_read'], n_layers=M['n_layers'],
           g=rows[0]['g'], s=rows[0]['s'], sigma=rows[0]['sigma'], summary=summary, tests=tests, privileged=priv,
           examples=examples, timecourse=timecourse)
json.dump(res, open(os.path.join(d, 'results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

fmt = lambda t:'—' if not t else f"{t['diff']:+.3f}±{t['se']:.3f} ({t['pos']}/{t['n']})"
print(f"{a['model']}  T {T}  seeds {len(seeds)}  g {rows[0]['g']} s {rows[0]['s']} sigma {rows[0]['sigma']}")
print('\ncondition    explore  stuck  lag1   lag100  skill   pred   selffulfil words')
for c in conds:
    v = summary[c];print(f"{c:11s} {v['explore']:7.2f} {v['stuck']:6.3f} {v['lag1']:6.3f} {v['lag100']:7.3f} {v['skill']:+7.3f} {v['pred']:6.3f} {v['selffulfil']:8.3f} {v['words']:6.1f}")
print('\nprimary: ring_err against each control (explore, stuck, skill, lag100)')
for k, v in tests.items():
    if v is None:continue
    if k.startswith('ring_') or k == 'freeze_effect':print(f"  {k:22s} " + '  '.join(f"{kk} {fmt(vv)}" for kk, vv in v.items() if vv))
print('\nprivileged self-knowledge (own-fitted vs other-fitted predictor on own last 200 steps)')
for c, v in priv.items():print(f"  {c:10s} own {v['own']:.3f} other {v['other']:.3f}  {fmt(v['test'])}")
print('\nwords on its mind (ring_err, seed 0):', examples.get('ring_err', {}).get('words', [[]])[0])
print('glimpses:', {c:examples[c]['glimpse'][:2] for c in ('free', 'ring_err', 'ring_pred') if c in examples})
