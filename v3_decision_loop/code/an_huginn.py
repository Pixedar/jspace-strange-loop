"""Huginn natural-recurrence analysis. Writes results_huginn.json next to the arrays.
    python an_huginn.py /workspace/v3/huginn/main"""
import sys, json, pathlib, collections
import numpy as np

RNG = np.random.default_rng(811)


def boot(v, n=1000):
    v = np.asarray(v, float);v = v[np.isfinite(v)]
    if not len(v):return [np.nan]*3
    b = [np.mean(RNG.choice(v, len(v))) for _ in range(n)];return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main(root):
    root = pathlib.Path(root);ctx = json.loads((root/'contexts.json').read_text());res = {}
    nat = np.load(root/'natural.npz');lo0, lo1, dlt0 = nat['lo0'], nat['lo1'], nat['dlt0'];K = lo0.shape[0]-1
    fam = np.array([c['fam'] for c in ctx]);truth = np.array([c['truth'] for c in ctx]);tag = np.array([c['tag'] for c in ctx])
    ks = [k for k in (0, 1, 2, 4, 8, 16, 32, 48, 64) if k <= K]
    acc = {}
    for f in np.unique(fam):
        m = (fam == f)&(truth >= 0) if f != 'F5' else (fam == f)
        want = (truth[m] == 0) if f != 'F5' else np.ones(m.sum(), bool)
        acc[f] = {k:boot(((lo0[k, m] > 0) == want).astype(float)) for k in ks}
    res['accuracy_by_recurrence'] = acc
    res['init_seed_gap_abs_logodds'] = {k:float(np.nanmedian(np.abs(lo0[k]-lo1[k]))) for k in ks}
    res['decision_state_rel_change'] = {k:float(np.median(dlt0[k-1])) for k in ks if k >= 1}
    # late-window period check of the decision-state change sequence (fixed point if change < 1e-3)
    late = dlt0[-16:];res['fraction_converged_late'] = float(np.mean(late.max(0) < 1e-3))
    res['fraction_converged_late_1e-2'] = float(np.mean(late.max(0) < 1e-2))
    # evidence decodability from the decision state across recurrences (A vs B, held-out scenarios)
    xs = nat['xs0'].astype(np.float32);ab = np.where(np.isin(tag, ['A', 'B'])&(fam != 'F5'))[0]
    y = (tag[ab] == 'A').astype(int);sid = np.array([ctx[i]['sid'] for i in ab]);us = np.unique(sid)
    folds = np.array_split(RNG.permutation(us), 5);dec = {}
    from sklearn.linear_model import LogisticRegression
    for k in ks:
        X0 = xs[k, ab];ok = []
        for fo in folds:
            te = np.isin(sid, fo);m_, s_ = X0[~te].mean(0), X0[~te].std(0)+1e-6;X = (X0-m_)/s_
            clf = LogisticRegression(C=0.01, max_iter=500).fit(X[~te], y[~te]);ok += list(clf.predict(X[te]) == y[te])
        dec[k] = float(np.mean(ok))
    res['evidence_decodability_from_decision_state'] = dec
    mem = np.load(root/'memory.npz');trip = mem['trip']
    tf = np.array([ctx[int(t[2])]['fam'] for t in trip]);M = {}
    for k0 in (4, 16, 32):
        if f'k{k0}_N_from_A' not in mem.files:continue
        g = mem[f'k{k0}_N_from_A']-mem[f'k{k0}_N_from_B'];d = {}
        for f in np.unique(tf):
            d[f] = {t:boot(g[t, tf == f]) for t in sorted({k0, k0+1, k0+2, k0+4, k0+8, k0+16, K}) if t <= K}
        M[f'k{k0}_transplant_gap'] = d
        if f'k{k0}_A_with_B_state' in mem.files:
            a = mem[f'k{k0}_A_with_B_state'];b = mem[f'k{k0}_B_with_A_state']
            M[f'k{k0}_swap'] = {t:dict(A_run_with_B_state=float(np.nanmean(a[t])), B_run_with_A_state=float(np.nanmean(b[t])))
                                for t in sorted({k0, k0+1, k0+4, k0+16, K}) if t <= K}
    res['memory'] = M
    nat_ab = {f:dict(A=float(np.nanmean(lo0[K, (fam == f)&(tag == 'A')])), B=float(np.nanmean(lo0[K, (fam == f)&(tag == 'B')])),
                     N=float(np.nanmean(lo0[K, (fam == f)&(tag == 'N')]))) for f in np.unique(fam) if f != 'F5'}
    res['natural_final_logodds_by_tag'] = nat_ab
    se = np.load(root/'sensitivity.npz');dist = se['dist'];dlo = se['dlo']
    res['sensitivity'] = dict(median_rel_dist={k:float(np.median(dist[k])) for k in ks if k >= 16},
                              mean_abs_dlogodds={k:float(np.mean(np.abs(dlo[k]))) for k in ks if k >= 16},
                              growth_ratio_K_over_17=float(np.median(dist[K]/np.maximum(dist[min(17, K)], 1e-9))))
    (root/'results_huginn.json').write_text(json.dumps(res, indent=1, default=float));print(json.dumps(res, indent=1, default=float)[:3000])


if __name__ == '__main__':
    main(sys.argv[1])
