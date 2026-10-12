"""Ouro analysis: accuracy per loop (1-4 trained, >4 out of distribution), memory after decision-state transplant and
after full-state swaps, sensitivity. Writes results_ouro.json.
    python an_ouro.py /workspace/v3/ouro/main"""
import sys, json, pathlib
import numpy as np

RNG = np.random.default_rng(811)


def boot(v, n=1000):
    v = np.asarray(v, float);v = v[np.isfinite(v)]
    if not len(v):return [np.nan]*3
    b = [np.mean(RNG.choice(v, len(v))) for _ in range(n)];return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main(root):
    root = pathlib.Path(root);ctx = json.loads((root/'contexts.json').read_text());res = {}
    nat = np.load(root/'natural.npz');lo = nat['lo'];K = lo.shape[0]
    fam = np.array([c['fam'] for c in ctx]);truth = np.array([c['truth'] for c in ctx]);tag = np.array([c['tag'] for c in ctx])
    acc = {}
    for f in np.unique(fam):
        m = (fam == f)&(truth >= 0) if f != 'F5' else (fam == f)
        want = (truth[m] == 0) if f != 'F5' else np.ones(m.sum(), bool)
        acc[f] = {k+1:boot(((lo[k, m] > 0) == want).astype(float)) for k in range(K)}
    res['accuracy_by_loop'] = acc
    res['decision_state_rel_change_by_loop'] = {k+2:float(np.median(nat['dlt'][k])) for k in range(K-1)}
    mem = np.load(root/'memory.npz');trip = mem['trip'];tf = np.array([ctx[int(t[2])]['fam'] for t in trip]);M = {}
    for k0 in (0, 1, 2):
        g = mem[f'k{k0}_N_from_A']-mem[f'k{k0}_N_from_B']
        M[f'k{k0}_transplant_gap'] = {f:{k+1:boot(g[k, tf == f]) for k in range(K)} for f in np.unique(tf)}
        if f'k{k0}_A_with_B_state' in mem.files:
            M[f'k{k0}_swap'] = {k+1:dict(A_run_with_B_state=float(np.nanmean(mem[f'k{k0}_A_with_B_state'][k])),
                                         B_run_with_A_state=float(np.nanmean(mem[f'k{k0}_B_with_A_state'][k]))) for k in range(K)}
    res['memory'] = M
    res['natural_final_logodds_by_tag'] = {f:{t:float(np.nanmean(lo[3, (fam == f)&(tag == t)])) for t in ('A', 'B', 'N')} for f in np.unique(fam) if f != 'F5'}
    se = np.load(root/'sensitivity.npz')
    res['sensitivity'] = dict(median_rel_dist={k+1:float(np.median(se['dist'][k])) for k in range(K)},
                              mean_abs_dlogodds={k+1:float(np.nanmean(np.abs(se['dlo'][k]))) for k in range(K)})
    (root/'results_ouro.json').write_text(json.dumps(res, indent=1, default=float));print(json.dumps(res, default=float)[:2500])


if __name__ == '__main__':
    main(sys.argv[1])
