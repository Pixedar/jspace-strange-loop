"""Pool one paired effect across independent runs (inverse-variance weighted mean of the per-run seed-paired differences).

Each run (model x read depth, plus Part 1's runs) is an independent experiment with its own minds, so a small effect
that no single run can detect can still show up in the pooled estimate.

    python pooled.py runs/ladder07 runs/ladder2 runs/own runs/mainA runs/mainB runs/mainC
"""
import json, math, os, sys

TESTS = {'individual self stability, loop − yoked':('selfmodel_tests', 'loop_vs_yoked_cons_ind'),
         'reconstruction of the fed-back self, loop − yoked':('selfmodel_tests', 'loop_vs_yoked_recon_ind'),
         'self-model stability, loop − yoked':('selfmodel_tests', 'loop_vs_yoked_cons'),
         'long-range coherence, loop − yoked':('coherence_tests', 'loop_vs_yoked'),
         'unused own self − injected foreign (prediction)':('prediction_tests', 'yoked_own_vs_injected'),
         'forward − backward question (observers)':('forward_tests', 'free', 'fwd_vs_back'),
         'observer self-model beyond the past, own − other':('prediction_tests', 'free', 'perp_own_vs_other'),
         'own − most similar stranger (stability)':('ownership_tests', 'loop_vs_match', 'stability'),
         'own outdated − random stranger (stability)':('ownership_tests', 'stale_vs_yoked', 'stability'),
         'own − own outdated (stability)':('ownership_tests', 'loop_vs_stale', 'stability')}


def runs(paths):
    for p in paths:
        if os.path.exists(f'{p}/results.json'):yield os.path.basename(p), f'{p}/results.json'
        elif os.path.isdir(p):
            for t in sorted(os.listdir(p)):
                for f in (f'{p}/{t}/main/results.json', f'{p}/{t}/results.json'):   # ladders / ownership runs
                    if os.path.exists(f):yield f'{os.path.basename(p)}/{t}', f;break


out = {}
R = {name:json.load(open(f, encoding='utf-8')) for name, f in runs(sys.argv[1:])}
for label, path in TESTS.items():
    rows = []
    for name, r in R.items():
        x = r
        try:
            for k in path:x = x[k]
        except (KeyError, TypeError):continue
        if x.get('se', 0) > 0:rows.append((name, x['diff'], x['se'], x['pos'], x['n']))
    if not rows:continue
    w = [1/se**2 for _, _, se, _, _ in rows];m = sum(wi*d for wi, (_, d, _, _, _) in zip(w, rows))/sum(w);se = math.sqrt(1/sum(w))
    z = m/se;p = math.erfc(abs(z)/math.sqrt(2))
    pos = sum(1 for _, d, _, _, _ in rows if d > 0)
    out[label] = dict(pooled=m, se=se, z=z, p=p, runs_positive=pos, n_runs=len(rows), per_run=rows)
    print(f'{label}\n   pooled {m:+.4f} ± {se:.4f}  z {z:+.2f}  p {p:.4f}  ({pos}/{len(rows)} runs positive)')
    for name, d, s, ps, n in rows:print(f'      {name:24s} {d:+.3f} ± {s:.3f} ({ps}/{n})')
json.dump(out, open('pooled.json', 'w'), indent=1)
