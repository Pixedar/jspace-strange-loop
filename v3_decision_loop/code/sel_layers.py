"""Apply the pre-declared loop-layer rule (protocol amendment 10.3/10.5) to the validation sweeps sel_<Lr>_<Lw>:
largest mean |lo(N <- code_A) - lo(N <- code_B)| at step 1 of the J loop at g = 1 over F2/F3/F6 scenarios; ties within
10 % go to the lower write layer. Also reports signed gaps over time per family (descriptive).
    python sel_layers.py /workspace/v3/e2/qwen3-8b"""
import sys, json, pathlib
import numpy as np

root = pathlib.Path(sys.argv[1]);res = {}
for d in sorted(root.glob('sel_*_*')):
    Lr, Lw = map(int, d.name.split('_')[1:]);ctxs = json.loads((d/'contexts.json').read_text())
    gaps = {}
    for f in sorted((d/'A1').glob('A1_mem_*.npz')):
        z = np.load(f);m = json.loads(f.with_suffix('.json').read_text())
        for g in sorted(set(r['g'] for r in m['rows'] if r['cond'] == 'J')):
            by = {}
            for i, r in enumerate(m['rows']):
                if r['cond'] != 'J' or r['g'] != g:continue
                u = m['units'][r['unit']];by.setdefault((u['fam'], u['sid']), {})[u['tag']] = z['lo'][:, i]
            for (fam, sid), v in by.items():
                if 'A' in v and 'B' in v:gaps.setdefault(g, {}).setdefault(fam, []).append(v['A']-v['B'])
    out = {}
    for g, fams in gaps.items():
        allg = np.concatenate([np.array(v) for v in fams.values()])
        out[g] = dict(metric_abs_step1=float(np.abs(allg[:, 1]).mean()),
                      by_family={f:dict(abs_step1=float(np.abs(np.array(v)[:, 1]).mean()),
                                        signed=[float(np.array(v)[:, t].mean()) for t in (0, 1, 4, 16, 64) if t < np.array(v).shape[1]])
                                 for f, v in fams.items()})
    res[f'{Lr}_{Lw}'] = dict(Lr=Lr, Lw=Lw, gains=out)
    print(Lr, Lw, {g:round(v['metric_abs_step1'], 3) for g, v in out.items()})
    for g, v in out.items():
        if g == 1.0:print('   g=1 by family', {f:(round(x['abs_step1'], 3), [round(s, 2) for s in x['signed']]) for f, x in v['by_family'].items()})
cand = {k:v['gains'].get(1.0, {}).get('metric_abs_step1', 0.) for k, v in res.items()}
best = max(cand.values());near = [k for k, v in cand.items() if v >= .9*best]
choice = min(near, key=lambda k:(res[k]['Lw'], res[k]['Lr']))
res['choice'] = dict(pair=choice, metric=cand, rule='max mean |gap| at step 1, g=1; ties within 10% -> lower Lw')
(root/'layer_choice.json').write_text(json.dumps(res, indent=1))
print('CHOICE', choice, cand)
