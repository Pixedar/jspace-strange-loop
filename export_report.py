"""Compact the analyses of the runs into the data files the report page loads.

    python export_report.py report/data runs/mainA runs/mainB runs/mainC
writes report/data/stats.json (every run's numbers), report/data/path_<run>.json (the explorer's data, for runs
that used the English dictionary) and report/index.html (report/template.html with the numbers inlined).
"""
import json, os, sys

out_dir, runs = sys.argv[1], sys.argv[2:]
os.makedirs(out_dir, exist_ok=True)
r3 = lambda x:[round(float(v), 3) for v in x]
stats = {}
for d in runs:
    name = os.path.basename(d.rstrip('/')).replace('main', '')
    R = json.load(open(f'{d}/results.json', encoding='utf-8'))
    keep = {k:R[k] for k in ('args', 'conds', 'seeds', 'N', 'NR', 'K', 't_int', 'rc', 'Rn', 'lags', 'fidelity', 'prediction_tests',
                             'coherence_tests', 'selfmodel_tests', 'pca_explained', 'generic_norm_share') if k in R}
    keep['acf'] = {k:dict(mean=r3(v['mean']), se=r3(v['se']), chance=round(v['chance'], 3)) for k, v in R['acf'].items()}
    keep['coh_time'] = {c:r3(v) for c, v in R['coh_time'].items()}
    keep['cross_time'] = {c:r3(v) for c, v in R['cross_time'].items()}
    sm = {}
    for c, v in R['selfmodel'].items():
        mean = lambda x:round(sum(x)/len(x), 4) if x and all(e == e for e in x) else None
        sm[c] = dict(cons=mean(v['cons_pre']), content=mean(v['wcons_pre']), cross=round(v['cross_pre'], 4),
                     recon=mean(v['recon']), generic=mean(v['generic_share']), cons_ind=mean(v.get('cons_ind_pre', [])),
                     cross_ind=round(v['cross_ind_pre'], 4) if 'cross_ind_pre' in v else None,
                     recon_ind=mean(v.get('recon_ind', [])), cons_time=r3(v['cons_time']),
                     cons_ind_time=r3(v.get('cons_ind_time', [])))
    keep['selfmodel'] = sm
    keep['prediction'] = {c:{k:round(sum(x)/len(x), 4) for k, x in v.items() if x} for c, v in R['prediction'].items()}
    if 'forward' in R:
        keep['forward'] = {c:{k:round(sum(x)/len(x), 4) for k, x in v.items()} for c, v in R['forward'].items()}
        keep['forward_tests'] = R['forward_tests']
    for grp in ('perturb', 'ablate', 'clamp'):
        if grp in R:
            g = R[grp];keep[grp] = {k:(dict(mean=r3(v['mean']), se=r3(v['se'])) if isinstance(v, dict) and 'mean' in v else v)
                                    for k, v in g.items()}
    keep['themes'] = R['themes']
    stats[name] = keep
    if R['args'].get('vocab', 'words') == 'english':
        P = dict(run=name, rows=R['rows'], inj_src=R['inj_src'], themes=R['themes'], theme_of=R['theme_of'],
                 explained=R['pca_explained'], t_int=R['t_int'], K=R['K'], clamp_len=R['args'].get('clamp_len'),
                 path=[[[round(p[0], 2), round(p[1], 2)] for p in row] for row in R['path_z']],
                 points=[[[round(p[0], 2), round(p[1], 2)] for p in row] for row in R['path']],
                 selfpts=[[[round(p[0], 2), round(p[1], 2)] for p in row] for row in R['path_m']],
                 words=[[w[:4] for w in row] for row in R['thought_words']],
                 texts=[[t[:150] for t in row] for row in R['texts']],
                 self_words=[[w[:6] for w in row] for row in R['self_words']],
                 answers=[[t[:110] for t in row] for row in R['answers']],
                 fanswers=[[t[:90] for t in row] for row in R.get('forward_answers', [])])
        with open(f'{out_dir}/path_{name}.json', 'w', encoding='utf-8') as f:json.dump(P, f, ensure_ascii=False, separators=(',', ':'))
with open(f'{out_dir}/stats.json', 'w', encoding='utf-8') as f:json.dump(stats, f, ensure_ascii=False, separators=(',', ':'))
tpl = os.path.join(os.path.dirname(out_dir.rstrip('/')), 'template.html')     # the page, with the numbers inlined
if os.path.exists(tpl):
    with open(tpl, encoding='utf-8') as f:page = f.read().replace('__STATS__', json.dumps(stats, ensure_ascii=False, separators=(',', ':')))
    with open(os.path.join(os.path.dirname(tpl), 'index.html'), 'w', encoding='utf-8') as f:f.write(page)
for fn in sorted(os.listdir(out_dir)):print(fn, os.path.getsize(f'{out_dir}/{fn}')//1024, 'KB')
