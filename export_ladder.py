"""Pack the model-ladder results, the dictionary test and the lens checks for the report (report/data/ladder.json).

    python export_ladder.py runs/ladder07 runs/ladder2 runs/ladder_paper/qwen3-1.7b runs/ladder/qwen3-4b/diag/fidelity_check.json
(read at 70 % of depth; read at 50 %; the paper-dictionary run of 1.7B; the 4B dictionary test)
"""
import json, os, sys

root, root05, paper_dir, diag4 = sys.argv[1:5]
L = json.load(open(f'{root}/ladder.json', encoding='utf-8'))
L05 = json.load(open(f'{root05}/ladder.json', encoding='utf-8')) if os.path.exists(f'{root05}/ladder.json') else {}
keep = ('size', 'layers', 'L', 'L_in', 'g', 'g_r', 'fsweep', 'chance_f', 'gains', 'fidelity', 'lag1_free', 'lag20_free',
        'coherence_loop_vs_free', 'coherence_loop_vs_yoked', 'self_vs_content_free', 'ind_stab', 'ind_cross', 'own_symbol_loop',
        'loop_vs_yoked_cons_ind', 'loop_vs_yoked_recon_ind', 'loop_vs_yoked_cons', 'observer_own_vs_other',
        'observer_beyond_past', 'yoked_own_vs_injected', 'fwd_vs_back_free', 'fwd_lean_free', 'fwd_beyond_past_vs_other',
        'fwd_back_sim', 'clamp', 'lexicon', 'theme_shift', 'answers', 'texts')
models = {t:{k:v[k] for k in keep if k in v} for t, v in L.items()}
models05 = {t:{k:v[k] for k in keep if k in v} for t, v in L05.items()}
val = {}
for t in L:
    p = f'{root}/{t}/val.json'
    if os.path.exists(p):
        V = json.load(open(p))
        val[t] = dict(n_layers=V['n_layers'], n_prompts=V.get('n_prompts'),
                      layers=[dict(layer=x['layer'], depth=round(x['depth'], 3), agree_J=round(x['agree_J'], 3),
                                   agree_logit=round(x['agree_logit'], 3), occupancy=x['occupancy'], excess_ve=round(x['excess_ve'], 4),
                                   ve_J25=round(x['ve_J25'], 4),
                                   probes={k:x['probe'][p_]['J'][:6] for k, p_ in zip(('boot', 'capital', 'ocean', 'self'), x['probe'])},
                                   boot_logit=x['probe'][list(x['probe'])[0]]['logit'][:6]) for x in V['layers']])
diag = {'qwen3-1.7b':json.load(open(f'{paper_dir}/diag/fidelity_check.json')), 'qwen3-4b':json.load(open(diag4))}
P = json.load(open(f'{paper_dir}/main/results.json', encoding='utf-8'))
paper = dict(fidelity=P['fidelity']['m'], coherence_loop_vs_free=P['coherence_tests']['loop_vs_free'],
             lag1_free=P['acf']['free/pre']['mean'][0], lag1_loop=P['acf']['loop/pre']['mean'][0],
             self_words=[w for w in P['self_words'][0][-1]][:8])
import numpy as np
pooled = json.load(open('pooled.json', encoding='utf-8')) if os.path.exists('pooled.json') else {}
own = {}
for t in sorted(os.listdir('runs/own')) if os.path.isdir('runs/own') else []:
    f = f'runs/own/{t}/results.json'
    if not os.path.exists(f):continue
    R = json.load(open(f, encoding='utf-8'));sm = R['selfmodel']
    mse = lambda v:(float(np.nanmean(v)), float(np.nanstd(v, ddof=1)/np.sqrt(np.sum(~np.isnan(v))))) if len(v) and (~np.isnan(v)).sum() > 1 else (None, None)
    cond = {}
    for c in ('free', 'loop', 'match', 'stale', 'yoked'):
        if c not in sm:continue
        st, st_se = mse(np.array(sm[c]['cons_ind_pre'], float));rc, rc_se = mse(np.array(sm[c].get('recon_inj', []), float))
        cond[c] = dict(stability=st, stability_se=st_se, recon=rc, recon_se=rc_se,
                       coherence=float(np.mean(R['acf'][f'{c}/pre']['long'])) if f'{c}/pre' in R['acf'] else None)
    own[t] = dict(cond=cond, tests=R.get('ownership_tests', {}), fidelity=R['fidelity']['m'])
out = dict(models07=models, models05=models05, validation=val, diag=diag, paper17=paper, pooled=pooled, ownership=own)
os.makedirs('report/data', exist_ok=True)
json.dump(out, open('report/data/ladder.json', 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'))
print('ladder.json', os.path.getsize('report/data/ladder.json')//1024, 'KB', list(models))
