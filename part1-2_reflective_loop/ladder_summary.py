"""One table across the model ladder: does anything about the loop change with scale?

    python ladder_summary.py runs/ladder2 > runs/ladder2/summary.txt   (also writes runs/ladder2/ladder.json)
"""
import json, os, sys
import numpy as np

root = sys.argv[1]
SIZES = {'qwen3-1.7b':1.7, 'qwen3-4b':4, 'qwen3-8b':8, 'qwen3-14b':14, 'qwen3-32b':32}
mean = lambda v:float(np.nanmean(v)) if v is not None and len(v) else float('nan')
out = {}
for tag, size in SIZES.items():
    d = f'{root}/{tag}'
    if not os.path.exists(f'{d}/main/results.json'):continue
    R = json.load(open(f'{d}/main/results.json', encoding='utf-8'));P = json.load(open(f'{d}/params.json'))
    G = json.load(open(f'{d}/gain.json'));V = json.load(open(f'{d}/val.json')) if os.path.exists(f'{d}/val.json') else None
    M = json.load(open(f'{d}/main/meta.json', encoding='utf-8'))
    sm, pt, ct, st = R['selfmodel'], R['prediction_tests'], R['coherence_tests'], R['selfmodel_tests']
    fw, cl = R.get('forward_tests', {}), R.get('clamp', {}).get('tests', {})
    vl = None
    if V:
        near = min(V['layers'], key=lambda x:abs(x['layer']-M['layer']))
        vl = dict(layer=near['layer'], agree_J=near['agree_J'], agree_logit=near['agree_logit'], occupancy=near['occupancy'],
                  excess_ve=near['excess_ve'], boot=near['probe'][list(near['probe'])[0]]['J'][:6],
                  band=[(x['layer'], x['occupancy'], round(x['excess_ve'], 3)) for x in V['layers']])
    out[tag] = dict(
        size=size, layers=M['n_layers'], L=M['layer'], L_in=M['L_in'], g=P['g'], g_r=P['g_r'], fsweep=P['fidelity'], chance_f=P['chance'],
        gains={k:round(v['lag1'], 3) for k, v in G['gains'].items()}, validation=vl,
        fidelity=R['fidelity']['m'],
        lag1_free=R['acf']['free/pre']['mean'][0], lag20_free=R['acf']['free/pre']['mean'][19],
        coherence_loop_vs_free=ct['loop_vs_free'], coherence_loop_vs_yoked=ct['loop_vs_yoked'],
        self_vs_content_free=st['free_self_vs_content_stability'],
        ind_stab={c:mean(sm[c].get('cons_ind_pre')) for c in ('free', 'loop', 'yoked')},
        ind_cross={c:sm[c].get('cross_ind_pre') for c in ('free', 'loop', 'yoked')},
        own_symbol_loop=st['loop_own_vs_other_symbol'],
        loop_vs_yoked_cons_ind=st['loop_vs_yoked_cons_ind'], loop_vs_yoked_recon_ind=st['loop_vs_yoked_recon_ind'],
        loop_vs_yoked_cons=st['loop_vs_yoked_cons'],
        observer_own_vs_other=pt['free']['own_vs_other'], observer_beyond_past=pt['free']['perp_own_vs_other'],
        yoked_own_vs_injected=pt.get('yoked_own_vs_injected'),
        fwd_vs_back_free=fw.get('free', {}).get('fwd_vs_back'), fwd_lean_free=fw.get('free', {}).get('lean_fwd_vs_back'),
        fwd_beyond_past_vs_other=fw.get('free', {}).get('perp_fwd_vs_other'),
        fwd_back_sim=R.get('forward', {}).get('free', {}).get('fwd_back_sim'),
        clamp=cl, lexicon=R.get('lexicon'),
        theme_shift=sorted([dict(words=t['words'][:5], loop_minus_free=round(t['share']['loop']-t['share']['free'], 3)) for t in R['themes']],
                           key=lambda x:x['loop_minus_free']),
        answers=[R['answers'][b][-1] for b in range(0, len(R['rows']), max(1, len(R['rows'])//4))],
        texts=[R['texts'][b][-1] for b in range(0, len(R['rows']), max(1, len(R['rows'])//4))])
json.dump(out, open(f'{root}/ladder.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

f3 = lambda t:'—' if not t else f"{t['diff']:+.3f}±{t['se']:.3f} ({t['pos']}/{t['n']})"
fx = lambda v:f'{v:.2f}' if isinstance(v, (int, float)) and v == v else '—'
rows = [('size (B)', lambda o:fx(o['size'])), ('read layer / trace layer', lambda o:f"{o['L']}/{o['L_in']} of {o['layers']}"),
        ('gain g (lag-1 of observers by g)', lambda o:f"{o['g']}  {o['gains']}"),
        ('fidelity (own window among minds)', lambda o:f"{o['fidelity']['retrieval']:.2f} (chance {o['fidelity']['chance']:.2f})"),
        ('observer lag-1 / lag-20', lambda o:f"{o['lag1_free']:.2f} / {o['lag20_free']:.2f}"),
        ('coherence loop − free', lambda o:f3(o['coherence_loop_vs_free'])),
        ('coherence loop − yoked', lambda o:f3(o['coherence_loop_vs_yoked'])),
        ('self vs content stability (observers)', lambda o:f3(o['self_vs_content_free'])),
        ('individual self stability free/loop/yoked', lambda o:' / '.join(fx(o['ind_stab'][c]) for c in ('free', 'loop', 'yoked'))),
        ('individual self: own vs other minds (loop)', lambda o:f3(o['own_symbol_loop'])),
        ('loop − yoked, individual stability', lambda o:f3(o['loop_vs_yoked_cons_ind'])),
        ('loop − yoked, reconstruction', lambda o:f3(o['loop_vs_yoked_recon_ind'])),
        ('observer self-model: own − other minds', lambda o:f3(o['observer_own_vs_other'])),
        ('  … beyond the past', lambda o:f3(o['observer_beyond_past'])),
        ('yoked: own (unused) − injected foreign', lambda o:f3(o['yoked_own_vs_injected'])),
        ('forward − backward question (observers)', lambda o:f3(o['fwd_vs_back_free'])),
        ('forward ~ backward cosine', lambda o:fx(o['fwd_back_sim'])),
        ('clamp: after release, self → foreign', lambda o:f3(o['clamp'].get('after_self_to_foreign'))),
        ('clamp: after release, self → old', lambda o:f3(o['clamp'].get('after_self_to_old'))),
        ('clamp: late, self → foreign', lambda o:f3(o['clamp'].get('late_self_to_foreign'))),
        ('identity words in self-models free/loop', lambda o:' / '.join(fx(o['lexicon'][c]['ident_self']*100) + '%' for c in ('free', 'loop'))),
        ('experiential words in thoughts free/loop', lambda o:' / '.join(fx(o['lexicon'][c]['exper_thought']*100) + '%' for c in ('free', 'loop')))]
tags = list(out)
print('measure'.ljust(44) + ''.join(t.ljust(30) for t in tags))
for name, fn in rows:print(name.ljust(44) + ''.join(fn(out[t]).ljust(30) for t in tags))
for t in tags:
    o = out[t];print(f'\n== {t}: lens check at layer {o["validation"]["layer"] if o["validation"] else "?"}', o['validation'] and {k:o['validation'][k] for k in ('agree_J', 'agree_logit', 'occupancy', 'excess_ve', 'boot')})
    print('   themes the loop gains:', [x['words'] for x in o['theme_shift'][-3:]], '| loses:', [x['words'] for x in o['theme_shift'][:3]])
    for s_ in o['texts'][:2]:print('   thought:', s_[:140])
    for s_ in o['answers'][:2]:print('   answer :', s_[:140])
