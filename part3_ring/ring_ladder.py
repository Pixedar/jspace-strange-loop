"""Across models: the ring's pre-registered tests from each results.json, side by side, plus Stouffer's z over models.

    python ring_ladder.py runs/ring            -> runs/ring/ladder.json and a table on stdout
"""
import json, math, os, sys

root = sys.argv[1]
ORDER = ['qwen3-1.7b', 'qwen3-4b', 'qwen3-8b', 'qwen3-14b', 'qwen3-4b-long']
models = [m for m in ORDER if os.path.exists(f'{root}/{m}/main/results.json')]
R = {m:json.load(open(f'{root}/{m}/main/results.json', encoding='utf-8')) for m in models}
PRIMARY = [('ring_err_vs_vel', ('explore', 'stuck', 'skill', 'lag100')), ('ring_err_vs_yoked_err', ('explore', 'stuck', 'skill', 'lag100')),
           ('ring_err_vs_shuf_err', ('explore', 'stuck', 'skill', 'lag100')), ('ring_err_vs_rand_err', ('explore', 'stuck', 'skill', 'lag100')),
           ('freeze_effect', ('explore', 'stuck', 'skill'))]
SECONDARY = [('ring_pred_vs_content', ('explore', 'stuck', 'skill', 'selffulfil')),
             ('ring_pred_vs_yoked_pred', ('explore', 'stuck', 'skill', 'selffulfil')),
             ('ring_err_vs_free', ('explore', 'stuck', 'skill')), ('ring_pred_vs_free', ('explore', 'stuck', 'skill'))]


def cell(t):
    return '      —      ' if not t else f"{t['diff']:+7.3f} t{t['t']:+6.1f}"


def stouffer(ts):
    ts = [t['t'] for t in ts if t and t['se'] > 0]
    return (sum(ts)/math.sqrt(len(ts)), len(ts)) if ts else (0., 0)


out = dict(models=models, regimes={}, tests={}, privileged={})
print('regime per model (pilot rule): g = s, sigma; free minds at that setting')
for m in models:
    r = R[m];out['regimes'][m] = dict(g=r['g'], sigma=r['sigma'], seeds=r['seeds'], T=r['T'], L_in=r['L_in'], L_read=r['L_read'])
    print(f"  {m:14s} g {r['g']:.2f} sigma {r['sigma']:.2f} | {r['seeds']} seeds, T {r['T']} | inject L{r['L_in']}, read L{r['L_read']} of {r['n_layers']}")
for title, block in (('PRIMARY', PRIMARY), ('SECONDARY', SECONDARY)):
    print(f'\n{title}: paired difference (first minus second), t over seeds; last column Stouffer z over the main models')
    print(f"{'test':26s} {'metric':8s} " + ' '.join(f'{m:>15s}' for m in models) + '   pooled z')
    for name, metrics in block:
        for k in metrics:
            ts = [R[m]['tests'].get(name, {}).get(k) if R[m]['tests'].get(name) else None for m in models]
            z, n = stouffer([t for m, t in zip(models, ts) if not m.endswith('-long')])
            out['tests'][f'{name}.{k}'] = dict(per_model=dict(zip(models, ts)), stouffer_z=z, n_models=n)
            print(f"{name:26s} {k:8s} " + ' '.join(f'{cell(t):>15s}' for t in ts) + f'   {z:+6.1f} ({n})')
print('\nprivileged self-knowledge: own-fitted minus other-fitted next-state predictor (cosine), t over seeds')
for c in ('free', 'ring_err', 'ring_pred', 'content', 'vel', 'yoked_err'):
    ts = [R[m]['privileged'].get(c, {}).get('test') for m in models]
    z, n = stouffer([t for m, t in zip(models, ts) if not m.endswith('-long')])
    out['privileged'][c] = dict(per_model=dict(zip(models, ts)), stouffer_z=z)
    print(f"{c:12s} " + ' '.join(f'{cell(t):>15s}' for t in ts) + f'   {z:+6.1f} ({n})')
print('\nper-condition means (explore / stuck / skill)')
conds = list(R[models[0]]['summary']) if models else []
for c in conds:
    print(f"{c:11s} " + ' '.join(f"{R[m]['summary'][c]['explore']:6.2f}/{R[m]['summary'][c]['stuck']:.2f}/{R[m]['summary'][c]['skill']:+.3f}"
                                 for m in models))
# stage 2 (exploratory): the gain ladders, each placed against its own critical gain g*, where the free minds' spectral
# exponent crosses 1 (interpolated in log g)
G = {m:json.load(open(f'{root}/{m}/gains/gains.json', encoding='utf-8')) for m in models if os.path.exists(f'{root}/{m}/gains/gains.json')}
out['gains'] = {}
if G:
    print('\nSTAGE 2 (exploratory): the ring across each substrate\'s transition; g* = gain where free beta crosses 1')
for m, L in G.items():
    gs = [r['g'] for r in L['gains']];fb = [r['conds']['free']['mean']['beta'] for r in L['gains']];gstar = None
    for (g0, b0), (g1, b1) in zip(zip(gs, fb), zip(gs[1:], fb[1:])):
        if b0 < 1 <= b1:gstar = math.exp(math.log(g0)+(1-b0)/(b1-b0)*(math.log(g1)-math.log(g0)));break
    print(f'\n{m}: picked g {L["picked_g"]}, g* {gstar and round(gstar, 3)}')
    print(f"{'g':>8s} {'g/g*':>5s} {'freeβ':>6s} {'stuck free/content/ring_err/vel':>32s} | ring_err − vel, − yoked, − shuf, − rand (explore) | freeze | sync s/m")
    rows = []
    for r in L['gains']:
        c, t = r['conds'], r['tests']
        d = lambda k:t[k]['explore']['diff'] if k in t else float('nan')
        row = dict(g=r['g'], rel=r['g']/gstar if gstar else None, free_beta=c['free']['mean']['beta'],
                   stuck={k:c[k]['mean']['stuck'] for k in ('free', 'content', 'ring_err', 'vel')},
                   vs={k:d(f'ring_err_vs_{k}') for k in ('vel', 'yoked_err', 'shuf_err', 'rand_err')}, freeze=d('freeze_effect'),
                   sync=(c['ring_err']['sync_state'], c['ring_err']['sync_motion']))
        rows.append(row)
        print(f"{r['g']:8.4f} {row['rel'] or 0:5.2f} {row['free_beta']:6.2f} " + '/'.join(f"{v:.2f}" for v in row['stuck'].values()).rjust(32)
              + ' | ' + '  '.join(f"{v:+6.2f}" for v in row['vs'].values()) + f"   | {row['freeze']:+6.2f} | {row['sync'][0]:.2f}/{row['sync'][1]:.2f}")
    out['gains'][m] = dict(gstar=gstar, picked=L['picked_g'], rows=rows)
json.dump(out, open(f'{root}/ladder.json', 'w', encoding='utf-8'), indent=1)
