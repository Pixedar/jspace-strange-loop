"""One readout of the workspace tests across models: ignition by depth band, trace conditioning and local-global under
ablation, criticality across temperature, self-talk attractors.

    python ws_summary.py runs/ws          -> runs/ws/summary.json and tables on stdout
"""
import json, os, sys
import numpy as np

root = sys.argv[1]
ORDER = ['qwen3-1.7b', 'qwen3-4b', 'qwen3-8b', 'qwen3-14b']
models = [m for m in ORDER if os.path.isdir(f'{root}/{m}')]
BANDS = (('early', 0, .4), ('middle', .4, .6), ('workspace', .6, .86), ('late', .86, 1.01))
f = lambda v, p='.2f':format(v, p) if isinstance(v, (int, float)) and v == v and v is not None else '  -  '
out = {}


def load(m, t):
    p = f'{root}/{m}/{t}/{t}.json'
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None


print('IGNITION: medians over the layers of each depth band (J = projection on the concept J-lens atom, lin = full residual)')
print('  step: share of the whole rise taken in one strength step (linear 0.03, all-or-none 1); BC > 0.555 and dBIC > 0: bimodal at threshold')
for m in models:
    r = load(m, 'ignition')
    if not r:continue
    n = r['n_layers'];rows = {}
    for name, lo, hi in BANDS:
        ls = [l for l in range(n) if lo <= l/n < hi];g = lambda key, sub:np.nanmedian([r['layers'][str(l)].get(key, {}).get(sub, np.nan) or np.nan
                                                                                      for l in ls]) if ls else np.nan
        bn = lambda sub:np.nanmedian([(r['layers'][str(l)].get('bottleneck') or {}).get(sub) or np.nan for l in ls]) if ls else np.nan
        rows[name] = dict(J_step=g('J_slot', 'step'), J_width=g('J_slot', 'width'), J_bc=g('J_slot', 'bc'), J_dbic=g('J_slot', 'dbic'),
                          lin_step=g('lin_slot', 'step'), lin_bc=g('lin_slot', 'bc'), end_step=g('J_end', 'step'), end_bc=g('J_end', 'bc'),
                          thr_shift=bn('thr_shift'), X_drop=bn('X_drop'))
    # the bottleneck, controlled: the held word's end-of-sentence readout minus its readout when it is not held,
    # weak against strong second concept (needs ignition.npz)
    npz = f'{root}/{m}/ignition/ignition.npz'
    if os.path.exists(npz):
        z = np.load(npz);BX = z['bn_X'].astype(np.float32);ev = {}
        for name, lo, hi in BANDS:
            ls = [l for l in range(n) if lo <= l/n < hi and l < BX.shape[-1]-1];chg = []
            for l in ls:
                S = (np.nanmean(BX[1, ..., l], 3)-np.nanmean(BX[0, ..., l], 3)).reshape(-1, BX.shape[3])
                base = np.abs(S[:, :3].mean(1)).mean();chg.append(np.nanmean(S[:, -3:].mean(1)-S[:, :3].mean(1))/max(base, 1e-6))
            rows[name]['evict_rel'] = float(np.nanmedian(chg)) if chg else np.nan
    out.setdefault(m, {})['ignition'] = rows
    print(f'\n  {m}')
    print(f"  {'band':10s} J step width   BC  dBIC | lin step  BC | end step  BC | held-X threshold shift | held-X specific readout change")
    for name, v in rows.items():
        print(f"  {name:10s} {f(v['J_step'])} {f(v['J_width'])} {f(v['J_bc'])} {f(v['J_dbic'], '.0f'):>5} | {f(v['lin_step'])} {f(v['lin_bc'])} |"
              f" {f(v['end_step'])} {f(v['end_bc'])} | {f(v['thr_shift'], '+.3f')} | {f(v.get('evict_rel'), '+.0%')}")

print('\nTRACE CONDITIONING: accuracy among the 6 targets; in brackets the accuracy drop as a share of the clean margin over chance')
for m in models:
    r = load(m, 'trace')
    if not r:continue
    out.setdefault(m, {})['trace'] = r
    print(f"\n  {m} (band {r['band'][0]}..{r['band'][-1]}, k {r['k']}, n {r['n']}) " + '  '.join(f"{c} agree {r['local_global'][c]['agree']:.3f}" for c in ('jspace', 'rand', 'matched')))
    for g in ('0', '1', '2', '4', '8'):
        t = r['trace'][g];rd = lambda c:'' if t[c].get('rel_acc_drop') is None else f" ({t[c]['rel_acc_drop']:+.0%})"
        print(f"  gap {g}: " + '  '.join(f"{c} {t[c]['acc_cand']:.2f}{rd(c) if c != 'clean' else ''}" for c in ('clean', 'jspace', 'rand', 'matched')))
    for c in ('jspace', 'rand', 'matched'):
        it = r['trace'][f'interaction_{c}'];print(f"  log-p drop at gaps 4-8 minus gap 0, {c}: {it['diff']:+.2f} ± {it['se']:.2f} (z {it['z']:+.1f})")
    lg = r['local_global']
    print(f"  local-global (nats): clean local {lg['clean']['local']:+.2f} global {lg['clean']['glob']:+.2f}; drops (local, global, global minus local, t):")
    for c in ('jspace', 'rand', 'matched'):
        d = lg[c];print(f"    {c:8s} {d['local_drop']['diff']:+.2f}  {d['glob_drop']['diff']:+.2f}  {d['glob_minus_local_drop']['diff']:+.2f} (t {d['glob_minus_local_drop']['t']:+.1f})")

print('\nCRITICALITY: J-space autocorrelation across sampling temperature')
for m in models:
    r = load(m, 'temperature')
    if not r:continue
    out.setdefault(m, {})['temperature'] = {k:{kk:vv for kk, vv in v.items() if kk not in ('C', 'C_within', 'lags', 'sample')} for k, v in r['temps'].items()}
    print(f"\n  {m}  T   rep  dist2   nll | C(16) C(128) | R2 pow  exp  expo | beta | words entropy occ")
    for T, v in r['temps'].items():
        C = lambda k:float(np.interp(k, v['lags'], v['C']));fi = v['fit']
        print(f"  {T:>12s} {v['repetition']:.2f} {v['distinct2']:.2f} {v['nll']:5.2f} | {C(16):5.2f} {C(128):5.2f} | {f(fi['r2_pow'])} {f(fi['r2_exp'])} {f(fi['exponent'])} |"
              f" {v['beta']:.2f} | {v['words']:5.0f} {v['entropy']:.2f} {v['occupancy']:.3f}")

print('\nSELF-TALK: convergence of conversations in J-space, theme words, the workspace of the last six turns')
for m in models:
    r = load(m, 'selftalk')
    if not r:continue
    c = r['convergence'];th = r['themes'];T = len(c)
    out.setdefault(m, {})['selftalk'] = dict(convergence=c, themes=th, lead_bliss=r['lead_bliss'], lead_self=r['lead_self'],
                                              top_first=r['top_first'][:15], top_last=r['top_last'][:15])
    q = lambda xs, a, b:float(np.mean(xs[a:b]))
    print(f"\n  {m}: between-opener similarity turns 1-5 {q([x['between'] for x in c], 0, 5):+.3f} -> last 5 {q([x['between'] for x in c], T-5, T):+.3f};"
          f" within {q([x['within'] for x in c], 0, 5):+.3f} -> {q([x['within'] for x in c], T-5, T):+.3f}")
    print(f"  bliss words: J {q(th['J_bliss'], 0, 5):.3f} -> {q(th['J_bliss'], T-5, T):.3f}, text {q(th['text_bliss'], 0, 5):.3f} -> {q(th['text_bliss'], T-5, T):.3f};"
          f" self words: J {q(th['J_self'], 0, 5):.3f} -> {q(th['J_self'], T-5, T):.3f}, text {q(th['text_self'], 0, 5):.3f} -> {q(th['text_self'], T-5, T):.3f}")
    if 'echo_onset' in r:
        on = {o:[x for x in v] for o, v in r['echo_onset'].items()}
        print('  permanent verbatim echo, onset turn per conversation:', {o:sorted(v, key=lambda x:(x is None, x)) for o, v in on.items()})
        out[m]['selftalk']['echo_onset'] = on
    print('  first turns, J-space:', [w for w, _ in r['top_first'][:12]])
    print('  last turns, J-space: ', [w for w, _ in r['top_last'][:12]])
print('\nSELF-REFERENCE: P(claims experience) by the model as judge; z-scored J-lens scores of word sets over the answer, workspace band')
for m in models:
    r = load(m, 'selfref')
    if not r:continue
    out.setdefault(m, {})['selfref'] = dict(summary=r['summary'], corr_self_ref=r['corr_self_ref'], by_claim=r['by_claim'])
    sets = list(r['sets'])
    print(f"\n  {m:11s} claim | " + ' '.join(f'{x:>13s}' for x in sets))
    for cond, v in r['summary'].items():
        print(f"  {cond:11s} {v['claim']['mean']:.2f}  | " + ' '.join(f"{v['J_'+x]['mean']:+.3f}±{v['J_'+x]['se']:.3f}".rjust(13) for x in sets))
    print('  within self_ref, corr(claim, set):', {k:(round(v, 2) if v is not None else None) for k, v in r['corr_self_ref'].items()})
json.dump(out, open(f'{root}/summary.json', 'w', encoding='utf-8'), indent=1, default=float)
