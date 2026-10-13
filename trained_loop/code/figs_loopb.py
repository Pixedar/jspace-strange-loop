"""Figures for the trained loop (loopb.py): the rotary time code under self-prediction pressure, the pressure dose-response of
the strange loop (enactment vs self-predictability vs content), and the loop's benefit.
    python figs_loopb.py <folder holding loopb/ and loopb14/>"""
import json, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = pathlib.Path(sys.argv[1]);F = R/'figs';F.mkdir(exist_ok=True)


def load(d, t):
    f = R/d/'eval'/f'{t}.json'
    if not f.exists():return None, None
    return json.load(open(f, encoding='utf-8')), dict(np.load(R/d/'eval'/f'{t}.npz'))


# 1. the time code: segment-mean states in their top-3 principal axes
arms = [('loopb', 'const_learned_l0', '8B plain loop'), ('loopb', 'fcst_learned_l1_c', '8B forecast fed back, pressure'),
        ('loopb', 'state_learned_l1_c', '8B state channel, pressure'), ('loopb', 'const_learned_l1_c', '8B no channel, pressure'),
        ('loopb14', 'fcst_learned_l1_c', '14B forecast fed back, pressure'), ('loopb14', 'state_learned_l1_c', '14B state channel, pressure')]
arms = [a for a in arms if (R/a[0]/'eval'/f'{a[1]}.npz').exists()]
fig = plt.figure(figsize=(4.2*len(arms), 4.4))
for i, (d, t, name) in enumerate(arms):
    Z = load(d, t)[1]['Z_own'];M = Z.mean(0);C = M-M.mean(0);Vt = np.linalg.svd(C, full_matrices=False)[2][:3]
    P = C@Vt.T;T_ = (Z[:24]-M.mean(0))@Vt.T
    ax = fig.add_subplot(1, len(arms), i+1, projection='3d')
    for tr in T_:ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color='0.75', lw=.5)
    ax.plot(P[:, 0], P[:, 1], P[:, 2], '-o', color='#c0392b', lw=2, ms=4)
    for k in range(8):ax.text(P[k, 0], P[k, 1], P[k, 2], str(k), fontsize=8)
    ax.set_title(name, fontsize=9);ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
fig.suptitle('Self-state over the 8 segments of a thought (red: mean over 400 traces, grey: single traces)', fontsize=10)
fig.tight_layout();fig.savefig(F/'time_code.png', dpi=130);print('saved', F/'time_code.png')

# 2. dose-response of the strange loop (forecast fed back) vs the matched state channel
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for chan, col in (('fcst', '#c0392b'), ('state', '#2e86c1')):
    xs, en, re, wo = [], [], [], []
    for lam, t in ((0, f'{chan}_learned_l0'), (0.1, f'{chan}_learned_l0.1_c'), (1, f'{chan}_learned_l1_c')):
        r, _ = load('loopb', t)
        if r is None:continue
        xs.append(lam);en.append(r['swap_pull_forecast'][0]);wo.append(r['necessity_foreign'][0])
        if 'retrieval' in r:re.append(r['retrieval']['S'])
        else:                                                  # older evaluations: recompute from the saved states
            Zn = load('loopb', t)[1];Zo, Zh = Zn['Z_own'], Zn['ZH_own'];acc = []
            for k in range(1, 8):
                A = Zh[:, k]/np.linalg.norm(Zh[:, k], axis=1, keepdims=True);Bm = Zo[:, k]/np.linalg.norm(Zo[:, k], axis=1, keepdims=True)
                acc.append(((A@Bm.T).argmax(1) == np.arange(len(Zo))).mean())
            re.append(float(np.mean(acc)))
    lab = 'forecast fed back' if chan == 'fcst' else 'state channel (control)'
    ax[0].plot(xs, en, '-o', color=col, label=lab);ax[1].plot(xs, re, '-o', color=col, label=lab);ax[2].plot(xs, wo, '-o', color=col, label=lab)
for a_, t_ in zip(ax, ['enactment: next state pulled toward a swapped-in forecast', 'self-model picks the true next state (out of 400)',
                       'what the state carries: own vs another trace' + chr(39) + 's state (nats/token)']):
    a_.set_xscale('symlog', linthresh=0.1);a_.set_xlabel('pressure to be predictable to the self-model (λ)');a_.set_title(t_, fontsize=9);a_.legend(fontsize=8)
fig.tight_layout();fig.savefig(F/'pressure_dose_8b.png', dpi=130);print('saved', F/'pressure_dose_8b.png')
