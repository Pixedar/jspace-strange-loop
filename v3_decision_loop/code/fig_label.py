"""Figure for the H2 content-label control (amendment 17): F2 disposition-prediction gains of each representation over
the output-only baseline (primary analysis) and over output + evidence label (control), per layer x position, 8B and 14B.
    python fig_label.py results_dir"""
import sys, json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else 'results')
C = dict(J25='#d1495b', residual='#3b5b92', remainder='#66a182', pca25='#edae49', rp25='#9a9a9a')
NAMES = dict(J25='J (K=25)', residual='residual', remainder='remainder', pca25='PCA-25', rp25='random-25')
plt.rcParams.update({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False, 'figure.dpi':130})
fig, ax = plt.subplots(2, 2, figsize=(10.5, 6.2), sharey='row')
for col, (model, tag) in enumerate((('qwen3-8b', 'main'), ('qwen3-14b', 'rep'))):
    main = json.loads((R/f'e1_{model}_{tag}_results_e1.json').read_text())['H2']['F2']['by_representation']
    lab = json.loads((R/f'e1_{model}_{tag}_results_h2_label.json').read_text())['F2']['by_representation']
    cells = sorted(lab, key=lambda k:(int(k.split('_')[0]), k.split('_')[1] != 'evid'));x = np.arange(len(cells));w = .16
    for row, (src, key, title) in enumerate(((main, 'gain_eligible_mean', 'gain over output-only (pre-registered)'),
                                             (lab, 'gain_over_output_plus_label', 'gain over output + evidence label (control)'))):
        a = ax[row, col]
        for i, rep in enumerate(C):
            y = np.array([src[c][rep][key] for c in cells]);ci = np.array([src[c][rep]['ci'] for c in cells])
            a.bar(x+(i-2)*w, y, w, color=C[rep], label=NAMES[rep], yerr=np.abs(ci.T-y), ecolor='#444', capsize=1.2, lw=0)
        a.axhline(0, color='k', lw=.6);a.set_xticks(x)
        a.set_xticklabels([c.replace('_evid', '\nevidence').replace('_dec', '\ndecision') for c in cells], fontsize=7)
        a.set_title(f'{model.replace("qwen3-", "Qwen3-").upper().replace("QWEN3", "Qwen3")}: {title}', fontsize=9)
        if col == 0:a.set_ylabel('held-out within-scenario R$^2$ gain')
ax[0, 0].legend(frameon=False, fontsize=7, ncol=5, loc='upper left')
fig.tight_layout();out = R/'figs'/'e1_h2_label_control.png';fig.savefig(out);print('wrote', out)
