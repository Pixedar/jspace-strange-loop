"""Figures for GEOMETRY_RESULTS.md from the result files in ../results (local, CPU).
    python figs_geom.py   (reads results/{spec,axis,transport}, writes results/figs/)"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = pathlib.Path(__file__).resolve().parents[1]/'results';O = R/'figs';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9})

# 1. persona-axis projection over loop steps
a = np.load(R/'axis'/'axis.npz');js = json.loads((R/'axis'/'axis.json').read_text())
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for mode, ls in (('J', '-'), ('rand', '--')):
    for g, c in ((0.25, 'C0'), (1.0, 'C1'), (4.0, 'C3')):
        P = a[f'axisproj_{mode}_{g}'];d = P-P[0][None];m = d.mean(1);s = d.std(1)
        ax[0].plot(np.arange(len(m)), m, ls, color=c, label=f'{"loop" if mode == "J" else "random writes"}, strength {g}')
        if mode == 'J':ax[0].fill_between(np.arange(len(m)), m-s, m+s, color=c, alpha=.12)
ax[0].axhline(0, color='k', lw=.5);ax[0].set_xscale('symlog', linthresh=1);ax[0].set_xlim(0, None);ax[0].set_xlabel('loop step');ax[0].set_ylabel('shift along the casual-wise persona axis\n(unit: spread of the 40 roles)')
ax[0].set_title('Qwen3-8B: the loop moves every context toward the casual end');ax[0].legend(fontsize=7)
pr = a['proj_roles'];roles_low = js['persona']['roles_low'];roles_high = js['persona']['roles_high']
ax[1].hist(pr/pr.std(), bins=20, color='0.7', label='the 40 role prompts');ax[1].axvline(js['persona']['default_proj_in_role_sd'], color='k', label='default assistant')
for g, c in ((1.0, 'C1'), (4.0, 'C3')):
    P = a[f'axisproj_J_{g}'];ax[1].axvline(P[-1].mean(), color=c, ls='-', label=f'after 64 loop steps, strength {g} (mean)')
ax[1].axvline(a['axisproj_J_4.0'][0].mean(), color='C0', ls=':', label='before the loop (mean)')
ax[1].set_title(f'persona axis (first principal axis of 40 role prompts)\ncasual end: {", ".join(roles_low[:3])}; wise end: {", ".join(roles_high[:3])}', fontsize=8)
ax[1].set_xlabel('position on the axis (unit: spread of the 40 roles)');ax[1].legend(fontsize=7)
fig.tight_layout();fig.savefig(O/'persona_axis_8b.png', dpi=130);plt.close(fig)

# 2. transport operator spectrum and words
t = json.loads((R/'transport'/'transport.json').read_text())
fig, ax = plt.subplots(figsize=(11, 3.6))
ev = [e['eig'][0] for e in t['leading_eigvecs']];ax.bar(range(len(ev)), ev, color='C2')
for i, e in enumerate(t['leading_eigvecs']):
    ax.text(i, ev[i]*1.02+5, '\n'.join(e['words'][:6]), ha='center', va='bottom', fontsize=7)
ax.set_xticks(range(len(ev)));ax.set_xticklabels([f'pattern {i+1}' for i in range(len(ev))]);ax.set_ylabel('eigenvalue (strength)')
ax.set_ylim(0, max(ev)*1.6);ax.set_title('Qwen3-8B: word patterns that reproduce themselves most strongly through the model\n(leading eigenvectors of the write-to-read transport matrix over 3,000 words)')
fig.tight_layout();fig.savefig(O/'transport_modes_8b.png', dpi=130);plt.close(fig)

# 3. bf16 wobble vs fp32 and spectral radius vs gain
b = json.loads((R/'spec'/'settle_bf16.json').read_text())['res'];f = json.loads((R/'spec'/'settle_fp32.json').read_text())['res']
jj = json.loads((R/'spec'/'jac32.json').read_text());gs = ['0.25', '1.0', '4.0']
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
x = np.arange(len(gs));ax[0].bar(x-.18, [b[g]['late_maxstep_median'] for g in gs], .36, label='16-bit numbers (bfloat16)')
ax[0].bar(x+.18, [f[g]['late_maxstep_median'] for g in gs], .36, label='32-bit numbers (float32)');ax[0].set_yscale('log');ax[0].set_xticks(x)
ax[0].set_xticklabels([f'strength {g}' for g in gs]);ax[0].set_ylabel('step size late in the loop (median)');ax[0].legend(fontsize=7)
ax[0].set_title('The remaining motion of settled states is rounding')
ax[1].errorbar([float(g) for g in gs], [jj[g]['spectral_radius_median'] for g in gs],
               yerr=[[jj[g]['spectral_radius_median']-0 for g in gs], [jj[g]['spectral_radius_q90']-jj[g]['spectral_radius_median'] for g in gs]],
               fmt='o-', capsize=3, lolims=False)
ax[1].axhline(1, color='k', ls=':', lw=.8);ax[1].set_xscale('log');ax[1].set_xlabel('loop strength g');ax[1].set_ylabel('factor by which a small perturbation\nshrinks per step (median; bar: 90th pct.)')
ax[1].set_ylim(0, 1.1);ax[1].text(0.3, 1.02, 'above 1: unstable', fontsize=7);ax[1].set_title('Each settled state is locally stable (float32)')
fig.tight_layout();fig.savefig(O/'loop_spectrum_8b.png', dpi=130);plt.close(fig)
print('figures in', O)
