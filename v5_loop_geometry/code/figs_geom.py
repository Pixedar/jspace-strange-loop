"""Figures for GEOMETRY_RESULTS.md from the downloaded result files (local, CPU).
    python figs_geom.py   (reads research_v5/results/{geom,spec,axis,transport}, writes research_v5/results/figs/)"""
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
        ax[0].plot(np.arange(len(m)), m, ls, color=c, label=f'{"J loop" if mode == "J" else "random writes"} g={g}')
        if mode == 'J':ax[0].fill_between(np.arange(len(m)), m-s, m+s, color=c, alpha=.12)
ax[0].axhline(0, color='k', lw=.5);ax[0].set_xscale('symlog', linthresh=1);ax[0].set_xlim(0, None);ax[0].set_xlabel('loop step');ax[0].set_ylabel('shift along persona PC1 (role SD)')
ax[0].set_title('Qwen3-8B: the J loop drifts toward the casual end of persona space');ax[0].legend(fontsize=7)
pr = a['proj_roles'];roles_low = js['persona']['roles_low'];roles_high = js['persona']['roles_high']
ax[1].hist(pr/pr.std(), bins=20, color='0.7');ax[1].axvline(js['persona']['default_proj_in_role_sd'], color='k', label='default assistant')
for g, c in ((1.0, 'C1'), (4.0, 'C3')):
    P = a[f'axisproj_J_{g}'];ax[1].axvline(P[-1].mean(), color=c, ls='-', label=f'J loop g={g}, t=64 (mean)')
ax[1].axvline(a['axisproj_J_4.0'][0].mean(), color='C0', ls=':', label='natural states (mean)')
ax[1].set_title(f'persona PC1: low = {", ".join(roles_low[:3])}\nhigh = {", ".join(roles_high[:3])}', fontsize=8)
ax[1].set_xlabel('projection (role SD)');ax[1].legend(fontsize=7)
fig.tight_layout();fig.savefig(O/'persona_axis_8b.png', dpi=130);plt.close(fig)

# 2. transport operator spectrum and words
t = json.loads((R/'transport'/'transport.json').read_text())
fig, ax = plt.subplots(figsize=(11, 3.6))
ev = [e['eig'][0] for e in t['leading_eigvecs']];ax.bar(range(len(ev)), ev, color='C2')
for i, e in enumerate(t['leading_eigvecs']):
    ax.text(i, ev[i]*1.02+5, '\n'.join(e['words'][:6]), ha='center', va='bottom', fontsize=7)
ax.set_xticks(range(len(ev)));ax.set_xticklabels([f'mode {i+1}' for i in range(len(ev))]);ax.set_ylabel('eigenvalue')
ax.set_ylim(0, max(ev)*1.6);ax.set_title('Qwen3-8B write->read word transport (3 000 words, fp32): leading modes')
fig.tight_layout();fig.savefig(O/'transport_modes_8b.png', dpi=130);plt.close(fig)

# 3. bf16 wobble vs fp32 and spectral radius vs gain
b = json.loads((R/'spec'/'settle_bf16.json').read_text())['res'];f = json.loads((R/'spec'/'settle_fp32.json').read_text())['res']
jj = json.loads((R/'spec'/'jac32.json').read_text());gs = ['0.25', '1.0', '4.0']
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
x = np.arange(len(gs));ax[0].bar(x-.18, [b[g]['late_maxstep_median'] for g in gs], .36, label='bf16 (v3 setting)')
ax[0].bar(x+.18, [f[g]['late_maxstep_median'] for g in gs], .36, label='fp32');ax[0].set_yscale('log');ax[0].set_xticks(x)
ax[0].set_xticklabels([f'g={g}' for g in gs]);ax[0].set_ylabel('late step of z (median)');ax[0].legend(fontsize=7)
ax[0].set_title('The "fixed points" wobble only in bf16')
ax[1].errorbar([float(g) for g in gs], [jj[g]['spectral_radius_median'] for g in gs],
               yerr=[[jj[g]['spectral_radius_median']-0 for g in gs], [jj[g]['spectral_radius_q90']-jj[g]['spectral_radius_median'] for g in gs]],
               fmt='o-', capsize=3, lolims=False)
ax[1].axhline(1, color='k', ls=':', lw=.8);ax[1].set_xscale('log');ax[1].set_xlabel('gain');ax[1].set_ylabel('spectral radius (median, q90)')
ax[1].set_ylim(0, 1.1);ax[1].set_title('Local loop map at its fixed points (fp32)')
fig.tight_layout();fig.savefig(O/'loop_spectrum_8b.png', dpi=130);plt.close(fig)
print('figures in', O)
