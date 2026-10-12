"""Figure for QUAL_RESULTS.md: G2 (self-state vs content-matched swap) and G3 (planted hidden state) contrasts."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = pathlib.Path(__file__).resolve().parents[1]/'results';O = R/'figs';O.mkdir(exist_ok=True)
q2, q3 = json.loads((R/'q'/'g2.json').read_text()), json.loads((R/'q'/'g3.json').read_text())
m2, m3 = json.loads((R/'qm'/'g2.json').read_text()), json.loads((R/'qm'/'g3.json').read_text())
rows = []
for loop, g2, g3, key2, key3, dk, ck in (('memoryless', q2, q3, None, None, 'delta', 'ci'), ('memory', m2, m3, None, None, 'delta_vs_swap', 'ci_vs_swap')):
    for cand in ('R', 'J'):
        for obs in ('dc_contra', 'dlo'):
            a = g2[cand][obs];b = (g3[cand]['contrast'][obs])
            rows.append((f'{loop}\nZ_{cand} {"Δc" if obs == "dc_contra" else "Δlo"}', a[dk], a[ck], b[dk], b[ck]))
fig, ax = plt.subplots(figsize=(11, 3.8));x = np.arange(len(rows))
for j, (lab, d2, c2, d3, c3) in enumerate(rows):
    ax.errorbar(j-.12, d2, yerr=[[d2-c2[0]], [c2[1]-d2]], fmt='o', color='C0', capsize=3, label='G2: own self-state vs swap' if j == 0 else None)
    ax.errorbar(j+.12, d3, yerr=[[d3-c3[0]], [c3[1]-d3]], fmt='s', color='C3', capsize=3, label='G3: with a planted hidden state' if j == 0 else None)
ax.axhline(0, color='k', lw=.6);ax.axhline(.05, color='k', ls=':', lw=.6);ax.set_xticks(x);ax.set_xticklabels([r[0] for r in rows], fontsize=7)
ax.set_ylabel('ΔR² (95 % CI)');ax.legend(fontsize=8);ax.set_title('Does the compact self-state know the system\'s own write-induced changes? (Qwen3-8B, F2, validation)')
fig.tight_layout();fig.savefig(O/'qual_g2_g3_8b.png', dpi=130);print('ok')
