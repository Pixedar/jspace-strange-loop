"""Figure for SELF_FORECAST_GATES.md: does the loop's compact state carry private information (own state vs a
content-matched other rollout's state), and the positive control with a planted hidden state."""
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
            rows.append((f'{"no memory" if loop == "memoryless" else "with memory"}\n{"residual" if cand == "R" else "J-space"} reader\n{"readiness to revise" if obs == "dc_contra" else "answer log-odds"}', a[dk], a[ck], b[dk], b[ck]))
fig, ax = plt.subplots(figsize=(11, 3.8));x = np.arange(len(rows))
for j, (lab, d2, c2, d3, c3) in enumerate(rows):
    ax.errorbar(j-.12, d2, yerr=[[d2-c2[0]], [c2[1]-d2]], fmt='o', color='C0', capsize=3, label='own state vs another rollout' + chr(39) + 's state (same content)' if j == 0 else None)
    ax.errorbar(j+.12, d3, yerr=[[d3-c3[0]], [c3[1]-d3]], fmt='s', color='C3', capsize=3, label='positive control: a hidden variable planted in the model' if j == 0 else None)
ax.axhline(0, color='k', lw=.6);ax.axhline(.05, color='k', ls=':', lw=.6);ax.set_xticks(x);ax.set_xticklabels([r[0] for r in rows], fontsize=7)
ax.set_ylabel('gain in explained variance\n(R², 95 % interval)');ax.legend(fontsize=8);ax.set_title('Does the loop\'s compact state predict its next change better than the prompt and the writes alone? (Qwen3-8B)')
fig.tight_layout();fig.savefig(O/'private_state_test_8b.png', dpi=130);print('ok')
