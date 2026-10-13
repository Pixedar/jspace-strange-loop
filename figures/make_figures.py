"""Figures drawn from files in this repository.
    python figures/make_figures.py      (from the repository root)"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R = pathlib.Path(__file__).resolve().parents[1];F = R/'figures'
INK, ACC, ACC2, GREY = '#1d2733', '#c0392b', '#2e86c1', '#7d8a97'


def box(ax, x, y, w, h, text, fc='#f4f6f8', fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.03', fc=fc, ec=INK, lw=1.1))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, color=INK)


def arrow(ax, a, b, color=INK, rad=0.0, ls='-'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=12, color=color, lw=1.4, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', clip_on=False))


def architecture():
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw=dict(width_ratios=[1, 1.35]))
    for a in axs:a.set_xlim(0, 1);a.set_ylim(0, 1);a.axis('off')
    a = axs[0];a.set_title('1. The frozen J-space loop', fontsize=12, color=INK, loc='left')
    box(a, .03, .72, .62, .15, 'the model reads a fixed prompt\n(state at layer 29, last position)')
    box(a, .03, .43, .62, .15, 'that state, described as a mix of\n25 words (J-space reading)')
    box(a, .03, .14, .62, .15, 'the 25-word description is written\nback into layer 22, fixed strength', fc='#fbeaea')
    arrow(a, (.34, .72), (.34, .58));arrow(a, (.34, .43), (.34, .29));arrow(a, (.65, .21), (.65, .80), color=ACC, rad=.55)
    a.text(.86, .5, 'repeat\n(the model\nre-reads\nits own\ndescription)', ha='center', va='center', fontsize=8.5, color=ACC)
    a.text(.03, .03, 'nothing is trained: what does the loop converge to, and why?', fontsize=9, color=GREY)
    a = axs[1];a.set_title('2. The trained self-describing loop', fontsize=12, color=INK, loc='left')
    for i, x in enumerate((.02, .36, .70)):
        box(a, x, .64, .27, .2, ['chunk k−1', 'chunk k', 'chunk k+1'][i] + '\n64 tokens of the model\'s\nown thinking', fs=8.5)
    a.text(.335, .885, '32 numbers', ha='center', fontsize=8.5, color=ACC)
    arrow(a, (.29, .74), (.36, .74), color=ACC);arrow(a, (.63, .74), (.70, .74), color=ACC)
    box(a, .36, .26, .27, .17, 'self-model\npredicts the next\n32 numbers', fc='#eaf2fb', fs=8.5)
    arrow(a, (.17, .64), (.38, .43), color=ACC2);arrow(a, (.5, .43), (.5, .64), color=ACC2, ls='--')
    a.text(.66, .5, 'the prediction can be\nwritten back into the\nnext chunk (the strange\nloop) or not (control)', fontsize=8.5, color=ACC2)
    a.text(.02, .14, 'Chunks never see each other\'s text. The only memory is the 32-number state, read from the model\n'
                     'at the end of a chunk and written into the next one. The language model is frozen; small readers,\n'
                     'writers and the self-model are trained. A knob (pressure) pushes the system to be predictable\nto its own self-model.',
           fontsize=8.5, color=INK)
    fig.tight_layout();fig.savefig(F/'how_it_works.png', dpi=140);plt.close(fig)


# plain-language reading of the J-lens words at each chunk (interpretation; the words are printed under each label)
STAGES = {
    '14B/const_learned_l0':['recalling, informal', 'tentative', 'hedging: maybe / might', 'hedging: perhaps',
                            'adding alternatives', 'checking ("ensure")', 'wrapping up', 'summarising'],
    '14B/fcst_learned_l1_c':['recalling general facts', 'loose exploration', 'uncertain search', 'uncertain search',
                             'asserting well-known facts', 'moving to the answer', 'moving to the answer', 'concluding'],
    '8B/const_learned_l0':['vague framing', 'tentative', 'hedging', 'adding options', 'elaborating',
                           'concluding', 'concluding', 'summarising'],
    '8B/fcst_learned_l1_c':['topic: coffee', 'topic: coffee', 'topic: films, games', 'topic: music', 'topic: music',
                            'topic: coffee', 'topic: sports', 'topic: films, animals'],
}
TITLES = {'14B/const_learned_l0':'14B, plain loop', '14B/fcst_learned_l1_c':'14B, own forecast fed back, strong pressure',
          '8B/const_learned_l0':'8B, plain loop', '8B/fcst_learned_l1_c':'8B, own forecast fed back, strong pressure'}


def word_clock():
    pos = json.load(open(R/'trained_loop'/'results'/'clock_plane_positions.json', encoding='utf-8'))
    keys = [k for k in TITLES if k in pos]
    fig, axs = plt.subplots(2, 2, figsize=(15, 12.5));axs = axs.ravel()
    for ax, k in zip(axs, keys):
        P = np.array([[r['x'], r['y']] for r in pos[k]]);P = (P-P.mean(0))/np.abs(P-P.mean(0)).max()
        P = P*np.array([1.0, 1.1])+np.array([-1.05, 0.0])                   # the path, left part of the panel
        for i in range(7):arrow(ax, tuple(P[i]), tuple(P[i+1]), color=ACC)
        ys = np.linspace(1.25, -1.25, 8)
        for i, r in enumerate(pos[k]):
            ax.plot(*P[i], 'o', color=ACC, ms=7, zorder=3);ax.text(P[i, 0], P[i, 1]+.09, str(i), ha='center', fontsize=9, color=ACC, weight='bold')
            ax.plot([P[i, 0], 0.55], [P[i, 1], ys[i]], color='#d5dbe1', lw=.8, zorder=1)
            ax.text(0.6, ys[i]+.04, f'{i}  {STAGES[k][i]}', ha='left', va='center', fontsize=10.5, color=INK, weight='bold')
            ax.text(0.6, ys[i]-.11, ', '.join(r['words'][:3]), ha='left', va='center', fontsize=8, color=GREY)
        ax.set_xlim(-2.25, 2.6);ax.set_ylim(-1.45, 1.45);ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(TITLES[k], fontsize=12, color=INK)
        for sp in ax.spines.values():sp.set_color('#d5dbe1')
    fig.suptitle('Where the loop\'s state sits at each chunk of a thought (0–7), read in the model\'s own words\n'
                 'bold: my reading of the J-lens words; grey: the top words. Plain loops and 14B under pressure move through the stages of '
                 'reasoning;\n8B under the strongest pressure cycles through topics unrelated to the thought.', fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, .94));fig.savefig(F/'word_clock.png', dpi=130);plt.close(fig)


if __name__ == '__main__':
    architecture();word_clock();print('figures written to', F)
