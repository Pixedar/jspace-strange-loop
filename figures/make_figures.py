"""Figures for REPORT.md that are drawn from the published numbers or from the result files in this repository.
    python figures/make_figures.py           (run from the repository root)"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R = pathlib.Path(__file__).resolve().parents[1];F = R/'figures'
INK, ACC, ACC2, GREY = '#1d2733', '#c0392b', '#2e86c1', '#8a96a3'


def box(ax, x, y, w, h, text, fc='#f4f6f8', ec=INK, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.03', fc=fc, ec=ec, lw=1.1))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, color=INK, wrap=True)


def arrow(ax, a, b, color=INK, rad=0.0, lw=1.3, ls='-'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=11, color=color, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', clip_on=False))


def architectures():
    fig, axs = plt.subplots(1, 4, figsize=(18, 5.2))
    for ax in axs:ax.set_xlim(0, 1);ax.set_ylim(0, 1);ax.axis('off')
    a = axs[0];a.set_title('Parts 1–2: reflective loop', fontsize=11, color=INK)
    box(a, .04, .74, .74, .14, 'sampled thought (40 tokens)\nfrom a think-aloud frame')
    box(a, .04, .45, .74, .14, 'every 10 thoughts: reflection\n→ J-space words of recent states')
    box(a, .04, .16, .74, .14, 'self-model added as a nudge\nto the residual (layer 16)', fc='#fbeaea')
    arrow(a, (.41, .74), (.41, .59));arrow(a, (.41, .45), (.41, .30));arrow(a, (.78, .23), (.78, .81), color=ACC, rad=.45)
    a.text(.5, .04, 'own ≈ foreign self; best-fitting self is kept', ha='center', fontsize=8.5, color=GREY)
    a = axs[1];a.set_title('Part 3: the ring', fontsize=11, color=INK)
    box(a, .04, .74, .74, .14, 'placeholder token (no text)\nJ-space content at 70 % depth')
    box(a, .04, .45, .74, .14, 'online linear self-model\npredicts the next J state')
    box(a, .04, .16, .74, .14, 'its current surprise written\nback at 50 % depth', fc='#fbeaea')
    arrow(a, (.41, .74), (.41, .59));arrow(a, (.41, .45), (.41, .30));arrow(a, (.78, .23), (.78, .81), color=ACC, rad=.45)
    a.text(.5, .04, 'surprise = learned exit from attractors; not self-specific', ha='center', fontsize=8.5, color=GREY)
    a = axs[2];a.set_title('Decision-position J loop', fontsize=11, color=INK)
    box(a, .04, .74, .74, .14, 'fixed prompt, last position\nresidual at L29')
    box(a, .04, .45, .74, .14, 'J code: 25 words\n(greedy non-negative pursuit)')
    box(a, .04, .16, .74, .14, 're-synthesised at L22,\nfixed norm (gain g)', fc='#fbeaea')
    arrow(a, (.41, .74), (.41, .59));arrow(a, (.41, .45), (.41, .30));arrow(a, (.78, .23), (.78, .81), color=ACC, rad=.45)
    a.text(.5, .04, 'keeps what reproduces itself (stance), drops entities', ha='center', fontsize=8.5, color=GREY)
    a = axs[3];a.set_title('Phase B: trained self-describing loop', fontsize=11, color=INK)
    for i, x in enumerate((.04, .37, .70)):
        box(a, x, .62, .26, .2, f'segment {["k−1", "k", "k+1"][i]}\n64 tokens of\nown thought', fs=8)
    box(a, .37, .28, .26, .16, 'self-model S\n' + r'$z_{k-1} \rightarrow \hat{z}_k$', fc='#eaf2fb', fs=8)
    a.text(.17, .53, r'$z_{k-1}$', ha='center', fontsize=9, color=INK)
    arrow(a, (.30, .70), (.37, .70), color=ACC);a.text(.335, .74, 'write z', ha='center', fontsize=7.5, color=ACC)
    arrow(a, (.20, .62), (.40, .44), color=ACC2);arrow(a, (.50, .44), (.50, .62), color=ACC2, ls='--')
    a.text(.62, .50, 'forecast ' + r'$\hat{z}_k$' + ' written\ninto the channel', fontsize=7.5, color=ACC2)
    arrow(a, (.63, .70), (.70, .70), color=ACC)
    a.text(.5, .14, 'λ: pressure on the system to be\npredictable to its own self-model', ha='center', fontsize=8, color=INK)
    a.text(.5, .04, 'contrastive λ → phase clock; fed-back forecast is enacted', ha='center', fontsize=8.5, color=GREY)
    fig.tight_layout();fig.savefig(F/'architectures.png', dpi=140);plt.close(fig)


def word_clock():
    f = R/'phaseB_trained_self_model_loop'/'results'/'loopb14'/'reader_words.json'
    if not f.exists():return
    cw = json.load(open(f, encoding='utf-8')).get('clock_words', {})
    tags = [t for t in ('const_learned_l0', 'fcst_learned_l1_c') if t in cw]
    if not tags:return
    fig = plt.figure(figsize=(7*len(tags), 7))
    for i, t in enumerate(tags):
        ax = fig.add_subplot(1, len(tags), i+1, projection='polar');rows = cw[t]
        th = np.radians([r['angle'] for r in rows]);ax.plot(np.append(th, th[0] if False else []), [1]*len(th), 'o-', color=ACC)
        for j, r in enumerate(rows):
            ax.text(th[j], 1.25, f"{r['segment']}: " + ', '.join(r['words'][:3]), ha='center', va='center', fontsize=8.5, color=INK)
        ax.set_ylim(0, 1.5);ax.set_yticks([]);ax.set_xticks([])
        ax.set_title({'const_learned_l0':'14B plain loop', 'fcst_learned_l1_c':'14B forecast fed back, contrastive pressure'}[t], fontsize=11)
    fig.suptitle('The phase of the self-state, read in the model\'s own words, at each segment of a thought', fontsize=12)
    fig.tight_layout();fig.savefig(F/'phaseB_word_clock_14b.png', dpi=140);plt.close(fig)


def earlier_parts():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    sizes = ['1.7B', '4B', '8B', '14B', '32B'];ind = [0.01, 0.06, 0.14, 0.23, 0.25]
    ax[0].plot(sizes, ind, 'o-', color=ACC);ax[0].set_ylabel('individuality of the loop\'s self\n(closer to own earlier self than to others)')
    ax[0].set_title('Part 2: self-related effects grow with size', fontsize=10)
    labels = ['momentum', "another mind's\nsurprise", 'own surprise,\nwrong time', 'random\ndirection', 'freezing the\nself-model']
    z = [26.9, 15.9, 28.0, 52.9, -33.0]
    ax[1].bar(labels, z, color=[ACC if v > 0 else ACC2 for v in z]);ax[1].axhline(0, color=INK, lw=.8)
    ax[1].set_ylabel('Stouffer z (4 models, 24 seeds each)');ax[1].set_title('Part 3: current surprise vs controls (pre-registered)', fontsize=10)
    fig.tight_layout();fig.savefig(F/'parts2_3_summary.png', dpi=140);plt.close(fig)


if __name__ == '__main__':
    architectures();word_clock();earlier_parts();print('figures written to', F)
