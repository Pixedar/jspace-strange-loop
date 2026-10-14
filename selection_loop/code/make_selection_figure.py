"""Architecture figure for the selection experiments: figures/selection_loop.png.
(a) restricted access: the state chooses which record the read takes in; (b) open access: the state is written into the
frozen model's own reading of the whole chunk.
    python make_selection_figure.py --out figures"""
import argparse, pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C_STATE, C_MODEL, C_TEXT, C_READ, C_OUT, C_SELF = '#4C72B0', '#8C8C8C', '#DD8452', '#55A868', '#C44E52', '#937860'


def box(ax, x, y, w, h, text, color, fs=8.5):
    for fc, al in ((color, 0.16), ('none', 1.0)):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=fc, ec=color, alpha=al, lw=1.2))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs)


def arrow(ax, x0, y0, x1, y1, color='#333333'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color=color, lw=1.3))


def loop_back(ax, x0, y0, x1, y1, y_low, color, text):
    """Down from (x0, y0) to y_low, across, and up into (x1, y1): the state carried to the next chunk."""
    ax.plot([x0, x0, x1], [y0, y_low, y_low], color=color, lw=1.3)
    ax.annotate('', xy=(x1, y1), xytext=(x1, y_low), arrowprops=dict(arrowstyle='-|>', color=color, lw=1.3))
    ax.text((x0+x1)/2, y_low-0.28, text, ha='center', va='center', fontsize=8, color=color)


def main():
    p = argparse.ArgumentParser();p.add_argument('--out', default='figures');a = p.parse_args()
    out = pathlib.Path(a.out);out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    for ax in axes:ax.set_xlim(0, 10);ax.set_ylim(0, 7);ax.axis('off')
    # (a) restricted access
    ax = axes[0];ax.set_title('(a) restricted access: the state chooses what the model takes in', fontsize=11, loc='left')
    ax.text(1.5, 6.35, 'one chunk: four records,\neach read by the frozen model on its own', ha='center', fontsize=8)
    for i, s in enumerate(['"The cat is now red."', '"The fox is now blue."', '"The owl turned gold."', '"Now the pig is gray."']):
        box(ax, 0.2, 5.6-i*0.62, 2.6, 0.5, s, C_TEXT, fs=8)
    box(ax, 0.2, 2.2, 2.6, 0.9, 'state (32 numbers)\nfrom the previous chunk', C_STATE)
    box(ax, 3.4, 4.3, 2.3, 1.3, 'read: attention over the\nfour records + "nothing new"\n(query from the state)', C_READ)
    arrow(ax, 2.85, 4.9, 3.35, 4.95)
    arrow(ax, 2.85, 2.9, 3.6, 4.25, color=C_STATE);ax.text(2.95, 3.55, 'query', fontsize=7.5, color=C_STATE)
    box(ax, 3.4, 1.9, 2.3, 1.5, 'frozen language model\nreads the state, the read\ncolour, the instruction\nand the question', C_MODEL)
    arrow(ax, 4.55, 4.25, 4.55, 3.45);ax.text(4.65, 3.85, 'only the read colour', fontsize=7.5)
    arrow(ax, 2.85, 2.5, 3.35, 2.5, color=C_STATE)
    box(ax, 6.4, 3.1, 2.4, 0.8, 'answer: "blue"\n(trained: the right colour)', C_OUT)
    box(ax, 6.4, 1.7, 2.4, 0.9, 'next state (32 numbers),\nread out of the model', C_STATE)
    arrow(ax, 5.75, 2.9, 6.35, 3.4);arrow(ax, 5.75, 2.3, 6.35, 2.15)
    box(ax, 6.4, 5.0, 2.4, 0.9, 'self-model: predicts\nthe next state', C_SELF, fs=8)
    arrow(ax, 8.95, 2.6, 8.95, 4.95, color=C_SELF)
    loop_back(ax, 7.6, 1.65, 1.5, 2.15, 1.0, C_STATE, 'carried to the next chunk (the loop)')
    ax.text(5.0, 0.15, 'Editing the state changes which record is read: figures/fig_focus_edits.png', ha='center', fontsize=8, style='italic')
    # (b) open access
    ax = axes[1];ax.set_title("(b) open access: the state steers the model's own reading", fontsize=11, loc='left')
    box(ax, 0.2, 4.2, 4.4, 1.6, 'one chunk as ordinary text, all records in view:\n"Notes: [state] The cat is now red. The fox is now\nblue. The owl turned gold. Now the pig is gray."', C_TEXT, fs=8.5)
    box(ax, 0.2, 2.2, 2.6, 0.9, 'state (32 numbers)\nfrom the previous chunk', C_STATE)
    arrow(ax, 1.5, 3.15, 1.5, 4.15, color=C_STATE)
    ax.text(1.65, 3.62, 'written into the placeholders\nbefore the model reads', fontsize=7.5, color=C_STATE, va='center')
    box(ax, 5.2, 4.2, 2.4, 1.6, 'frozen language model\nreads the whole chunk;\nits own attention decides\nwhat reaches the last word', C_MODEL)
    arrow(ax, 4.65, 5.0, 5.15, 5.0)
    box(ax, 5.2, 2.2, 2.4, 1.0, 'next state (32 numbers),\nread at the last word', C_STATE)
    arrow(ax, 6.4, 4.15, 6.4, 3.25)
    box(ax, 8.0, 2.2, 1.9, 1.0, 'question with the\nnew state: answer', C_OUT)
    arrow(ax, 7.65, 2.7, 7.95, 2.7)
    loop_back(ax, 6.4, 2.15, 1.5, 2.15, 1.2, C_STATE, 'carried to the next chunk (the loop)')
    ax.text(5.0, 0.15, "Nothing restricts access: any selection must happen inside the model's own attention", ha='center', fontsize=8, style='italic')
    fig.tight_layout();fig.savefig(out/'selection_loop.png', dpi=140);plt.close(fig)


if __name__ == '__main__':
    main()
