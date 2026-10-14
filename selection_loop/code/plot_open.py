"""Figure for the open-access experiment (focus_open.py): fig_open.png — what the state holds, focus edits, and the frozen
model's own attention on A's and B's records per layer (trained state vs after the focus edit).
    python plot_open.py --runs <eval_open_*.json> [...] --labels <label> [...] --held <held.json> --out figures"""
import argparse, json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EDITS = [('none', 'no edit'), ('focus_A_to_B', 'focus A→B\nbefore reading'), ('focus_A_to_B_after_read', 'focus A→B\nafter reading'),
         ('focus_A_to_B_x0.5', 'half\nstrength'), ('random_same_size', 'random,\nsame size'), ('toward_absent_D', 'toward an\nabsent animal')]


def main():
    p = argparse.ArgumentParser();p.add_argument('--runs', nargs='+');p.add_argument('--labels', nargs='+');p.add_argument('--held', default=None)
    p.add_argument('--out', default='figures');a = p.parse_args();out = pathlib.Path(a.out);out.mkdir(parents=True, exist_ok=True)
    R = [json.load(open(f, encoding='utf-8')) for f in a.runs];Hd = json.load(open(a.held, encoding='utf-8')) if a.held else {}
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    x = np.arange(len(R));w = 0.38
    nz = lambda v:np.nan if v is None else v
    tr = [nz(Hd.get(r['tag']+'|'+r['cfg']['model'], {}).get('nonlinear', {}).get('tracked_new_colour')) for r in R]
    ot = [nz(Hd.get(r['tag']+'|'+r['cfg']['model'], {}).get('nonlinear', {}).get('other_new_colours')) for r in R]
    ax[0].bar(x-w/2, tr, w, label="tracked animal's new colour");ax[0].bar(x+w/2, ot, w, label="other animals' new colours (same chunk)")
    ax[0].axhline(0.1, color='gray', ls=':', lw=1);ax[0].set_xticks(x);ax[0].set_xticklabels(a.labels, fontsize=8);ax[0].set_ylim(0, 1.05)
    ax[0].set_title('what the state holds after a chunk\n(decodability, whole streams held out; dotted: chance)', fontsize=10);ax[0].legend(fontsize=7, loc='upper right')
    xe = np.arange(len(EDITS));we = 0.8/len(R)
    for i, (r, lab) in enumerate(zip(R, a.labels)):
        ax[1].bar(xe+(i-(len(R)-1)/2)*we, [r['focus_edits'][c]['answers_B_new'] for c, _ in EDITS], we, label=lab)
    ax[1].set_xticks(xe);ax[1].set_xticklabels([n for _, n in EDITS], fontsize=7);ax[1].set_ylim(0, 1.05)
    ax[1].set_title("focus edits on twin streams: answers B's new colour", fontsize=10);ax[1].legend(fontsize=7)
    for i, (r, lab) in enumerate(zip(R, a.labels)):
        At = r.get('attention_chunk3')
        if not At:continue
        L = sorted(At['own']['per_layer'], key=int);ls = [int(l) for l in L]
        ax[2].plot(ls, [At['own']['per_layer'][l]['A']-At['own']['per_layer'][l]['B'] for l in L], '-o', ms=3, color=f'C{i}', label=f'{lab}: trained state')
        ax[2].plot(ls, [At['focus_A_to_B']['per_layer'][l]['A']-At['focus_A_to_B']['per_layer'][l]['B'] for l in L], '--o', ms=3, color=f'C{i}', label=f'{lab}: after edit A→B')
    ax[2].axhline(0, color='gray', lw=0.8);ax[2].set_xlabel('layer');ax[2].legend(fontsize=7)
    ax[2].set_title("frozen model's own attention from the chunk's last token:\non A's record minus on B's record", fontsize=10)
    fig.tight_layout();fig.savefig(out/'fig_open.png', dpi=130);plt.close(fig)


if __name__ == '__main__':
    main()
