"""Figures for RESULTS.md.
  fig_focus_edits.png  the focus-code edit and its controls: answers B's colour / keeps A's stored colour / attention on
                       "nothing new", per condition, for each model and seed (edits_<tag>.json, chunks 2-4)
  fig_retry.png        the retry experiment: second read on the tracked record after a failed vs a delivered first
                       read (per arm), and the paired counterfactual with exchanged intermediate states
    python plot_v7.py --edits <edits json> [...] --labels <label> [...] --retry <folder with evalr_*.json> [...] --out <folder>"""
import argparse, json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COND = [('own', 'no edit'), ('focus_edit_x1', 'focus A→B'), ('focus_edit_query_only', 'query only'), ('focus_edit_answer_only', 'answer prompt only'),
        ('focus_edit_x0.5', '× 0.5'), ('focus_edit_x2', '× 2'), ('random_edit_same_size', 'random, same size'),
        ('focus_edit_toward_absent_D', 'toward absent D')]


def mean_over_chunks(r, c, k):
    v = [r['by_chunk'][t][c].get(k) for t in r['by_chunk'] if c in r['by_chunk'][t] and r['by_chunk'][t][c].get(k) is not None]
    return float(np.mean(v)) if v else np.nan


def focus_fig(files, labels, out):
    R = [json.load(open(f, encoding='utf-8')) for f in files]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    keys = [('answers_B', "answers B's new colour"), ('answers_A_stored', "keeps A's stored colour"), ('att_nothing_new', 'attention on "nothing new"')]
    x = np.arange(len(COND));w = 0.8/len(R)
    for ax, (k, title) in zip(axes, keys):
        for i, (r, lab) in enumerate(zip(R, labels)):
            ax.bar(x+(i-(len(R)-1)/2)*w, [mean_over_chunks(r, c, k) for c, _ in COND], w, label=lab)
        ax.set_xticks(x);ax.set_xticklabels([n for _, n in COND], rotation=40, ha='right', fontsize=8);ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.05);ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('fraction (edit at chunks 2-4, 256 twin streams each)');axes[1].legend(fontsize=8, loc='upper left')
    fig.suptitle('Editing the state\'s focus code: the effect passes through the read only, is specific and graded', fontsize=11)
    fig.tight_layout();fig.savefig(out/'fig_focus_edits.png', dpi=130);plt.close(fig)


NAMES = {'retry_c0.1_ws':'retry (seed-0 start)', 'retry_c0.1_ws_s1':'retry (seed-1 start)', 'retry_c0.1_nonull_ws':'retry, no "nothing new" weight',
         'blind_c0.1_ws':'blind (seed-0 start)', 'blind_c0.1_ws_s1':'blind (seed-1 start)'}


def retry_fig(dirs, out):
    R = {}
    for d in dirs:
        for f in sorted(pathlib.Path(d).glob('evalr_*.json')):R[f.stem[6:]] = json.load(open(f, encoding='utf-8'))
    tags = [t for t in NAMES if t in R]
    if not tags:return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    x = np.arange(len(tags));w = 0.27
    rows = [('present_first_failed', 'second_read_on_tracked', 'first read failed: second read on the tracked record'),
            ('present_first_delivered', 'second_read_on_tracked', 'first read delivered: second read on the tracked record'),
            ('absent', None, 'tracked animal absent: second read on any record')]
    for j, (c, k, lab) in enumerate(rows):
        v = [(R[t]['by_first_read'][c][k] if k else 1-R[t]['by_first_read'][c]['second_read_nothing_new']) for t in tags]
        axes[0].bar(x+(j-1)*w, v, w, label=lab)
    axes[0].set_xticks(x);axes[0].set_xticklabels([NAMES[t] for t in tags], rotation=20, ha='right', fontsize=8);axes[0].set_ylim(0, 1.05)
    axes[0].set_title('what the second read does, by what happened to the first', fontsize=10)
    axes[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=1);axes[0].grid(axis='y', alpha=0.3)
    cf = [(t, R[t]['counterfactual_first_delivery']) for t in tags if R[t].get('counterfactual_first_delivery')]
    conds = [('delivered', 'first read delivered'), ('delivered_query_from_failed_state', "delivered,\nsecond query from the\nfailed branch's state"),
             ('failed', 'first read failed'), ('failed_query_from_delivered_state', "failed,\nsecond query from the\ndelivered branch's state")]
    x2 = np.arange(len(conds));w2 = 0.8/max(1, len(cf))
    for i, (t, c) in enumerate(cf):
        axes[1].bar(x2+(i-(len(cf)-1)/2)*w2, [c[k]['second_read_on_tracked'] for k, _ in conds], w2, label=NAMES[t])
        for xx, (k, _) in zip(x2, conds):
            axes[1].text(xx+(i-(len(cf)-1)/2)*w2, 0.02, f"{c[k]['acc']:.2f}", ha='center', fontsize=6, color='white', rotation=90)
    axes[1].set_xticks(x2);axes[1].set_xticklabels([n for _, n in conds], fontsize=8);axes[1].set_ylim(0, 1.05)
    axes[1].set_title('paired counterfactual: second read on the tracked record\n(white: answer accuracy at that question)', fontsize=10)
    axes[1].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=3);axes[1].grid(axis='y', alpha=0.3)
    fig.tight_layout();fig.savefig(out/'fig_retry.png', dpi=130);plt.close(fig)


def main():
    p = argparse.ArgumentParser();p.add_argument('--edits', nargs='*', default=[]);p.add_argument('--labels', nargs='*', default=[])
    p.add_argument('--retry', nargs='*', default=[]);p.add_argument('--out', default='.');a = p.parse_args()
    out = pathlib.Path(a.out);out.mkdir(parents=True, exist_ok=True)
    if a.edits:focus_fig(a.edits, a.labels or [pathlib.Path(f).stem for f in a.edits], out)
    if a.retry:retry_fig(a.retry, out)


if __name__ == '__main__':
    main()
