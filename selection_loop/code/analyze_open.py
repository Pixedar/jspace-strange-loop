"""Research v8 analysis (open access, DESIGN_open.md): accuracy with and without memory (paired by stream, bootstrap),
what the state holds (the tracked animal's new colour vs the new colours of the other animals updated in the same
chunk; linear and small nonlinear decoders, folds hold out whole streams), focus edits (with 95 % Wilson intervals) and
the frozen model's own attention.
    python analyze_open.py <run folder> [...] [--out summary.md]"""
import argparse, json, pathlib, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def boot(x, n=2000, seed=0):
    r = np.random.default_rng(seed);x = np.asarray(x, float);m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wilson(p, n, z=1.96):
    if n == 0:return (np.nan, np.nan)
    d = 1+z*z/n;c = (p+z*z/(2*n))/d;h = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h, c+h


def decode(X, y, g, kind):
    """Cross-validated accuracy (folds hold out whole streams)."""
    if len(np.unique(y)) < 2 or len(y) < 50:return None
    m = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)) if kind == 'linear' else
         make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128,), max_iter=600, early_stopping=True, random_state=0)))
    p = cross_val_predict(m, X, y, cv=GroupKFold(5), groups=g);return float((p == y).mean())


def held(n):
    """What the state holds after chunk k (k >= 1): (a) the new colour of the tracked animal, on chunks where it was
    updated; (b) the new colour of each other animal updated in the same chunk (one decoder per animal, pooled)."""
    z, focus, rec_an, rec_col = n['z'], n['focus'], n['rec_an'], n['rec_col'];N, K = focus.shape;out = {}
    for kind in ('linear', 'nonlinear'):
        Xa, ya, ga = [], [], [];per = {}
        for b in range(N):
            for k in range(1, K):
                for j in range(rec_an.shape[2]):
                    an, c = int(rec_an[b, k, j]), int(rec_col[b, k, j])
                    if an == focus[b, k]:Xa.append(z[b, k]);ya.append(c);ga.append(b)
                    else:per.setdefault(an, ([], [], []));per[an][0].append(z[b, k]);per[an][1].append(c);per[an][2].append(b)
        a_t = decode(np.array(Xa), np.array(ya), np.array(ga), kind)
        accs, ns = [], []
        for an, (X, y, g) in sorted(per.items()):
            r = decode(np.array(X), np.array(y), np.array(g), kind)
            if r is not None:accs.append(r);ns.append(len(y))
        out[kind] = dict(tracked_new_colour=a_t, n_tracked=len(ya), other_new_colours=float(np.average(accs, weights=ns)) if accs else None,
                         n_other=int(sum(ns)), other_by_animal=[round(x, 3) for x in accs])
    return out


def main():
    p = argparse.ArgumentParser();p.add_argument('dirs', nargs='+');p.add_argument('--out', default=None);p.add_argument('--held_out', default=None)
    a = p.parse_args();sys.stdout.reconfigure(encoding='utf-8');L = [];HELD = {}
    for d in map(pathlib.Path, a.dirs):
        for f in sorted(d.glob('eval_open_*.json')):
            r = json.load(open(f, encoding='utf-8'));n = dict(np.load(f.with_suffix('.npz')));tag = r['tag'];cfg = r['cfg']
            L += [f"## {tag} — {cfg['model']} (state of {cfg['dz']} numbers; written at block {cfg['lw']}, read at block {cfg['lr']})", '']
            ok, okn, upd, sw = n['ok'], n['ok_no_memory'], n['upd'].astype(bool), n['switch']
            free = np.ones_like(ok, bool);free[:, 0] = False
            for b, k in enumerate(sw):
                if k >= 0:free[b, k] = False
            dd = np.array([ok[b][free[b]].mean()-okn[b][free[b]].mean() for b in range(len(ok))]);m, lo, hi = boot(dd)
            L += ['| overall | tracked animal updated | colour kept (absent) | instruction-free chunks | same, no memory | per stream: with minus without memory (instruction-free) |',
                  '|---|---|---|---|---|---|',
                  f"| {r['acc']:.3f} | {r['acc_updated']:.3f} | {r['acc_kept']:.3f} | {r['acc_instruction_free']:.3f} | {r['no_memory_instruction_free']:.3f} | {m:+.3f} [{lo:+.3f}, {hi:+.3f}] |", '']
            L += ['By chunk: ' + ', '.join(f'{x:.2f}' for x in r['acc_by_chunk']) + ' (no memory: ' + ', '.join(f'{x:.2f}' for x in r['no_memory_acc_by_chunk']) + ')', '']
            H = held(n);HELD[tag+'|'+cfg['model']] = H
            L += ['What the state holds after a chunk (cross-validated decoding, whole streams held out; chance about 0.10):', '',
                  '| decoder | new colour of the tracked animal | new colours of the other animals updated in the same chunk |', '|---|---|---|']
            fmt = lambda v:'–' if v is None else f'{v:.3f}'
            for kind, h in H.items():
                L.append(f"| {kind} | {fmt(h['tracked_new_colour'])} (n {h['n_tracked']}) | {fmt(h['other_new_colours'])} (n {h['n_other']}; per animal {h['other_by_animal']}) |")
            E = r['focus_edits'];npairs = 3*r.get('n_pairs', 256)
            L += ['', f'Focus edits on twin streams (edit at chunk 2, 3 or 4; {npairs} pairs; 95 % intervals):', '',
                  "| edit | A's new colour | B's new colour | A's previous colour | next chunk: A / B |", '|---|---|---|---|---|']
            names = {'none':'no edit', 'focus_A_to_B':'focus A → B before the chunk is read', 'focus_A_to_B_after_read':'same edit after the chunk is read (question only)',
                     'focus_A_to_B_x0.5':'half strength', 'random_same_size':'random direction, same size', 'toward_absent_D':'toward an animal absent from the chunk'}
            for c, nm in names.items():
                e = E.get(c)
                if not e:continue
                ci = lambda v:'%.3f [%.2f, %.2f]' % ((v,)+wilson(v, npairs))
                L.append(f"| {nm} | {ci(e['answers_A_new'])} | {ci(e['answers_B_new'])} | {ci(e['answers_A_stored'])} | {e['next_answers_A']:.3f} / {e['next_answers_B']:.3f} |")
            At = r.get('attention_chunk3')
            if At:
                L += ['', "The frozen model's own attention from the chunk's last token while it reads chunk 3 (mean over heads and over layers "
                      f"{cfg['lw']+1}-{cfg['lr']}; twin A streams):", '', "| condition | on A's record | on B's record | on each other record | on the placeholders |", '|---|---|---|---|---|']
                cn = {'own':'trained state', 'focus_A_to_B':'focus edit A → B', 'no_memory':'no memory (initial state)', 'no_write':'nothing written'}
                for c, nm in cn.items():
                    x = At.get(c)
                    if x:L.append(f"| {nm} | {x['A']:.4f} | {x['B']:.4f} | {x['other']:.4f} | {x['placeholders']:.4f} |")
                pl = At['own']['per_layer'];pe = At['focus_A_to_B']['per_layer']
                L += ['', 'Per layer, A / B (trained state → after the edit): ' + '; '.join(f"{l}: {pl[l]['A']:.3f}/{pl[l]['B']:.3f} → {pe[l]['A']:.3f}/{pe[l]['B']:.3f}" for l in sorted(pl, key=int)), '']
    txt = '\n'.join(L);print(txt)
    if a.out:pathlib.Path(a.out).write_text(txt+'\n', encoding='utf-8')
    if a.held_out:json.dump(HELD, open(a.held_out, 'w'), indent=1)


if __name__ == '__main__':
    main()
