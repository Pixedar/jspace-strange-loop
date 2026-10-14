"""Research v7 analysis: all evaluated arms of one run folder (eval_<tag>.json/.npz; the same 512 held-out streams for
every arm, so differences are paired by stream).
  1. accuracy by question type (tracked animal updated / absent so the colour must be kept / at a switch), seed means
  2. paired bootstrap 95 % intervals for state-set vs constant query (per stream, over seeds)
  3. interventions: query-only swap, sham, counterfactual replay
  4. what the state holds: cross-validated decoding of the tracked animal and of its current colour from the state, and
     of the tracked animal from the query; does the self-model's forecast keep the tracked animal?
    python analyze_v7.py --dir <run folder> [--out summary.md]"""
import argparse, json, pathlib, re, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def arm_of(tag):
    return re.sub(r'_s\d+$', '', tag)


def boot(x, n=2000, seed=0):
    r = np.random.default_rng(seed);x = np.asarray(x, float);m = [x[r.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def decode(X, y, groups=None, folds=5):
    """Cross-validated accuracy; folds hold out whole streams when groups (stream ids) are given."""
    if len(np.unique(y)) < 2:return None
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    cv = GroupKFold(folds) if groups is not None else folds
    return float(cross_val_score(m, X, y, cv=cv, groups=groups).mean())


def main():
    p = argparse.ArgumentParser();p.add_argument('--dir', required=True);p.add_argument('--out', default=None);a = p.parse_args()
    sys.stdout.reconfigure(encoding='utf-8');d = pathlib.Path(a.dir)
    E = {f.stem[5:]:json.load(open(f)) for f in sorted(d.glob('eval_*.json'))};N = {t:dict(np.load(d/f'eval_{t}.npz')) for t in E if (d/f'eval_{t}.npz').exists()}
    L = [f'# v7 results: {d}', '', '| run | overall | tracked updated | colour kept (absent) | at a switch | no memory (chunks 1-5) | on tracked record | on "nothing new" when absent |',
         '|---|---|---|---|---|---|---|---|']
    for t, r in E.items():
        s = r.get('selection') or {};nm = np.mean(r['no_memory_acc_by_chunk'][1:])
        L.append(f"| {t} | {r['acc']:.3f} | {r['acc_updated']:.3f} | {r['acc_kept']:.3f} | {r['acc_at_switch'] or float('nan'):.3f} | {nm:.3f} | "
                 f"{s.get('on_tracked_record_when_present', float('nan')):.3f} | {s.get('on_nothing_new_when_absent', float('nan')):.3f} |")
    # seed means per arm
    arms = {}
    for t, r in E.items():arms.setdefault(arm_of(t), []).append(r)
    L += ['', '| arm | seeds | overall | tracked updated | colour kept | at a switch |', '|---|---|---|---|---|---|']
    for ar, rs in arms.items():
        f = lambda k:(np.mean([x[k] for x in rs if x[k] is not None]), np.std([x[k] for x in rs if x[k] is not None]))
        cells = [f'{f(k)[0]:.3f} ± {f(k)[1]:.3f}' for k in ('acc', 'acc_updated', 'acc_kept', 'acc_at_switch')]
        L.append(f'| {ar} | {len(rs)} | ' + ' | '.join(cells) + ' |')
    # paired: state vs const, per stream (mean over the stream's questions), pooled over matching seeds
    pairs = [(f'sel-state_l0{suf}', f'sel-const_l0{suf}', 'constant query') for suf in ('', '_s1', '_s2')]   # matched seeds only
    pairs.append(('sel-state_l0', 'sel-instr_l0', 'instruction-only query (matched control)'))
    for ts, tc, cname in pairs:
        suf = ts[len('sel-state_l0'):]
        if ts not in N or tc not in N:continue
        okS, okC, upd = N[ts]['ok'], N[tc]['ok'], N[ts]['upd']
        free = np.ones_like(okS, dtype=bool);free[:, 0] = False                  # instruction-free chunks: not chunk 0, not a switch
        for i, k in enumerate(N[ts]['switch']):
            if k >= 0:free[i, k] = False
        ds = okS.mean(1)-okC.mean(1);later = np.array([okS[i][free[i]].mean()-okC[i][free[i]].mean() for i in range(len(okS))])
        du = np.array([(okS[i][upd[i]].mean()-okC[i][upd[i]].mean()) if upd[i].any() else np.nan for i in range(len(upd))]);du = du[~np.isnan(du)]
        (m, lo, hi), (ml, lol, hil), (mu, lou, hiu) = boot(ds), boot(later), boot(du)
        L += ['', f'Seed{suf or "_s0"}: state minus {cname}, per stream (paired, 95 % bootstrap over streams): overall {m:+.3f} [{lo:+.3f}, {hi:+.3f}]; '
                  f'instruction-free chunks only (not chunk 0, not a switch) {ml:+.3f} [{lol:+.3f}, {hil:+.3f}]; chunks where the tracked animal is updated {mu:+.3f} [{lou:+.3f}, {hiu:+.3f}].']
    # interventions
    L += ['', '| run | own: answers A / B | sham: A / B | query from B: A / B | replay chunks 1-t: follows B / A | after restoring: follows A |', '|---|---|---|---|---|---|']
    for t, r in E.items():
        qs = r.get('query_swap') or {};rp = r.get('replay') or {}
        g = lambda k:(f"{qs[k]['answers_A']:.3f} / {qs[k]['answers_B']:.3f}" if k in qs else '–')
        L.append(f"| {t} | {g('own')} | {g('sham_identical_query')} | {g('query_from_B')} | "
                 + (f"{rp['chunks_1_to_t_follow_B']:.3f} / {rp['chunks_1_to_t_follow_A']:.3f} | {rp['after_restoring_follow_A']:.3f} |" if rp else '– | – |'))
    # what the state holds
    L += ['', '| run | tracked animal from the state | its colour from the state | tracked animal from the query | chance |', '|---|---|---|---|---|']
    for t, n in N.items():
        Z = n['z'].reshape(-1, n['z'].shape[-1]);fo = n['focus'].reshape(-1);go = n['gold'].reshape(-1)
        Q = n['q'].reshape(-1, n['q'].shape[-1]) if 'q' in n else None
        sub = np.arange(len(Z))[::2]                                        # every second question keeps it fast
        gid = np.repeat(np.arange(n['z'].shape[0]), n['z'].shape[1])        # stream of each question (grouped folds)
        a_f = decode(Z[sub], fo[sub], gid[sub]);a_c = decode(Z[sub], go[sub], gid[sub])
        Qn = n['q'][:, 1:].reshape(-1, n['q'].shape[-1]) if 'q' in n else None
        gq = np.repeat(np.arange(n['q'].shape[0]), n['q'].shape[1]-1) if 'q' in n else None
        a_q = decode(Qn[::2], n['focus'][:, 1:].reshape(-1)[::2], gq[::2]) if Qn is not None and Qn.std() > 1e-6 else None
        L.append(f"| {t} | {a_f:.3f} | {a_c:.3f} | {('%.3f' % a_q) if a_q is not None else '–'} | 0.100 |")
    # facts about its own processing: did the state just take in something new; how confident was its own selection
    L += ['', '| run | "took in something new" from the state (balanced acc.) | selection confidence from the state (R²) | self-model keeps the focus (forecast vs next state; persistence) |',
          '|---|---|---|---|']
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict
    for t, n in N.items():
        if 'att' not in n:continue
        Z = n['z'].reshape(-1, n['z'].shape[-1]);att = n['att'].reshape(-1, n['att'].shape[-1]);took = (att[:, -1] < 0.5).astype(int)
        conf = att.max(-1);sub = np.arange(len(Z))[::2]
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))
        if min((took[sub] == 0).sum(), (took[sub] == 1).sum()) >= 10:
            pred = cross_val_predict(m, Z[sub], took[sub], cv=5);bal = float(np.mean([(pred[took[sub] == c] == c).mean() for c in (0, 1)]))
        else:bal = None                                      # (the read almost never or almost always chose 'nothing new')
        rp = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 11))), Z[sub], conf[sub], cv=5)
        r2 = float(1-((conf[sub]-rp)**2).sum()/((conf[sub]-conf[sub].mean())**2).sum())
        fk = '–'
        ck = d/'ckpt'/f'{t}.pt'
        if ck.exists():
            import torch
            sd = torch.load(ck, map_location='cpu');Wn = [k for k in sd if k.startswith('S.')]
            S = torch.nn.Sequential(torch.nn.Linear(Z.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, Z.shape[1]))
            S.load_state_dict({k[2:]:sd[k] for k in Wn})
            zz = torch.tensor(n['z'], dtype=torch.float32)
            with torch.no_grad():
                fc = S(zz[:, :-1]);fc = (np.sqrt(Z.shape[1])*fc/fc.norm(dim=-1, keepdim=True)).numpy()   # the loop's forecast lives on the sphere
            fo = n['focus'];clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Z[sub], n['focus'].reshape(-1)[sub])
            nxt = fo[:, 1:].reshape(-1);pf = clf.predict(fc.reshape(-1, fc.shape[-1]));pp = clf.predict(n['z'][:, :-1].reshape(-1, Z.shape[1]))
            fk = f'{(pf == nxt).mean():.3f}; {(pp == nxt).mean():.3f}'
        L.append(f"| {t} | {('%.3f' % bal) if bal is not None else '–'} | {r2:.3f} | {fk} |")
    # own action or outside event? where the read's own choice ('took something in') disagrees with what happened ('the
    # tracked animal was updated'), which of the two does the state carry?
    L += ['', '| run | cases where own choice ≠ what happened | decoder of own choice right there | decoder of what happened right there |', '|---|---|---|---|']
    for t, n in N.items():
        if 'att' not in n or 'upd' not in n:continue
        Z = n['z'].reshape(-1, n['z'].shape[-1]);took = (n['att'].reshape(-1, n['att'].shape[-1])[:, -1] < 0.5).astype(int)
        upd = n['upd'].reshape(-1).astype(int);D = took != upd
        if D.sum() < 20 or min((took == 0).sum(), (took == 1).sum(), (upd == 0).sum(), (upd == 1).sum()) < 10:
            L.append(f'| {t} | {int(D.sum())} | – | – |');continue
        mk = lambda:make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))
        pt = cross_val_predict(mk(), Z, took, cv=5);pu = cross_val_predict(mk(), Z, upd, cv=5)
        L.append(f'| {t} | {int(D.sum())} ({D.mean():.1%}) | {(pt[D] == took[D]).mean():.3f} | {(pu[D] == upd[D]).mean():.3f} |')
    txt = '\n'.join(L);print(txt)
    if a.out:pathlib.Path(a.out).write_text(txt+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
