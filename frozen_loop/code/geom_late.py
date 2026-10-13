"""Late-window structure of the v3 loops (GEOMETRY_PLAN.md M6/M7 follow-up). CPU only.

v3 called a trajectory 'fixed' when successive late states had cosine > 0.99 (research_v3/code/an_e2.py late_class);
exact equality of z shows that the states keep moving. This script measures what the late motion is: amplitude relative
to the natural spread, flip-flop (period-2-like) vs drift vs white jitter, its dimension, whether its direction is shared
across contexts, atom turnover, and v3's own classification for comparison. Also saves compact z arrays for figures.
    python geom_late.py --root <.../v3/e2/qwen3-8b/main> --out <dir> [--bc BC --bcs BCs]"""
import argparse, collections, json, pathlib
import numpy as np

RNG = np.random.default_rng(20261012)


def v3_class(z):
    """research_v3/code/an_e2.py late_class, verbatim logic."""
    w = z[-16:].astype(np.float64);nw = np.linalg.norm(w, axis=1)+1e-9
    cons = 1-(w[1:]*w[:-1]).sum(1)/(nw[1:]*nw[:-1])
    if np.nanmax(cons) < 0.01:return 'fixed'
    for p in (2, 3, 4):
        c = (w[p:]*w[:-p]).sum(1)/(nw[p:]*nw[:-p])
        if np.nanmin(c) > 0.99:return f'cycle{p}'
    return 'aperiodic'


def late_stats(Z, idx, nat_spread, late=32):
    """Z: (T+1, n, 32) float; idx: (T+1, n, 25). Late window = last `late` steps."""
    ok = np.isfinite(Z.astype(np.float64)).all(axis=(0, 2));n_bad = int((~ok).sum());Z, idx = Z[:, ok], idx[:, ok]
    if Z.shape[1] == 0:return dict(n=0, n_nonfinite=n_bad)
    T = Z.shape[0]-1;W = Z[T-late+1:].astype(np.float64)              # (late, n, 32)
    dev = W-W.mean(0, keepdims=True);amp = np.sqrt((dev**2).sum(-1).mean(0))      # RMS distance to the late mean
    out = collections.defaultdict(list);dirs = []
    for i in range(W.shape[1]):
        D = dev[:, i]
        if amp[i] < 1e-9:
            out['ac1'].append(np.nan);out['pr'].append(np.nan);out['nyq'].append(np.nan);continue
        _, s, Vt = np.linalg.svd(D, full_matrices=False);p = D@Vt[0];dirs.append(Vt[0])
        lam = s**2;out['pr'].append(float(lam.sum()**2/(lam**2).sum()))
        out['ac1'].append(float(np.corrcoef(p[1:], p[:-1])[0, 1]) if np.std(p) > 0 else np.nan)
        F = np.abs(np.fft.rfft(p-p.mean()))**2;out['nyq'].append(float(F[-1]/F[1:].sum()) if F[1:].sum() > 0 else np.nan)
    ac1 = np.array(out['ac1']);pr = np.array(out['pr']);nyq = np.array(out['nyq'])
    # alignment of the main jitter direction between contexts vs random 32-d directions
    al = None
    if len(dirs) > 10:
        Dm = np.stack(dirs);sel = RNG.choice(len(Dm), min(400, len(Dm)), replace=False);C = np.abs(Dm[sel]@Dm[sel].T)[np.triu_indices(len(sel), 1)]
        R = RNG.standard_normal((400, Dm.shape[1]));R /= np.linalg.norm(R, axis=1, keepdims=True);Cr = np.abs(R@R.T)[np.triu_indices(400, 1)]
        al = dict(median_abs_cos=float(np.median(C)), random_median=float(np.median(Cr)), frac_gt_0_9=float(np.mean(C > .9)))
    # atom turnover: share of the 25 atoms that change between successive late steps
    I = idx[T-late+1:];turn = np.mean([[1-len(set(I[t, i].tolist())&set(I[t+1, i].tolist()))/25 for t in range(late-1)] for i in range(I.shape[1])], 1)
    cls = collections.Counter(v3_class(Z[:, i]) for i in range(Z.shape[1]))
    return dict(n=int(Z.shape[1]), n_nonfinite=n_bad, v3_class={k:v/Z.shape[1] for k, v in cls.items()},
                exact_fixed=float(np.mean((Z[-1] == Z[-2]).all(-1)&(Z[-2] == Z[-3]).all(-1))),
                amp_rel_median=float(np.median(amp)/nat_spread), amp_rel_q90=float(np.quantile(amp, .9)/nat_spread),
                ac1_median=float(np.nanmedian(ac1)), frac_flipflop=float(np.nanmean(ac1 < -.5)), frac_drift=float(np.nanmean(ac1 > .5)),
                nyquist_share_median=float(np.nanmedian(nyq)), jitter_pr_median=float(np.nanmedian(pr)),
                jitter_direction_alignment=al, atom_turnover_mean=float(turn.mean()), atom_turnover_q90=float(np.quantile(turn, .9)))


def load(root, stage, pat):
    G = collections.defaultdict(lambda:collections.defaultdict(list))
    for f in sorted((root/stage).glob(pat)):
        meta = json.loads(f.with_suffix('.json').read_text());a = np.load(f);z, idx = a['z'], a['idx']
        for i, r in enumerate(meta['rows']):
            d = G[(r['cond'], r['g'])];d['z'].append(z[:, i]);d['idx'].append(idx[:, i]);d['ctx'].append(r['ctx'])
    return {k:{f:(np.stack(v, 1) if f != 'ctx' else np.array(v)) for f, v in d.items()} for k, d in G.items()}


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True);p.add_argument('--out', required=True)
    p.add_argument('--bc', default='BC');p.add_argument('--bcs', default='BCs')
    a = p.parse_args();root = pathlib.Path(a.root);out = pathlib.Path(a.out);out.mkdir(parents=True, exist_ok=True)
    R = {};keep = {}
    A1 = load(root, 'A1', 'A1_std_*.npz');nat = A1[('nofb', 0.)]['z'][0].astype(np.float64)
    nat_spread = float(np.sqrt(((nat-nat.mean(0))**2).sum(1).mean()))       # RMS distance of natural z to their mean
    R['nat_spread_z'] = nat_spread
    for k, d in sorted(A1.items(), key=str):
        if k[0] == 'nofb':continue
        R[f'A1|{k[0]}@{k[1]}'] = late_stats(d['z'], d['idx'], nat_spread);keep[f'A1|{k[0]}@{k[1]}'] = d['z'].astype(np.float16)
        print('A1', k, R[f'A1|{k[0]}@{k[1]}'], flush=True)
    keep['nat'] = A1[('nofb', 0.)]['z'][0].astype(np.float16);keep['nat_ctx'] = A1[('nofb', 0.)]['ctx']
    for k, d in sorted(load(root, 'A2', 'A2_std_*.npz').items(), key=str):
        if k[0] == 'nofb':continue
        R[f'A2|{k[0]}@{k[1]}'] = late_stats(d['z'], d['idx'], nat_spread);keep[f'A2|{k[0]}@{k[1]}'] = d['z'][:, :300].astype(np.float16)
        print('A2', k, R[f'A2|{k[0]}@{k[1]}'], flush=True)
    for st in (a.bc, a.bcs):
        if not (root/st).exists():continue
        for k, d in sorted(load(root, st, 'BC*_std_*.npz').items(), key=str):
            R[f'{st}|{k[0]}@{k[1]}'] = late_stats(d['z'], d['idx'], nat_spread);keep[f'{st}|{k[0]}@{k[1]}'] = d['z'][:, :300].astype(np.float16)
            print(st, k, R[f'{st}|{k[0]}@{k[1]}'], flush=True)
    (out/'late.json').write_text(json.dumps(R, indent=1, default=float))
    np.savez_compressed(out/'late_z.npz', **{k.replace('|', '__'):v for k, v in keep.items()})
    print('written', out/'late.json', flush=True)


if __name__ == '__main__':
    main()
