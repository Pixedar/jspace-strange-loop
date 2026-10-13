"""One table of every arm of the trained loop (8B and 14B): what the loop carries, self-prediction, enactment, the time code, and the
controls. Reads <dir>/eval/<arm>.json (and <arm>.npz for the geometry measures) and <dir>/ref.npz.
    python summarize_arms.py <results/loopb> <results/loopb14> --out <folder>"""
import argparse, csv, json, pathlib
import numpy as np
from scipy.linalg import orthogonal_procrustes

KS = 8


def rotation_measures(Z, rng):
    """True-order rotation fit of the segment means vs scrambled orders, held-out single-trace prediction, angles."""
    def fit(C, B):
        X, Y = C[:-1]@B.T, C[1:]@B.T;R, _ = orthogonal_procrustes(X, Y)
        return 1-((X@R-Y)**2).sum()/((Y-Y.mean(0))**2).sum(), R
    M = Z.mean(0);C = M-M.mean(0);B = np.linalg.svd(C, full_matrices=False)[2][:6];true, R = fit(C, B)
    perms = np.array([fit(C[list(rng.permutation(KS))], B)[0] for _ in range(2000)])
    n = len(Z);tr, te = np.arange(0, n, 2), np.arange(1, n, 2);Mt = Z[tr].mean(0);Ct = Mt-Mt.mean(0)
    Bt = np.linalg.svd(Ct, full_matrices=False)[2][:6];_, Rt = fit(Ct, Bt);Zc = (Z[te]-Mt.mean(0))@Bt.T
    ho = 1-((Zc[:, :-1]@Rt-Zc[:, 1:])**2).sum()/((Zc[:, 1:]-Zc[:, 1:].mean((0, 1)))**2).sum()
    ang = sorted({round(abs(float(np.degrees(np.angle(e)))), 0) for e in np.linalg.eigvals(R)}, reverse=True)
    return dict(rot_fit=float(true), rot_scrambled_q99=float(np.quantile(perms, .99)), rot_p=float((perms >= true).mean()),
                rot_heldout_r2=float(ho), rot_angles='/'.join(str(int(a)) for a in ang[:3]))


def segment_index_acc(Z):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    n = len(Z);X = Z.reshape(-1, Z.shape[-1]);k = np.tile(np.arange(KS), n);g = np.repeat(np.arange(n), KS)
    return float(np.mean([(LogisticRegression(max_iter=3000).fit(X[a], k[a]).predict(X[b]) == k[b]).mean()
                          for a, b in GroupKFold(5).split(X, k, g)]))


def main():
    p = argparse.ArgumentParser();p.add_argument('dirs', nargs='+');p.add_argument('--out', required=True)
    a = p.parse_args();out = pathlib.Path(a.out);out.mkdir(parents=True, exist_ok=True);rows = [];rng = np.random.default_rng(0)
    for d in map(pathlib.Path, a.dirs):
        model = '14B' if d.name.endswith('14') else '8B'
        cut = d/'eval'/'cut_learned_l0.json';cut_nll = np.mean(json.load(open(cut))['nll_own'][1:]) if cut.exists() else np.nan
        for f in sorted((d/'eval').glob('*.json')):
            r = json.load(open(f, encoding='utf-8'));tag = f.stem
            row = dict(model=model, arm=tag, channel=r.get('chan'), reader=r.get('reader'), lam=r.get('lam'), loss=r.get('ploss', 'cos'),
                       nll_seg1_7=float(np.mean(r['nll_own'][1:])), own_vs_foreign=r['necessity_foreign'][0], own_vs_foreign_se=r['necessity_foreign'][1],
                       own_vs_stale=r['necessity_stale'][0], benefit_over_cut=float(cut_nll-np.mean(r['nll_own'][1:])) if model == '8B' else np.nan,
                       enactment=r.get('swap_pull_forecast', [np.nan])[0], channel_swap_cost=r.get('swap_dnll', np.nan),
                       forecast_retrieval=r.get('retrieval', {}).get('S', np.nan), persistence_retrieval=r.get('retrieval', {}).get('persistence', np.nan),
                       collapse=r.get('collapse', np.nan), return_after_corruption_first=r['identity_own'][0], return_after_corruption_last=r['identity_own'][-1],
                       free_topic_seg1=r.get('free', {}).get('own', {}).get('topic_own', [np.nan, np.nan])[1])
            npz = d/'eval'/f'{tag}.npz'
            if npz.exists():
                z = dict(np.load(npz));Z = z['Z_own']
                if 'retrieval' not in r:                         # older evaluations: recompute retrieval from the states
                    acc = []
                    for k in range(1, KS):
                        A = z['ZH_own'][:, k]/np.linalg.norm(z['ZH_own'][:, k], axis=1, keepdims=True);Bm = Z[:, k]/np.linalg.norm(Z[:, k], axis=1, keepdims=True)
                        acc.append(((A@Bm.T).argmax(1) == np.arange(len(Z))).mean())
                    row['forecast_retrieval'] = float(np.mean(acc))
                if np.isnan(row['collapse']):
                    zf = Z.reshape(-1, Z.shape[-1]);row['collapse'] = float(((zf@zf.mean(0))/(np.linalg.norm(zf, axis=1)*np.linalg.norm(zf.mean(0)))).mean())
                row.update(rotation_measures(Z, rng));row['segment_index_acc'] = segment_index_acc(Z)
            rows.append(row);print(model, tag, 'done')
    keys = list(rows[0].keys()) + [k for k in ('rot_fit', 'rot_scrambled_q99', 'rot_p', 'rot_heldout_r2', 'rot_angles', 'segment_index_acc') if k not in rows[0]]
    with open(out/'arms_summary.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=keys);w.writeheader();[w.writerow({k:r.get(k, '') for k in keys}) for r in rows]
    show = ['model', 'arm', 'nll_seg1_7', 'own_vs_foreign', 'benefit_over_cut', 'enactment', 'forecast_retrieval', 'collapse', 'rot_fit',
            'rot_scrambled_q99', 'rot_heldout_r2', 'rot_angles', 'segment_index_acc']
    fmt = lambda v:(f'{v:.3f}' if isinstance(v, float) and not np.isnan(v) else ('' if isinstance(v, float) else str(v)))
    md = ['| ' + ' | '.join(show) + ' |', '|' + '---|'*len(show)] + ['| ' + ' | '.join(fmt(r.get(k, '')) for k in show) + ' |' for r in rows]
    (out/'arms_summary.md').write_text('\n'.join(md)+'\n', encoding='utf-8');print('wrote', out)


if __name__ == '__main__':
    main()
