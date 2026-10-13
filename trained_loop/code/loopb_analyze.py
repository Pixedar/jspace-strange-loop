"""Local analysis of Phase B (loopb.py) outputs: one table across arms, probes of what the state carries, geometry of the
state and of the self-model, free-generation steering. Reads <dir>/eval/*.json|npz, data.npz, ref.npz, ckpt/*/loop.pt.
    python loopb_analyze.py <dir> [--tok Qwen/Qwen3-8B]"""
import argparse, json, math, pathlib, re
import numpy as np

KS, DZ = 8, 32
HEDGE = re.compile(r"\b(maybe|perhaps|might|possibly|probably|not sure|unsure|I think|I guess|seems|apparently)\b", re.I)
WAIT = re.compile(r"\b(Wait|wait|Hmm|hmm|Actually|actually|Alternatively|alternatively|But wait|Hold on)\b")
SELF = re.compile(r"\b(I|me|my|myself|I'm|I'll|I've|I'd)\b")
USER = re.compile(r"\b(user|they|their|them|you|your)\b", re.I)


def ci(x):
    x = np.asarray(x, float);return float(x.mean()), float(1.96*x.std()/math.sqrt(len(x)))


def twonn(X):
    from scipy.spatial import cKDTree
    d, _ = cKDTree(X).query(X, k=3);r = d[:, 2]/np.maximum(d[:, 1], 1e-12);r = r[np.isfinite(r) & (r > 1)]
    return float(len(r)/np.log(r).sum())


def pr(X):
    ev = np.linalg.eigvalsh(np.cov(X.T));return float(ev.sum()**2/(ev**2).sum())


def grouped_cv(X, y, groups, kind='clf', folds=5):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold
    sc = [];folds = min(folds, len(set(groups.tolist())))
    for tr, te in GroupKFold(folds).split(X, y, groups):
        if kind == 'clf':
            if len(set(y[tr])) < 2:continue
            m = LogisticRegression(C=1., max_iter=3000).fit(X[tr], y[tr]);sc.append((m.predict(X[te]) == y[te]).mean())
        else:
            m = Ridge(alpha=1.).fit(X[tr], y[tr]);p = m.predict(X[te])
            sc.append(1-((y[te]-p)**2).sum()/((y[te]-y[tr].mean())**2).sum())
    return float(np.mean(sc))


def seg_feats(texts):
    f = []
    for t in texts:
        n = max(1, len(t.split()))
        f.append([len(HEDGE.findall(t))/n*100, float(bool(WAIT.search(t))), len(SELF.findall(t))/n*100,
                  len(USER.findall(t))/n*100, float('</think>' in t)])
    return np.array(f)


def main():
    p = argparse.ArgumentParser();p.add_argument('dir');p.add_argument('--tok', default='Qwen/Qwen3-8B')
    p.add_argument('--rev', default='b968826d9c46dd6066d109eabc6255188de91218')
    a = p.parse_args();D = pathlib.Path(a.dir);E = D/'eval'
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok, revision=a.rev or None)
    data = dict(np.load(D/'data.npz'));toks, cat, split = data['toks'], data['cat'], data['split']
    seg = toks.shape[1]//KS;te = np.load(next(E.glob('*.npz')))['te']                  # the evaluated test traces
    texts = [[tok.decode(toks[i, k*seg:(k+1)*seg]) for k in range(KS)] for i in te]
    feats = np.stack([seg_feats(t) for t in texts])                                 # (N, K, F)
    ref = dict(np.load(D/'ref.npz')) if (D/'ref.npz').exists() else None
    if ref is not None:
        pos = {int(i):j for j, i in enumerate(ref['idx'])};full = np.stack([ref['full'][pos[int(i)]] for i in te])
        iso = np.stack([ref['iso'][pos[int(i)]] for i in te])
    else:full = iso = None
    # bag-of-words of each segment (content baseline for probes)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    flat = [t for tt in texts for t in tt];V = TfidfVectorizer(max_features=4000, min_df=2).fit(flat)
    B = TruncatedSVD(min(DZ, V.transform(flat).shape[1]-1), random_state=0).fit_transform(V.transform(flat)).reshape(len(te), KS, -1)
    out = {};tags = sorted(f.stem for f in E.glob('*.json'))
    cut = json.load(open(E/'cut_learned_l0.json')) if (E/'cut_learned_l0.json').exists() else None
    for tag in tags:
        r = json.load(open(E/f'{tag}.json'));z = dict(np.load(E/f'{tag}.npz'));Z = z['Z_own'];N = len(Z)
        row = dict(chan=r['chan'], reader=r['reader'], lam=r['lam'])
        row['nll_own_1to7'] = float(np.mean(r['nll_own'][1:]));row['bnd_own'] = r['bnd_own']
        row['necessity_foreign'] = r['necessity_foreign'];row['necessity_stale'] = r['necessity_stale']
        if cut is not None:row['benefit_vs_cut'] = float(np.mean(cut['nll_own'][1:])-row['nll_own_1to7'])
        if full is not None:row['gap_closed'] = float((iso[:, 1:].mean()-row['nll_own_1to7'])/(iso[:, 1:].mean()-full[:, 1:].mean()))
        row['S_cos'] = float(np.mean(r['S_cos']));row['observer_cos'] = float(np.mean(r['observer_cos']))
        row.update({f'base_{k}':float(np.mean(v)) for k, v in r['baselines'].items()})
        row['pr'] = pr(Z.reshape(-1, DZ));row['twonn'] = twonn(Z.reshape(-1, DZ))
        # collapse-proof predictability (variance-normalised R^2 on test states; segment means leave-one-trace-out)
        Y = Z[:, 1:];vy = Y.reshape(-1, DZ).var(0)+1e-9
        r2 = lambda P_:float(1-(((P_-Y).reshape(-1, DZ)**2).mean(0)/vy).mean())
        loo = (Z[:, 1:].sum(0, keepdims=True)-Z[:, 1:])/(len(Z)-1)
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GroupKFold
        Xl, Yl, gl = Z[:, :-1].reshape(-1, DZ), Y.reshape(-1, DZ), np.repeat(np.arange(len(Z)), KS-1);Pl = np.zeros_like(Yl)
        for tr_, te_ in GroupKFold(5).split(Xl, Yl, gl):Pl[te_] = Ridge(alpha=1.).fit(Xl[tr_], Yl[tr_]).predict(Xl[te_])
        row['R2_S'] = r2(z['ZH_own'][:, 1:]);row['R2_persist'] = r2(Z[:, :-1]);row['R2_segmean'] = r2(loo)
        row['R2_linear'] = r2(Pl.reshape(Y.shape))
        zf = Z.reshape(-1, DZ);row['collapse'] = float(((zf@zf.mean(0))/(np.linalg.norm(zf, axis=1)*np.linalg.norm(zf.mean(0)))).mean())
        for k in ('swap_pull_state', 'swap_pull_forecast', 'swap_stay', 'swap_dnll'):
            if k in r:row[k] = r[k]
        row['identity_own'] = r['identity_own'];row['identity_donor'] = r['identity_donor']
        row['track_fresh'] = float(np.mean(r['track_fresh']));row['track_stale'] = float(np.mean(r['track_stale']))
        # probes (grouped by trace): what z_k carries about segment k (current) and k+1 (next)
        g = np.repeat(np.arange(N), KS);Zf = Z.reshape(-1, DZ);kk = np.tile(np.arange(KS), N)
        row['probe_segment_index'] = grouped_cv(Zf, kk, g)
        row['probe_category'] = grouped_cv(Zf, np.repeat(cat[te], KS), g)
        Bf = B.reshape(N*KS, -1)
        row['probe_category_bow'] = grouped_cv(Bf, np.repeat(cat[te], KS), g)
        nxt = {};names = ['hedge', 'reconsider', 'self_ref', 'other_ref', 'answer_phase']
        Zc, gc = Z[:, :-1].reshape(-1, DZ), np.repeat(np.arange(N), KS-1)
        Bc = np.concatenate([B[:, :-1], np.eye(KS)[None, :-1].repeat(N, 0)], -1).reshape(-1, B.shape[-1]+KS)
        for j, nm in enumerate(names):
            y = feats[:, 1:, j].reshape(-1)
            if y.std() == 0:continue
            nxt[nm] = dict(z=grouped_cv(Zc, y, gc, 'reg'), bow_k=grouped_cv(Bc, y, gc, 'reg'),
                           both=grouped_cv(np.concatenate([Bc, Zc], 1), y, gc, 'reg'))
        if full is not None:
            y = full[:, 1:].reshape(-1)
            nxt['own_surprise_next'] = dict(z=grouped_cv(Zc, y, gc, 'reg'), bow_k=grouped_cv(Bc, y, gc, 'reg'),
                                            both=grouped_cv(np.concatenate([Bc, Zc], 1), y, gc, 'reg'))
        row['probe_next'] = nxt
        # geometry of the segment-index code: class means, their curvature and planarity
        M = np.stack([Z[:, k].mean(0) for k in range(KS)]);d1 = np.diff(M, axis=0)
        ang = [float(np.degrees(np.arccos(np.clip((d1[i]@d1[i+1])/(np.linalg.norm(d1[i])*np.linalg.norm(d1[i+1])+1e-12), -1, 1)))) for i in range(KS-2)]
        U, s, _ = np.linalg.svd(M-M.mean(0), full_matrices=False)
        P2 = (Z-M.mean(0))@np.linalg.svd(M-M.mean(0), full_matrices=False)[2][:2].T     # per-trace phase in the clock plane
        da = np.diff(np.arctan2(P2[..., 1], P2[..., 0]), axis=1);ph = np.exp(1j*da)
        rngc = np.random.default_rng(0);nullc = []
        for _ in range(100):
            Zs = np.stack([Z[i, rngc.permutation(KS)] for i in range(N)]);Ms = np.stack([Zs[:, j].mean(0) for j in range(KS)])
            Ps = (Zs-Ms.mean(0))@np.linalg.svd(Ms-Ms.mean(0), full_matrices=False)[2][:2].T
            nullc.append(np.abs(np.exp(1j*np.diff(np.arctan2(Ps[..., 1], Ps[..., 0]), axis=1)).mean()))
        row['clock'] = dict(turn_deg=float(np.degrees(np.angle(ph.mean()))), coherence=float(np.abs(ph.mean())),
                            null_q99=float(np.quantile(nullc, .99)), same_dir=float((np.sign(np.angle(ph)) == np.sign(np.angle(ph.mean()))).mean()),
                            plane_share=float((np.linalg.norm(P2, axis=-1)**2).mean()/(np.linalg.norm(Z-M.mean(0), axis=-1)**2).mean()))
        if 'retrieval' in r:row['retrieval'] = r['retrieval']
        row['index_code'] = dict(turn_angles=ang, var_top2=float((s[:2]**2).sum()/(s**2).sum()),
                                 spread_vs_within=float(np.linalg.norm(M-M.mean(0), axis=1).mean()/np.mean([Z[:, k].std(0).mean() for k in range(KS)])))
        if 'free' in r:
            fr = r['free'];row['free'] = {k:v for k, v in fr.items() if k != 'examples'}
        row['words'] = r.get('words', {})
        out[tag] = row
    (D/'analysis.json').write_text(json.dumps(out, indent=1))
    keys = ['nll_own_1to7', 'necessity_foreign', 'necessity_stale', 'benefit_vs_cut', 'gap_closed', 'collapse', 'pr', 'R2_S',
            'R2_persist', 'R2_segmean', 'R2_linear', 'swap_dnll', 'swap_pull_forecast', 'probe_segment_index', 'probe_category',
            'probe_category_bow']
    print('tag'.ljust(20), ' '.join(k[:12].rjust(12) for k in keys))
    for tag, row in out.items():
        vals = []
        for k in keys:
            v = row.get(k);v = v[0] if isinstance(v, list) else v
            vals.append(('%.4f' % v if isinstance(v, float) else str(v)).rjust(12))
        print(tag.ljust(20), ' '.join(vals))
    for tag, row in out.items():
        print('\n==', tag, '| necessity foreign %.4f+-%.4f stale %.4f+-%.4f' % (*row['necessity_foreign'], *row['necessity_stale']))
        print('   identity own', np.round(row['identity_own'], 3).tolist(), 'donor', np.round(row['identity_donor'], 3).tolist())
        c = row['clock'];print('   clock: turn %+.1f deg/segment, coherence %.2f (null q99 %.2f), same direction %.2f, plane %.2f of variance' % (c['turn_deg'], c['coherence'], c['null_q99'], c['same_dir'], c['plane_share']))
        if 'retrieval' in row:print('   forecast retrieval', row['retrieval'])
        print('   index code turn angles', np.round(row['index_code']['turn_angles'], 1).tolist(),
              'var top2 %.2f spread/within %.2f' % (row['index_code']['var_top2'], row['index_code']['spread_vs_within']))
        for nm, v in row['probe_next'].items():print('   next %-18s R2 z %.3f  bow+k %.3f  both %.3f' % (nm, v['z'], v['bow_k'], v['both']))
        if 'free' in row:
            for cond, v in row['free'].items():
                if isinstance(v, dict):print('   free %-9s topic_own' % cond, np.round(v['topic_own'], 3).tolist(), 'donor', np.round(v['topic_donor'], 3).tolist())
            print('   free S_cos', np.round(row['free'].get('S_cos_free', []), 3).tolist())


if __name__ == '__main__':
    main()
