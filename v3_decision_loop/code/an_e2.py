"""Experiment 2/3 analysis of a loop run directory (protocol sections 4-5). Writes results_e2.json.
    python an_e2.py /workspace/v3/e2/qwen3-8b/main --model qwen3-8b"""
import sys, json, pathlib, argparse, collections
import numpy as np

RNG = np.random.default_rng(811)


def boot(v, groups, n=1000, fn=np.nanmean):
    v = np.asarray(v, float);groups = np.asarray(groups);ug = np.unique(groups);idx = {g:np.where(groups == g)[0] for g in ug}
    vals = [fn(v[np.concatenate([idx[g] for g in RNG.choice(ug, len(ug))])]) for _ in range(n)]
    return [float(fn(v)), float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def npz(f):
    """Load every array once (NpzFile re-reads and decompresses an array on each access); hidden states are skipped."""
    zz = np.load(f);return {k:zz[k] for k in zz.files if k != 'H'}


def load_stage(root, stage, prefix):
    for f in sorted((root/stage).glob(f'{prefix}_*.npz')):
        if '.tmp' in f.name or not f.with_suffix('.json').exists():continue    # a stage still writing
        yield f, npz(f), json.loads(f.with_suffix('.json').read_text())


def late_class(z):
    """Pre-declared late-window class for one trajectory z (T+1, 32)."""
    w = z[-16:].astype(np.float64);nw = np.linalg.norm(w, axis=1)+1e-9
    cons = 1-(w[1:]*w[:-1]).sum(1)/(nw[1:]*nw[:-1])
    if np.nanmax(cons) < 0.01:return 'fixed'
    for p in (2, 3, 4):
        c = (w[p:]*w[:-p]).sum(1)/(nw[p:]*nw[:-p])
        if np.nanmin(c) > 0.99:return f'cycle{p}'
    return 'aperiodic'


def ridge(X, Y, a=1.):
    mx, my = X.mean(0), Y.mean(0);Xc = X-mx
    W = np.linalg.solve(Xc.T@Xc+a*np.eye(X.shape[1]), Xc.T@(Y-my));return lambda Z:(Z-mx)@W+my


def predictability(Z, nat, sids):
    """Held-out-scenario ridge prediction of late z_{t+1} from (z_t, natural code z); R^2 of late steps per trajectory."""
    T = Z.shape[1]-1;us = np.unique(sids);folds = np.array_split(RNG.permutation(us), 5);r2 = np.full(len(Z), np.nan)
    for fo in folds:
        te = np.isin(sids, fo);tr = ~te
        if tr.sum() < 5 or te.sum() == 0:continue
        X = np.concatenate([np.concatenate([Z[tr, t], nat[tr]], 1) for t in range(T-16, T)]);Y = np.concatenate([Z[tr, t+1] for t in range(T-16, T)])
        f = ridge(X, Y, 10.)
        for i in np.where(te)[0]:
            Xi = np.concatenate([Z[i, T-16:T], np.repeat(nat[i][None], 16, 0)], 1);Yi = Z[i, T-15:T+1]
            P = f(Xi);den = ((Yi-Yi.mean(0))**2).sum()
            r2[i] = 1-((Yi-P)**2).sum()/den if den > 1e-9 else np.nan
    return r2


def analyse(root, words):
    ctxs = json.loads((root/'contexts.json').read_text());res = {}
    # ---------------- regime A (A1 sweep + A2 controls) ----------------
    for stage in ('A1', 'A2', 'X'):
        if not (root/stage).exists():continue
        R = collections.defaultdict(lambda:collections.defaultdict(list))
        for f, z, m in load_stage(root, stage, stage):
            kind = m['kind'];T = z['idx'].shape[0]-1
            nat = {r['unit']:i for i, r in enumerate(m['rows']) if r['cond'] == 'nofb'}
            for i, r in enumerate(m['rows']):
                u = m['units'][r['unit']];key = (kind, r['cond'], r['g']);d = R[key]
                d['fam'].append(u['fam']);d['sid'].append(u['sid']);d['tag'].append(u['tag']);d['truth'].append(ctxs[u['init']]['truth'])
                d['lo'].append(z['lo'][:, i]);d['z'].append(z['z'][:, i].astype(np.float32))
                d['znat'].append(z['z'][1, nat[r['unit']]].astype(np.float32));d['digits'].append(z['digits'][:, i].astype(np.float32))
                ii = z['idx'][:, i];natset = set(z['idx'][1, nat[r['unit']]].tolist())
                d['ov_nat'].append([len(set(ii[t].tolist())&natset)/25 for t in range(T+1)])
                d['turn'].append([np.nan]+[1-len(set(ii[t].tolist())&set(ii[t-1].tolist()))/25 for t in range(1, T+1)])
                d['late_words'].append(collections.Counter(ii[-8:].ravel().tolist()))
                if 'widx' in z and r['cond'] in ('inc', 'out', 'lens'):        # codes actually written at steps T-8..T-1
                    d['late_write_words'].append(collections.Counter(z['widx'][T-8:T, i].ravel().tolist()))
                d['wn'].append(float(z['wnorm'][:, i].mean()))
        out = {}
        for key, d in R.items():
            kind, c, g = key;lo = np.array(d['lo']);Z = np.array(d['z']);sid = np.array(d['sid']);tag = np.array(d['tag'])
            fam = np.array(d['fam']);truth = np.array(d['truth']);T = lo.shape[1]-1;o = dict(n=len(lo), write_norm=float(np.mean(d['wn'])))
            cls = [late_class(x) for x in Z];o['classes'] = dict(collections.Counter(cls))
            o['overlap_with_natural'] = {t:float(np.mean(np.array(d['ov_nat'])[:, t])) for t in (1, 2, 4, 16, T)}
            o['late_turnover'] = float(np.nanmean(np.array(d['turn'])[:, -16:]))
            zs = np.linalg.norm(np.diff(Z, axis=1), axis=-1);o['late_z_step'] = float(zs[:, -16:].mean())
            o['predictability_late_R2'] = float(np.nanmedian(predictability(Z, np.array(d['znat']), sid)))
            # collapse: do different contexts converge onto shared states? ratio of final pairwise distance to natural
            zf = Z[:, -1];zn = np.array(d['znat']);sel = RNG.choice(len(zf), min(200, len(zf)), replace=False)
            dfin = np.linalg.norm(zf[sel][:, None]-zf[sel][None], axis=-1);dnat = np.linalg.norm(zn[sel][:, None]-zn[sel][None], axis=-1)
            iu = np.triu_indices(len(sel), 1);o['context_contraction'] = float(np.median(dfin[iu])/max(np.median(dnat[iu]), 1e-9))
            wc = collections.Counter()
            for x in d['late_words']:wc.update(x)
            o['late_top_words'] = [(words[k], v) for k, v in wc.most_common(25)]
            if d.get('late_write_words'):
                ww = collections.Counter()
                for x in d['late_write_words']:ww.update(x)
                o['late_write_words'] = [(words[k], v) for k, v in ww.most_common(25)]
            if kind == 'mem':
                gaps = {};gid = []
                for s in np.unique(sid):
                    ia = np.where((sid == s)&(tag == 'A'))[0];ib = np.where((sid == s)&(tag == 'B'))[0]
                    if len(ia) and len(ib):gaps.setdefault(fam[ia[0]], []).append(lo[ia[0]]-lo[ib[0]])
                o['memory_gap'] = {f:{t:boot(np.array(v)[:, t], np.arange(len(v)), 500) for t in (0, 1, 2, 4, 8, 16, 32, T)} for f, v in gaps.items()}
                # is the initial evidence still present in the workspace state? held-out-scenario decoding of A vs B init
                from sklearn.linear_model import LogisticRegression
                ab = np.isin(tag, ['A', 'B']);y = (tag[ab] == 'A').astype(int);gs = sid[ab];us = np.unique(gs)
                folds = np.array_split(RNG.permutation(us), 5);dec = {}
                for t in (1, 4, 16, T):
                    X0 = Z[ab, t];ok = [];fin = np.isfinite(X0).all(1)
                    for fo in folds:
                        te = np.isin(gs, fo)&fin;trm = (~np.isin(gs, fo))&fin
                        if not trm.any() or not te.any() or len(np.unique(y[trm])) < 2:continue
                        m_, s_ = X0[trm].mean(0), X0[trm].std(0)+1e-6
                        clf = LogisticRegression(C=0.1, max_iter=500).fit((X0[trm]-m_)/s_, y[trm]);ok += list(clf.predict((X0[te]-m_)/s_) == y[te])
                    dec[t] = float(np.mean(ok)) if ok else np.nan
                o['evidence_decodability'] = dec
            else:
                acc = {}
                for f in np.unique(fam):
                    m_ = fam == f
                    if f == 'F5':ok = (np.array(d['digits'])[m_].argmax(-1) == truth[m_][:, None])
                    else:
                        mm = m_&(truth >= 0)
                        if not mm.any():continue
                        ok = (lo[mm] > 0) == (truth[mm] == 0)[:, None];m_ = mm
                    acc[f] = {t:boot(ok[:, t].astype(float), sid[m_], 300) for t in (0, 1, 4, 16, T)}
                o['accuracy'] = acc
            out[f'{kind}|{c}|{g}'] = o
        res[stage] = out
    # ---------------- regimes B / C (pre-declared gains) and BCs (exploratory strong coupling) ----------------
    for bcdir in ('BC', 'BCs', 'BC1'):
        if (root/bcdir).exists():res[bcdir] = analyse_bc(root, bcdir, ctxs)
    return analyse_pert(root, res)


def analyse_bc(root, bcdir, ctxs):
    if True:
        R = collections.defaultdict(lambda:collections.defaultdict(list))
        import itertools
        for f, z, m in itertools.chain(load_stage(root, bcdir, 'BC'), load_stage(root, bcdir, 'BCY')):
            for i, r in enumerate(m['rows']):
                u = m['units'][r['unit']];d = R[(m['kind'], r['cond'], r['g'])]
                d['pe'].append(z['perr'][:, i]);d['act'].append(z['act'][:, i]);d['z'].append(z['z'][:, i].astype(np.float32))
                d['lo'].append(z['lo'][:, i]);d['sid'].append(u['sid']);d['tag'].append(u['tag']);d['fam'].append(u['fam'])
                d['truth'].append(ctxs[u['init']]['truth']);d['digits'].append(z['digits'][:, i].astype(np.float32))
                d['seed'].append(r.get('seed', m.get('seed', 0)));d['chunk'].append(f.name.split('_')[2].split('.')[0])
        out = {}
        for key, d in R.items():
            kind, c, g = key;pe = np.array(d['pe']);Z = np.array(d['z']);act = np.array(d['act']);T = pe.shape[1]
            sysid = np.array([f'{a}_{b}' for a, b in zip(d['chunk'], d['seed'])])            # independent replicate systems
            zs = np.linalg.norm(np.diff(Z, axis=1), axis=-1)
            win = lambda x, a, b:np.nanmean(x[:, a:b], 1)
            late = act[:, T//2:];p = np.stack([(late == k).mean(1) for k in range(9)], 1)
            Hs = -(np.where(p > 0, p*np.log(np.where(p > 0, p, 1)), 0)).sum(1)
            o = dict(n=len(pe), perr={w:boot(win(pe, a, b), sysid, 500) for w, (a, b) in dict(early=(0, 8), mid=(16, 32), late=(48, T)).items()},
                     z_step={w:boot(win(zs, a, b), sysid, 500) for w, (a, b) in dict(early=(0, 8), mid=(16, 32), late=(48, T)).items()},
                     action_entropy_late=boot(Hs, sysid, 500), classes=dict(collections.Counter(late_class(x) for x in Z)),
                     late_action_hist=[float((late == k).mean()) for k in range(9)],
                     perr_trend=float(np.polyfit(np.arange(16, T), np.nanmean(pe[:, 16:], 0), 1)[0]),
                     zstep_trend=float(np.polyfit(np.arange(16, T), zs[:, 16:].mean(0), 1)[0]))
            if kind == 'mem':
                sid = np.array(d['sid']);tag = np.array(d['tag']);fam = np.array(d['fam']);seed = np.array(d['seed']);lo = np.array(d['lo'])
                gaps = {};gsys = {}
                for s in np.unique(sid):
                    for sd in np.unique(seed):
                        ia = np.where((sid == s)&(tag == 'A')&(seed == sd))[0];ib = np.where((sid == s)&(tag == 'B')&(seed == sd))[0]
                        if len(ia) and len(ib):
                            gaps.setdefault(fam[ia[0]], []).append(lo[ia[0]]-lo[ib[0]]);gsys.setdefault(fam[ia[0]], []).append(sysid[ia[0]])
                # pairs sharing one online self-model are coupled: resample replicate systems (chunk x seed)
                o['memory_gap'] = {f:{t:boot(np.array(v)[:, t], np.array(gsys[f]), 300) for t in (1, 4, 16, T)} for f, v in gaps.items()}
            out[f'{kind}|{c}|{g}'] = o
        # paired self-model contrasts at matched replicate systems: B - R, C1 - R, C2 - R, F - R (late perr, late z step)
        con = {}
        for kind in ('std', 'mem'):
            for g in sorted({k[2] for k in R if k[0] == kind}):
                ref = R.get((kind, 'R', g))
                if ref is None:continue
                for c in ('B', 'C1', 'C2', 'F'):
                    x = R.get((kind, c, g))
                    if x is None:continue
                    sx = np.array([f'{a}_{b}' for a, b in zip(x['chunk'], x['seed'])]);sr = np.array([f'{a}_{b}' for a, b in zip(ref['chunk'], ref['seed'])])
                    for meas in ('pe', 'z'):
                        if meas == 'pe':vx = np.nanmean(np.array(x['pe'])[:, 48:], 1);vr = np.nanmean(np.array(ref['pe'])[:, 48:], 1)
                        else:
                            vx = np.linalg.norm(np.diff(np.array(x['z']), axis=1), axis=-1)[:, 48:].mean(1)
                            vr = np.linalg.norm(np.diff(np.array(ref['z']), axis=1), axis=-1)[:, 48:].mean(1)
                        us = sorted(set(sx)&set(sr));dx = np.array([vx[sx == s].mean()-vr[sr == s].mean() for s in us])
                        con[f'{kind}|{c}-R|{g}|{"late_perr" if meas == "pe" else "late_zstep"}'] = boot(dx, np.arange(len(dx)), 1000)+[len(dx)]
        out['contrasts_vs_random_actions'] = con
        return out


def analyse_pert(root, res):
    if (root/'pert').exists():
        out = collections.defaultdict(lambda:collections.defaultdict(list))
        for f, z, m in load_stage(root, 'pert', 'P'):
            tp = m['t_p'];rows = m['rows'];base = {(r['unit'], r['g']):i for i, r in enumerate(rows) if r['pert'] == 'base'}
            for i, r in enumerate(rows):
                if r['pert'] == 'base':continue
                if r['pert'] == 'trans' and r.get('partner') is None:continue        # no opposite partner in the chunk
                b = base[(r['unit'], r['g'])];zz = z['z'][:, i].astype(np.float32);zb = z['z'][:, b].astype(np.float32)
                dist = np.linalg.norm(zz-zb, axis=-1)/(np.linalg.norm(zb, axis=-1)+1e-6);dlo = z['lo'][:, i]-z['lo'][:, b]
                ov = [len(set(z['idx'][t, i].tolist())&set(z['idx'][t, b].tolist()))/25 for t in range(len(dist))]
                d = out[(r['pert'], r['g'])];d['dist'].append(dist);d['dlo'].append(dlo);d['ov'].append(ov)
                d['pre_exact'].append(float(np.abs(dist[:tp+1]).max()))
        P = {}
        for (k, g), d in out.items():
            dist = np.array(d['dist']);dlo = np.array(d['dlo']);ov = np.array(d['ov']);T = dist.shape[1]-1
            init = np.abs(dlo[:, 17]) if T >= 17 else np.abs(dlo[:, -1])
            fate = []
            for i in range(len(dist)):
                if dist[i, -1] < 0.05*max(dist[i, 17], 1e-6):fate.append('decays')
                elif abs(dlo[i, -1]) < 0.5:fate.append('persists_numerically')
                else:fate.append('persistent_behavioural_change')
            P[f'{k}|{g}'] = dict(n=len(dist), pre_perturbation_max_dist=float(np.max(d['pre_exact'])),
                                 dist={t:float(np.median(dist[:, t])) for t in (17, 20, 32, T)},
                                 abs_dlo={t:float(np.mean(np.abs(dlo[:, t]))) for t in (17, 20, 32, T)},
                                 atom_overlap={t:float(np.mean(ov[:, t])) for t in (17, 20, 32, T)}, fates=dict(collections.Counter(fate)))
        res['pert'] = P
        # system-level perturbations of the self-model regimes: every row of a replicate system perturbed at t_p,
        # compared with the unperturbed twin pass ('none') of the same chunk and seed
        Q = collections.defaultdict(lambda:collections.defaultdict(list))
        for f in sorted((root/'pert').glob('PBC_*_none.npz')):
            pi = f.name.split('_')[1];zb = npz(f);mb = json.loads(f.with_suffix('.json').read_text())
            for kind in ('eps', 'self', 'trans'):
                fp = root/'pert'/f'PBC_{pi}_{kind}.npz'
                if not fp.exists():continue
                zp = npz(fp)
                for i, r in enumerate(mb['rows']):
                    if kind == 'trans' and r.get('partner') is None:continue
                    a = zp['z'][:, i].astype(np.float32);b = zb['z'][:, i].astype(np.float32)
                    d = Q[(kind, r['cond'])];d['dist'].append(np.linalg.norm(a-b, axis=-1)/(np.linalg.norm(b, axis=-1)+1e-6))
                    d['dlo'].append(zp['lo'][:, i]-zb['lo'][:, i]);d['dpe'].append(zp['perr'][:, i]-zb['perr'][:, i])
        S = {}
        for (kind, c), d in Q.items():
            dist = np.array(d['dist']);dlo = np.array(d['dlo']);dpe = np.array(d['dpe']);T = dist.shape[1]-1
            S[f'{kind}|{c}'] = dict(n=len(dist), pre_max=float(dist[:, :16].max()), dist={t:float(np.median(dist[:, t])) for t in (17, 20, 32, T)},
                                    abs_dlo={t:float(np.mean(np.abs(dlo[:, t]))) for t in (17, 20, 32, T)},
                                    dperr_late=float(np.nanmean(dpe[:, 48:])))
        res['pert_selfmodel'] = S
    return res


def main():
    p = argparse.ArgumentParser();p.add_argument('root');p.add_argument('--model', default='qwen3-8b')
    a = p.parse_args();root = pathlib.Path(a.root)
    from an_e1 import Dict
    import torch
    lr = json.loads((root/'provenance.json').read_text())['Lr']
    D = Dict(a.model, [lr], 'cuda' if torch.cuda.is_available() else 'cpu')
    res = analyse(root, D.words)
    (root/'results_e2.json').write_text(json.dumps(res, indent=1, default=float))
    print('written', root/'results_e2.json')


if __name__ == '__main__':
    main()
