"""H2 content-route control (added after the main analysis, exploratory): do the representations predict dispositions
beyond the experimenter's own label of the evidence the state has seen? Baseline = output features + one-hot evidence
label (type x direction) and its interaction with the ambiguity subtype, all centred within scenario like the targets.
If the gains of J / residual / PCA over this baseline vanish, H2's positive result is the content route (the state
encodes what kind of evidence it saw, and evidence type governs robustness); if they remain, the state carries
scenario-specific information about its own responses beyond that label.
    python an_h2_label.py --model qwen3-8b --tag main"""
import argparse, json
import numpy as np, torch
from common import OUT, MODELS, atomic_json, log
from an_e1 import Dict, within, fit_eval, r2, boot_ci, ridge_fit, RNG


def onehot(keys):
    u = sorted(set(keys));ix = {k:i for i, k in enumerate(u)};M = np.zeros((len(keys), len(u)))
    for r, k in enumerate(keys):M[r, ix[k]] = 1.
    return M, u


def main():
    p = argparse.ArgumentParser();p.add_argument('--model', default='qwen3-8b');p.add_argument('--tag', default='main')
    a = p.parse_args();root = OUT/'e1'/a.model/a.tag;layers = MODELS[a.model]['layers']
    dev = 'cuda' if torch.cuda.is_available() else 'cpu';D = Dict(a.model, layers, dev);out = {}
    for fam in ('F2', 'F3'):
        z = np.load(root/'h2'/fam/'states.npz');m = json.loads((root/'h2'/fam/'meta.json').read_text())
        S = m['states'];sid = np.array([s['sid'] for s in S]);split = np.array([s['split'] for s in S])
        lo0 = z['lo0'];LC, cdir, LI, idir = z['LC'], z['cdir'], z['LI'], z['idir']
        sg = np.where(cdir == 0, 1., -1.);Yt = (LC-lo0[:, None])*sg
        si = np.where(idir == 0, 1., -1.)[:, None];Yi = np.concatenate([(LI[:, li, 1:]-LI[:, li, :1])*si for li in range(LI.shape[1])], 1)
        Y = np.concatenate([Yt, Yi], 1);names = m['chal_names']+[f'int{l}_{c}' for l in m['int_layers'] for c in m['icond'][1:]]
        ok = np.isfinite(Y).all(1);tv = z['tops_v'].astype(np.float32)
        p0 = 1/(1+np.exp(-lo0));Hc = -(p0*np.log(p0+1e-9)+(1-p0)*np.log(1-p0+1e-9))
        pt = np.exp(tv);Hf = -(pt*tv).sum(1)
        O = np.stack([lo0, np.abs(lo0), p0, Hc, Hf]+[tv[:, k] for k in range(5)], 1)
        lab = [f"{s['evidence']}|{s['edir']}" for s in S];labs = [f"{s['evidence']}|{s['edir']}|{s.get('sub')}" for s in S]
        L1, u1 = onehot(lab);L2, u2 = onehot(labs)
        tr, va, te = (split == 'train')&ok, (split == 'val')&ok, (split == 'test')&ok
        Yw = within(Y, sid);Ow = within(O, sid);Lw = within(np.concatenate([L1, L2], 1), sid)
        OL = np.concatenate([Ow, Lw], 1)
        Po, _ = fit_eval(Ow[tr], Yw[tr], Ow[va], Yw[va], Ow[te])
        Pl, _ = fit_eval(OL[tr], Yw[tr], OL[va], Yw[va], OL[te])
        Plo, _ = fit_eval(Lw[tr], Yw[tr], Lw[va], Yw[va], Lw[te])
        bestv = None                                                   # same validation-based eligibility as analyse_h2
        for a_ in (1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5):
            sv = r2(Yw[va], ridge_fit(Ow[tr], Yw[tr], a_)(Ow[va]))
            if bestv is None or sv.mean() > bestv.mean():bestv = sv
        wv = [float(Yw[tr, k].var()/max(Y[tr, k].var(), 1e-9)) for k in range(Y.shape[1])]
        el = [k for k in range(Y.shape[1]) if wv[k] >= .2 and bestv[k] <= .7]
        q = dict(n=int(ok.sum()), names=names, eligible=[names[k] for k in el], n_label_levels=[len(u1), len(u2)],
                 R2_output_only=r2(Yw[te], Po).tolist(), R2_label_only=r2(Yw[te], Plo).tolist(),
                 R2_output_plus_label=r2(Yw[te], Pl).tolist(),
                 mean_eligible=dict(output=float(np.mean(r2(Yw[te], Po)[el])), label=float(np.mean(r2(Yw[te], Plo)[el])),
                                    output_plus_label=float(np.mean(r2(Yw[te], Pl)[el]))))
        log('h2label', fam, 'eligible mean R2', q['mean_eligible'])
        rows = {}
        for l in layers:
            for j, pname in ((0, 'evid'), (1, 'dec')):
                H = z[f'H{l}'][:, j].astype(np.float32);mu = H[tr].mean(0);X = H-mu
                J25 = D.recon(l, z[f'{l}_{pname}_idx'][:, :25], z[f'{l}_{pname}_coef25'])
                U, Sv, Vt = np.linalg.svd(X[tr]-X[tr].mean(0), full_matrices=False)
                reps = dict(J25=J25, residual=X, remainder=X-J25, pca25=X@Vt[:25].T,
                            rp25=X@np.linalg.qr(RNG.standard_normal((X.shape[1], 25)))[0])
                rr = {}
                for name, R in reps.items():
                    if R.shape[1] > 512:
                        Ur, Sr, Vr = np.linalg.svd(R[tr]-R[tr].mean(0), full_matrices=False);R = R@Vr[:512].T
                    Xa = np.concatenate([OL, within(R, sid)], 1)
                    P, alpha = fit_eval(Xa[tr], Yw[tr], Xa[va], Yw[va], Xa[te])
                    gte = sid[te];Yt_ = Yw[te][:, el];Pf = P[:, el];Pb = Pl[:, el]
                    def stat(ix):
                        return float(np.mean(r2(Yt_[ix], Pf[ix])-r2(Yt_[ix], Pb[ix])))
                    lo_, hi_, vals = boot_ci(stat, gte, n=500)
                    rr[name] = dict(gain_over_output_plus_label=stat(np.arange(len(gte))), ci=[lo_, hi_],
                                    p_one_sided=float((np.sum(vals <= 0)+1)/(len(vals)+1)), alpha=alpha,
                                    per_target=(r2(Yw[te], P)-r2(Yw[te], Pl)).tolist())
                rows[f'{l}_{pname}'] = rr
                log('h2label', fam, l, pname, {k:(round(v['gain_over_output_plus_label'], 3), [round(x, 3) for x in v['ci']]) for k, v in rr.items()})
        q['by_representation'] = rows;out[fam] = q
        atomic_json(root/'results_h2_label.json', out)
    log('written', root/'results_h2_label.json')


if __name__ == '__main__':
    main()
