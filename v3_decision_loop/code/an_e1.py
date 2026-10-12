"""Experiment 1 analysis (protocol section 3 + amendment 9). Runs where the raw arrays live (the instance) and writes a
compact results JSON. Needs the lens dictionaries to re-synthesize J reconstructions, so it loads the tokenizer/lens
but not the model weights.
    python an_e1.py --model qwen3-8b --tag main"""
import argparse, json, math, pathlib
import numpy as np, torch
from common import OUT, MODELS, LENS_REPO, LENS_REV, atomic_json, log, english_vocab

RNG = np.random.default_rng(811);NB = 2000


# ------------------------------------------------------------------------------------------------------------------
class Dict:
    """Dictionary atoms per layer without loading the model (only the unembedding and final-norm weights)."""

    def __init__(self, key, layers, dev):
        from transformers import AutoTokenizer
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        cfg = MODELS[key];self.dev = dev
        tok = AutoTokenizer.from_pretrained(cfg['name'], revision=cfg['rev'])
        idx = json.loads(open(hf_hub_download(cfg['name'], 'model.safetensors.index.json', revision=cfg['rev'])).read())['weight_map']
        def get(name):
            f = hf_hub_download(cfg['name'], idx[name], revision=cfg['rev'])
            with safe_open(f, 'pt') as h:return h.get_tensor(name)
        W = get('lm_head.weight') if 'lm_head.weight' in idx else get('model.embed_tokens.weight')
        gamma = get('model.norm.weight').float().to(dev)
        keep, self.words = english_vocab(tok, W)
        n = min(W.shape[0], len(tok));wbar = W[:n].float().mean(0).to(dev);ids = torch.tensor(keep)
        ck = torch.load(hf_hub_download(LENS_REPO, cfg['lens'], revision=LENS_REV), map_location='cpu', weights_only=True)['J']
        self.D = {}
        for l in layers:
            J = ck[l].to(dev, torch.float32);rows = []
            for i in range(0, len(keep), 4096):
                a = ((W[ids[i:i+4096]].float().to(dev)-wbar)*gamma)@J;rows.append(a/a.norm(dim=-1, keepdim=True).clamp_min(1e-6))
            self.D[l] = torch.cat(rows)

    def recon(self, l, idx, coef):
        out = []
        for i in range(0, len(idx), 4096):
            ii = torch.tensor(idx[i:i+4096], device=self.dev).long();cc = torch.tensor(coef[i:i+4096], device=self.dev).float()
            out.append((cc[:, :, None]*self.D[l][ii]).sum(1).cpu().numpy())
        return np.concatenate(out)


def boot_ci(stat_fn, groups, n=NB):
    ug = np.unique(groups);idx = {g:np.where(groups == g)[0] for g in ug};vals = []
    for _ in range(n):
        s = RNG.choice(ug, len(ug));vals.append(stat_fn(np.concatenate([idx[g] for g in s])))
    vals = np.array(vals);return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)), vals


def holm(ps):
    o = np.argsort(ps);m = len(ps);adj = np.zeros(m);run = 0
    for r, i in enumerate(o):run = max(run, (m-r)*ps[i]);adj[i] = min(1., run)
    return adj.tolist()


# ------------------------------------------------------------------------------------------------------------------
def cosd(a, b):
    a = a/(np.linalg.norm(a, axis=-1, keepdims=True)+1e-9);b = b/(np.linalg.norm(b, axis=-1, keepdims=True)+1e-9)
    return 1-(a*b).sum(-1)


def binding_index(R, sid, bind, tmpl, dist='cos'):
    """Per scenario BI = [mean d(A_i,B_j) - mean d(same binding, other template)] / mean d(same, other template)."""
    out = {}
    for s in np.unique(sid):
        ii = np.where(sid == s)[0]
        if len(set(tmpl[ii])) < 2:continue
        cross, same = [], []
        for x in range(len(ii)):
            for y in range(x+1, len(ii)):
                a, b = ii[x], ii[y]
                if tmpl[a] == tmpl[b]:continue
                d = cosd(R[a], R[b]) if dist == 'cos' else 1-len(R[a]&R[b])/max(1, len(R[a]|R[b]))
                (cross if bind[a] != bind[b] else same).append(d)
        if cross and same and np.mean(same) > 0:out[s] = (np.mean(cross)-np.mean(same))/np.mean(same)
        elif cross and same:out[s] = 0.
    return out


def analyse_h1(root, D, layers, res):
    st = np.load(root/'h1'/'states.npz');meta = json.loads((root/'h1'/'meta.json').read_text())
    sid = np.array(meta['sid']);bind = np.array(meta['binding']);tmpl = np.array(meta['template']);sub = np.array(meta['sub'])
    split = np.array(meta['split']);lo = st['lo']
    out = dict(in_context_accuracy=float((lo > 0).mean()), by_relation={r:float((lo[sub == r] > 0).mean()) for r in np.unique(sub)})
    tr = split == 'train';ev = ~tr if tr.any() else np.ones(len(split), bool)
    rsa = {}
    for l in layers:
        H = st[f'H{l}'][:, 0].astype(np.float32);mu = H[tr].mean(0) if tr.any() else H.mean(0);X = H-mu
        idx = st[f'{l}_idx'][:, :25];J = D.recon(l, idx, st[f'{l}_coef25']);rem = X-J
        bag = [set(r.tolist()) for r in idx]
        fit = X[tr] if tr.any() else X
        U, S, Vt = np.linalg.svd(fit-fit.mean(0), full_matrices=False)
        reps = dict(residual=X, J25=J, remainder=rem, pca25=X@Vt[:25].T, pca64=X@Vt[:64].T,
                    rp25=X@np.linalg.qr(RNG.standard_normal((X.shape[1], 25)))[0], rp64=X@np.linalg.qr(RNG.standard_normal((X.shape[1], 64)))[0])
        rl = {}
        for name, R in list(reps.items())+[('bag', bag)]:
            per = {}
            for r in np.unique(sub):
                m = ev&(sub == r)
                if name == 'bag':bi = binding_index([bag[i] for i in np.where(m)[0]], sid[m], bind[m], tmpl[m], 'jac')
                else:bi = binding_index(R[m], sid[m], bind[m], tmpl[m])
                if not bi:continue
                v = np.array(list(bi.values()))
                # null: permute binding labels within scenario
                null = []
                for _ in range(200):
                    pb = bind[m].copy()
                    for s in np.unique(sid[m]):
                        j = np.where(sid[m] == s)[0];pb[j] = RNG.permutation(pb[j])
                    if name == 'bag':nb_ = binding_index([bag[i] for i in np.where(m)[0]], sid[m], pb, tmpl[m], 'jac')
                    else:nb_ = binding_index(R[m], sid[m], pb, tmpl[m])
                    null.append(np.mean(list(nb_.values())))
                per[r] = dict(BI=float(v.mean()), se=float(v.std()/math.sqrt(len(v))), null_mean=float(np.mean(null)),
                              null_95=float(np.percentile(null, 95)), p=float((np.sum(np.array(null) >= v.mean())+1)/(len(null)+1)), n=len(v))
            rl[name] = per
        rsa[l] = rl;log('h1 rsa layer', l, {k:{r:round(x['BI'], 3) for r, x in v.items()} for k, v in rl.items()})
    out['binding_index'] = rsa
    # context patch and single-position patches: normalized shift toward the source answer
    pz = np.load(root/'h1'/'patch.npz');tasks = pz['tasks'];conds = list(pz['conds'])
    a, b = tasks[:, 0], tasks[:, 1];g = sid[a]
    def shift(P):        # P (n, 2, L, C) log-odds of the target's correct answer
        full = lo[a]+lo[b]                                  # target own minus source-equivalent
        return (P[:, :, :, :1]-P)/np.where(np.abs(full) < 1e-3, np.nan, full)[:, :, None, None]
    cp = {}
    for name, P in (('context', pz['CP']), ('scenario_end', pz['SP'][..., 0]), ('decision', pz['SP'][..., 1])):
        sh = shift(P);d = {}
        for li, l in enumerate(layers):
            d[l] = {}
            for k, c in enumerate(conds):
                v = np.nanmean(sh[:, :, li, k], 1)
                lo_, hi_, _ = boot_ci(lambda ix:np.nanmean(v[ix]), g, n=500)
                d[l][c] = dict(shift=float(np.nanmean(v)), ci=[lo_, hi_], flip=float(np.nanmean(P[:, :, li, k] < 0)))
        cp[name] = d
        log('h1', name, {l:{c:round(x['shift'], 3) for c, x in v.items()} for l, v in d.items()})
    out['patch'] = cp;out['context_patch_positions_mean'] = float(pz['NP'].mean())
    tz = np.load(root/'h1'/'transplant.npz');TR = tz['TR'];tc = list(tz['conds'])
    out['transplant'] = {int(l):{c:float(np.nanmean(TR[:, :, li, k] > 0)) for k, c in enumerate(tc)} for li, l in enumerate(tz['layers'])}
    best_full = max(v['full'] for v in out['transplant'].values());out['transplant_qualified'] = bool(best_full >= .65)
    f3 = np.load(root/'h1'/'f3patch.npz');f3m = json.loads((root/'h1'/'f3meta.json').read_text())
    P3 = f3['P3'];t3 = f3['tasks'];truth = np.array([0 if (m['qkind'] == 'belief' and m['evidence'] == 'none') else 1 for m in f3m])
    # correct-answer sign: cands=[c1,c2]; belief/false-belief -> c1 (log-odds > 0), otherwise c2 (log-odds < 0)
    sgn = np.where(truth[t3[:, 0]] == 0, 1., -1.)
    out['f3_context_patch'] = {int(l):{c:dict(target_correct=float(np.nanmean(sgn*P3[:, li, k] > 0)),
                                             mean_signed=float(np.nanmean(sgn*P3[:, li, k]))) for k, c in enumerate(conds)}
                               for li, l in enumerate(layers)}
    res['H1'] = out


# ------------------------------------------------------------------------------------------------------------------
def ridge_fit(X, Y, alpha):
    mx, sx = X.mean(0), X.std(0)+1e-6;my = Y.mean(0);Xs = (X-mx)/sx
    A = Xs.T@Xs+alpha*np.eye(X.shape[1]);W = np.linalg.solve(A, Xs.T@(Y-my))
    return lambda Z:((Z-mx)/sx)@W+my


def within(M, groups):
    out = M.astype(np.float64).copy()
    for g in np.unique(groups):
        i = groups == g;out[i] -= out[i].mean(0)
    return out


def r2(Y, P):
    return 1-((Y-P)**2).sum(0)/((Y-Y.mean(0))**2).sum(0).clip(1e-9)


def fit_eval(Xtr, Ytr, Xva, Yva, Xte):
    best = None
    for a in (1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5):
        f = ridge_fit(Xtr, Ytr, a);s = r2(Yva, f(Xva)).mean()
        if best is None or s > best[0]:best = (s, a)
    f = ridge_fit(np.vstack([Xtr, Xva]), np.vstack([Ytr, Yva]), best[1])
    return f(Xte), best[1]


def analyse_h2(root, D, layers, res):
    out = {}
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
        tr, va, te = (split == 'train')&ok, (split == 'val')&ok, (split == 'test')&ok
        if not tr.any():tr = va = te = ok                                    # pilot: no splits, in-sample only
        Yw = within(Y, sid);Ow = within(O, sid)
        q = dict(n=int(ok.sum()), names=names,
                 within_var_frac=[float(Yw[tr, k].var()/max(Y[tr, k].var(), 1e-9)) for k in range(Y.shape[1])],
                 mean_disposition=[float(np.nanmean(Y[ok, k])) for k in range(Y.shape[1])])
        Po, _ = fit_eval(Ow[tr], Yw[tr], Ow[va], Yw[va], Ow[te])
        q['output_only_within_R2'] = r2(Yw[te], Po).tolist()
        # eligibility (qualification) is decided on train/validation only and frozen before any test score is used
        bestv = None
        for a_ in (1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5):
            sv = r2(Yw[va], ridge_fit(Ow[tr], Yw[tr], a_)(Ow[va]))
            if bestv is None or sv.mean() > bestv.mean():bestv = sv
        q['output_only_within_R2_val'] = bestv.tolist()
        eligible = [k for k in range(Y.shape[1]) if q['within_var_frac'][k] >= .2 and q['output_only_within_R2_val'][k] <= .7]
        q['qualified_targets'] = [names[k] for k in eligible];q['qualified'] = bool(eligible)
        Yres = Yw-0;rows = {}
        tests = []
        for l in layers:
            for pt_name in ('evid', 'dec'):
                j = 0 if pt_name == 'evid' else 1
                H = z[f'H{l}'][:, j].astype(np.float32);mu = H[tr].mean(0);X = H-mu
                reps = {}
                for K in (8, 16, 25, 32, 64):
                    reps[f'J{K}'] = D.recon(l, z[f'{l}_{pt_name}_idx'][:, :K], z[f'{l}_{pt_name}_coef{K}'])
                reps['residual'] = X;reps['remainder'] = X-reps['J25']
                U, Sv, Vt = np.linalg.svd(X[tr]-X[tr].mean(0), full_matrices=False)
                for k in (8, 16, 25, 32, 64):
                    reps[f'pca{k}'] = X@Vt[:k].T
                    reps[f'rp{k}'] = X@np.linalg.qr(RNG.standard_normal((X.shape[1], k)))[0]
                rr = {}
                for name, R in reps.items():
                    if R.shape[1] > 512:                                  # d-dim representations: PCA-compress on train to 512
                        Ur, Sr, Vr = np.linalg.svd(R[tr]-R[tr].mean(0), full_matrices=False);R = R@Vr[:512].T
                    Rw = within(R, sid);Xa = np.concatenate([Ow, Rw], 1)
                    P, alpha = fit_eval(Xa[tr], Yw[tr], Xa[va], Yw[va], Xa[te])
                    full = r2(Yw[te], P);base = np.array(q['output_only_within_R2'])
                    gain = full-base
                    rec_ = dict(R2_with_output=full.tolist(), partial_gain=gain.tolist(), alpha=alpha)
                    if eligible:
                        gte = sid[te];Yt_ = Yw[te][:, eligible];Pf = P[:, eligible];Pb = Po[:, eligible]
                        def stat(ix):
                            return float(np.mean(r2(Yt_[ix], Pf[ix])-r2(Yt_[ix], Pb[ix])))
                        lo_, hi_, vals = boot_ci(stat, gte, n=500)
                        rec_['gain_eligible_mean'] = stat(np.arange(len(gte)));rec_['ci'] = [lo_, hi_]
                        rec_['p_one_sided'] = float((np.sum(vals <= 0)+1)/(len(vals)+1))
                        if name == 'J25':tests.append((f'{l}_{pt_name}', rec_['p_one_sided']))
                    rr[name] = rec_
                # leave-one-challenge-out: disposition factors (rank 3) predicted from the representation are learned
                # without challenge c, then c is predicted on held-out scenarios through those factors
                def loco(F):
                    out_r2 = []
                    for c in range(Y.shape[1]):
                        keep = [j for j in range(Y.shape[1]) if j != c]
                        P, _ = fit_eval(F[tr], Yw[tr][:, keep], F[va], Yw[va][:, keep], F)
                        U_, S_, Vt_ = np.linalg.svd(P[tr]-P[tr].mean(0), full_matrices=False);Uf = (P-P[tr].mean(0))@Vt_[:3].T
                        w = np.linalg.lstsq(np.c_[Uf[tr], np.ones(tr.sum())], Yw[tr][:, c], rcond=None)[0]
                        pred = np.c_[Uf[te], np.ones(te.sum())]@w
                        out_r2.append(float(1-((Yw[te][:, c]-pred)**2).sum()/max(((Yw[te][:, c]-Yw[te][:, c].mean())**2).sum(), 1e-9)))
                    return out_r2
                Rj = reps['J25'];Uj, Sj, Vj = np.linalg.svd(Rj[tr]-Rj[tr].mean(0), full_matrices=False)
                Rres = X;Ur, Sr2, Vr2 = np.linalg.svd(Rres[tr]-Rres[tr].mean(0), full_matrices=False)
                lo_o = loco(Ow);lo_j = loco(np.concatenate([Ow, within(Rj@Vj[:512].T, sid)], 1))
                lo_r = loco(np.concatenate([Ow, within(Rres@Vr2[:512].T, sid)], 1))
                rr['_loco'] = dict(output=lo_o, J25=lo_j, residual=lo_r, mean_gain_J25=float(np.mean(np.array(lo_j)-np.array(lo_o))),
                                   mean_gain_residual=float(np.mean(np.array(lo_r)-np.array(lo_o))))
                rows[f'{l}_{pt_name}'] = rr
                log('h2', fam, l, pt_name, {k:round(v.get('gain_eligible_mean', np.nan), 3) for k, v in rr.items()
                                            if k in ('J25', 'residual', 'remainder', 'pca25', 'rp25')})
        if tests:
            adj = holm([p for _, p in tests]);q['primary_holm'] = {t:a for (t, _), a in zip(tests, adj)}
        q['by_representation'] = rows
        out[fam] = q
    res['H2'] = out


def analyse_h1full(root, res):
    f = root/'h1full'/'patch_full.npz'
    if not f.exists():return
    st = np.load(root/'h1'/'states.npz');meta = json.loads((root/'h1'/'meta.json').read_text());lo = st['lo'];sid = np.array(meta['sid'])
    z = np.load(f);SP = z['SP'];tasks = z['tasks'];conds = list(z['conds']);a, b = tasks[:, 0], tasks[:, 1]
    full = lo[a]+lo[b];sh = (SP[:, :, :, :1]-SP)/np.where(np.abs(full) < 1e-3, np.nan, full)[:, :, None, None];out = {}
    for li, l in enumerate(z['layers']):
        out[int(l)] = {}
        for k, c in enumerate(conds):
            v = np.nanmean(sh[:, :, li, k], 1);lo_, hi_, _ = boot_ci(lambda ix:np.nanmean(v[ix]), sid[a], n=500)
            out[int(l)][c] = dict(shift=float(np.nanmean(v)), ci=[lo_, hi_], correct_rate=float(np.nanmean(SP[:, :, li, k] > 0)))
    out['words_example'] = json.loads((root/'h1full'/'words_full.json').read_text()) if (root/'h1full'/'words_full.json').exists() else {}
    res['H1_full_dictionary'] = out;log('h1full', {k:{c:round(x['shift'], 3) for c, x in v.items()} for k, v in out.items() if isinstance(k, int)})


def analyse_epi(root, D, layers, res):
    """Exploratory (amendment 13): epistemic-concept loading of the decision-position workspace, by evidence type, its
    relation to confidence and to challenge responses, and the causal steering contrast vs random concept sets."""
    out = {}
    for fam in ('F2', 'F3'):
        f = root/'epi'/f'{fam}_epi.npz'
        if not f.exists():continue
        z = np.load(f);epi = set(z['epi'].tolist());LO = z['LO'];conds = list(z['conds']);sl = list(z['layers'])
        meta = json.loads((root/'epi'/f'{fam}_meta.json').read_text());sid = np.array([m['sid'] for m in meta])
        sg = np.array([1. if m['edir'] in (0, None) else -1. for m in meta])
        o = {}
        for li, l in enumerate(sl):
            base = np.abs(LO[:, li, conds.index('none0')]);rr = {}
            for c in conds:
                if c.startswith('none'):continue
                d = np.abs(LO[:, li, conds.index(c)])-base
                lo_, hi_, _ = boot_ci(lambda ix:np.nanmean(d[ix]), sid, n=500);rr[c] = dict(delta_abs_logodds=float(np.nanmean(d)), ci=[lo_, hi_])
            for a in ('1', '-1'):
                d = (np.abs(LO[:, li, conds.index('epi'+a)])-np.abs(LO[:, li, conds.index('rand'+a)]))
                lo_, hi_, vals = boot_ci(lambda ix:np.nanmean(d[ix]), sid, n=1000)
                rr[f'epi{a}_minus_rand{a}'] = dict(mean=float(np.nanmean(d)), ci=[lo_, hi_], p_two_sided=float(2*min(np.mean(vals <= 0), np.mean(vals >= 0))))
            o[int(l)] = rr
        # descriptive loading from the saved H2 decision codes (all splits)
        h2 = np.load(root/'h2'/fam/'states.npz');m2 = json.loads((root/'h2'/fam/'meta.json').read_text())
        ev = np.array([s['evidence'] for s in m2['states']]);lo0 = h2['lo0']
        LCd = (h2['LC']-lo0[:, None])*np.where(h2['cdir'] == 0, 1., -1.);names = m2['chal_names']
        for l in sl:
            idx = h2[f'{l}_dec_idx'][:, :25];c = h2[f'{l}_dec_coef25']
            ind = np.array([c[i][[j in epi for j in idx[i]]].sum()/max(c[i].sum(), 1e-9) for i in range(len(c))])
            o[f'{l}_loading_by_evidence'] = {e:float(ind[ev == e].mean()) for e in np.unique(ev)}
            o[f'{l}_corr_with_abs_logodds'] = float(np.corrcoef(ind, np.abs(lo0))[0, 1])
            o[f'{l}_corr_with_challenge_response'] = {n:float(np.corrcoef(ind[np.isfinite(LCd[:, k])], LCd[np.isfinite(LCd[:, k]), k])[0, 1]) for k, n in enumerate(names)}
        out[fam] = o
        log('epi', fam, {k:v for k, v in o.items() if isinstance(k, int)})
    res['epistemic'] = out


def analyse_think(root, res):
    f = root/'think'/'think.json'
    if not f.exists():return
    t = json.loads(f.read_text())
    res['think'] = dict(n=len(t), closed=float(np.mean([x['closed'] for x in t])), tokens=float(np.mean([x['n_think_tokens'] for x in t])),
                        examples=[dict(fam=x['family'], ev=x['evidence'], lo=x['logodds_think'], L25=[w for w, c in x['codes'].get('25', x['codes'].get(25, []))[:12]]) for x in t[:8]])


def main():
    p = argparse.ArgumentParser();p.add_argument('--model', default='qwen3-8b');p.add_argument('--tag', default='main')
    p.add_argument('--parts', default='h1,h2,think,epi,h1full')
    a = p.parse_args();root = OUT/'e1'/a.model/a.tag;layers = MODELS[a.model]['layers']
    dev = 'cuda' if torch.cuda.is_available() else 'cpu';D = Dict(a.model, layers, dev)
    res = json.loads((root/'results_e1.json').read_text()) if (root/'results_e1.json').exists() else {}
    parts = a.parts.split(',')
    if 'h1' in parts:analyse_h1(root, D, layers, res)
    if 'h2' in parts:analyse_h2(root, D, layers, res)
    if 'think' in parts:analyse_think(root, res)
    if 'epi' in parts:analyse_epi(root, D, layers, res)
    if 'h1full' in parts:analyse_h1full(root, res)
    atomic_json(root/'results_e1.json', res);log('written', root/'results_e1.json')


if __name__ == '__main__':
    main()
