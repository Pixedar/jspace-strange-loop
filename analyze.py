"""Analysis of a strange_loop.py run: the J-space path, the self-models, and the tests asked of the loop.

    python analyze.py runs/mainA            -> runs/mainA/results.json (numbers + what the report draws)

All similarities are cosines between J-space content vectors measured relative to the typical workspace (so two
unrelated thoughts score ~0). Statistics are over seeds (each seed = one independent mind per condition); paired
comparisons use the same seed in both conditions (they share their sampling noise and are identical until treated).
"""
import json, sys
import numpy as np

d = sys.argv[1]
Z = np.load(f'{d}/traj.npz');M = json.load(open(f'{d}/meta.json', encoding='utf-8'))
rows, a = M['rows'], M['args'];K, T_INT = a['K'], a['t_int'];B = len(rows)
INDIVIDUATED = any(r['cond'] in ('loopc', 'yokedc', 'clampc') for r in rows)
for r in rows:r['cond'] = {'loopc':'loop', 'yokedc':'yoked', 'clampc':'clamp'}.get(r['cond'], r['cond'])
words = np.array(M['words'])
C = Z['j'].astype(np.float32);N = C.shape[0]                             # (N, B, d) thought contents
MS = Z['m'].astype(np.float32);NR = MS.shape[0]                           # (NR, B, d) self-models
INJ = Z['inj'].astype(np.float32);SRC = Z['inj_src']
R_INT = T_INT//K-1                                                        # reflection whose output perturb replaces
conds = list(dict.fromkeys(r['cond'] for r in rows));seeds = sorted({r['seed'] for r in rows})
FREE = [b for b, r in enumerate(rows) if r['cond'] == 'free']
row = {(r['cond'], r['seed']):b for b, r in enumerate(rows)}
unit = lambda x:x/np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
cos = lambda x, y:(unit(x)*unit(y)).sum(-1)
alpha = a['alpha']
Zs = np.zeros_like(C);s = np.zeros_like(C[0])
for t in range(N):
    s = (1-alpha)*s+alpha*C[t];Zs[t] = s                                 # recurrent state z_t (after thought t)
WIN = np.stack([C[r*K:(r+1)*K].mean(0) for r in range(NR)])               # mean content of each window
Cu, MSu = unit(C), unit(MS)


def ms(v):
    v = np.asarray(v, float);n = len(v)
    return dict(mean=float(v.mean()), se=float(v.std(ddof=1)/np.sqrt(n)) if n > 1 else 0., n=n)


def paired(x, y):
    """x - y over seeds: mean, se, t, and how many seeds go the same way."""
    d_ = np.asarray(x, float)-np.asarray(y, float);n = len(d_)
    se = d_.std(ddof=1)/np.sqrt(n) if n > 1 else 0.
    return dict(diff=float(d_.mean()), se=float(se), t=float(d_.mean()/se) if se > 0 else 0., pos=int((d_ > 0).sum()), n=n)


out = dict(args=a, conds=conds, seeds=seeds, N=N, NR=NR, K=K, t_int=T_INT, rc=M['rc'], Rn=M['Rn'])

# ------------------------------------------------------------------ 1. wandering: autocorrelation of the J-space path
lags = list(range(1, 61))
def acf(b, t0, t1, proj=None):
    x = C[t0:t1, b].copy()
    if proj is not None:                                                  # remove the injected self-model direction
        p = unit(proj[t0:t1]);x = x-(x*p).sum(-1, keepdims=True)*p
    xu = unit(x)
    return [float((xu[:-L]*xu[L:]).sum(-1).mean()) if L < len(xu) else np.nan for L in lags]

inj_at = np.zeros_like(C)                                                 # the vector injected during thought t
for r in range(NR-1):inj_at[(r+1)*K:(r+2)*K] = INJ[r]
out['acf'] = {}
for c in conds:
    for ph, (t0, t1) in {'pre':(20, T_INT), 'post':(T_INT, N)}.items():
        A = np.array([acf(row[c, s_], t0, t1) for s_ in seeds])
        Ap = np.array([acf(row[c, s_], t0, t1, inj_at[:, row[c, s_]]) for s_ in seeds])
        ch = np.mean([cos(C[t0:t1, row[c, s1]], C[t0:t1, row[c, s2]]).mean() for s1 in seeds for s2 in seeds if s1 < s2])
        out['acf'][f'{c}/{ph}'] = dict(mean=A.mean(0).tolist(), se=(A.std(0, ddof=1)/np.sqrt(len(seeds))).tolist(),
                                       proj_mean=Ap.mean(0).tolist(), chance=float(ch),
                                       long=A[:, 19:60].mean(1).tolist(), long_proj=Ap[:, 19:60].mean(1).tolist(),
                                       short=A[:, 0:5].mean(1).tolist())
out['lags'] = lags
# over time: each mind's coherence with itself 20 thoughts earlier, and how alike different minds are at the same moment
out['coh_time'], out['cross_time'] = {}, {}
for c in conds:
    bs = [row[c, s_] for s_ in seeds]
    out['coh_time'][c] = np.round(np.mean([cos(C[20:, b], C[:-20, b]) for b in bs], 0), 4).tolist()
    out['cross_time'][c] = np.round(np.mean([cos(C[:, b1], C[:, b2]) for b1 in bs for b2 in bs if b1 < b2], 0), 4).tolist()

# ------------------------------------------------------------------ 2. the self-models: a persistent self-symbol?
sm = {}
for c in conds:
    bs = [row[c, s_] for s_ in seeds]
    cons = np.array([[cos(MS[k, b], MS[k+1, b]) for k in range(NR-1)] for b in bs])          # (S, NR-1)
    wcons = np.array([[cos(WIN[k, b], WIN[k+1, b]) for k in range(NR-1)] for b in bs])       # content's own
    lag = {L:np.array([[cos(MS[k, b], MS[k+L, b]) for k in range(NR-L)] for b in bs]).mean(1) for L in (1, 2, 5, 10)}
    wlag = {L:np.array([[cos(WIN[k, b], WIN[k+L, b]) for k in range(NR-L)] for b in bs]).mean(1) for L in (1, 2, 5, 10)}
    cross = np.array([np.mean([cos(MS[k, b1], MS[k, b2]) for b1 in bs for b2 in bs if b1 < b2]) for k in range(NR)])
    rec = np.array([MSu[:, b]@MSu[:, b].T for b in bs]).mean(0)                              # (NR, NR) recurrence
    # individuality: is a row's early self-model closer to its own later self-symbol than to other rows'?
    # (both inside the pre-intervention period, so all conditions are measured alike)
    half = R_INT//2
    sym = unit(np.stack([MS[half:R_INT, b].mean(0) for b in bs]))
    # reconstruction: does the next self-model re-form what was fed in (own in loop, foreign in yoked)?
    recon = [np.mean([cos(MS[k+1, b], INJ[k, b]) for k in range(R_INT-1) if SRC[k, b] != -1]) if c != 'free' else np.nan
             for b in bs]
    generic = unit(MS[:R_INT].reshape(-1, MS.shape[-1]).mean(0))
    gen_share = [float(np.mean(cos(MS[:R_INT, b], generic))) for b in bs]
    own_sym = [np.mean([cos(MS[k, b], sym[i]) for k in range(half)]) for i, b in enumerate(bs)]
    oth_sym = [np.mean([cos(MS[k, b], sym[j]) for k in range(half) for j in range(len(bs)) if j != i]) for i, b in enumerate(bs)]
    sm[c] = dict(cons_time=cons.mean(0).tolist(), cons_time_se=(cons.std(0, ddof=1)/np.sqrt(len(bs))).tolist(),
                 wcons_time=wcons.mean(0).tolist(), cross_time=cross.tolist(),
                 cons_pre=cons[:, 2:R_INT].mean(1).tolist(), wcons_pre=wcons[:, 2:R_INT].mean(1).tolist(),
                 cons_post=cons[:, R_INT+1:].mean(1).tolist(), wcons_post=wcons[:, R_INT+1:].mean(1).tolist(),
                 lag={L:v.tolist() for L, v in lag.items()}, wlag={L:v.tolist() for L, v in wlag.items()},
                 cross_pre=float(cross[2:R_INT].mean()), recurrence=np.round(rec, 3).tolist(),
                 own_symbol=own_sym, other_symbol=oth_sym, recon=[float(x) for x in recon], generic_share=gen_share)
out['selfmodel'] = sm

# individuated self-models: what sets one mind's self-model apart from the population's generic one (the mean over
# the observer rows at the same reflection, as the individuated conditions of run C feed back)
GEN = MS[:, FREE].mean(1);MSI = MS-GEN[:, None]
out['generic_norm_share'] = float(np.linalg.norm(GEN, axis=-1).mean()/np.linalg.norm(MS, axis=-1).mean())
for c in conds:
    bs = [row[c, s_] for s_ in seeds]
    sm[c]['cons_ind_pre'] = [float(np.mean([cos(MSI[k, b], MSI[k+1, b]) for k in range(2, R_INT)])) for b in bs]
    sm[c]['cross_ind_pre'] = float(np.mean([cos(MSI[k, b1], MSI[k, b2]) for k in range(2, R_INT) for b1 in bs for b2 in bs if b1 < b2]))
    sm[c]['recon_ind'] = [float(np.mean([cos(MSI[k+1, b], MSI[k, SRC[k, b]]) for k in range(R_INT-1) if SRC[k, b] >= 0]))
                          if c not in ('free', 'avg') else np.nan for b in bs]
    sm[c]['cons_ind_time'] = np.array([[cos(MSI[k, b], MSI[k+1, b]) for k in range(NR-1)] for b in bs]).mean(0).tolist()
out.setdefault('selfmodel_tests', {})
if 'yoked' in conds:
    out['selfmodel_tests']['loop_vs_yoked_cons_ind'] = paired(sm['loop']['cons_ind_pre'], sm['yoked']['cons_ind_pre'])
    out['selfmodel_tests']['loop_vs_yoked_recon_ind'] = paired(sm['loop']['recon_ind'], sm['yoked']['recon_ind'])
    out['selfmodel_tests']['loop_vs_free_cons_ind'] = paired(sm['loop']['cons_ind_pre'], sm['free']['cons_ind_pre'])
    out['selfmodel_tests']['yoked_vs_free_cons_ind'] = paired(sm['yoked']['cons_ind_pre'], sm['free']['cons_ind_pre'])

# self-prediction: the forward question ("in the next few moments, my mind will probably turn to")
if 'mf' in Z.files:
    MF = Z['mf'].astype(np.float32);fw = {}
    for c in conds:
        bs = [row[c, s_] for s_ in seeds];rr = [r for r in range(1, NR-1) if not (c in ('perturb', 'ablate', 'clamp') and r >= R_INT)]
        res = {k:[] for k in ('fwd', 'back', 'fwd_other', 'fwd_perp', 'back_perp', 'fwd_other_perp', 'fwd_lean', 'back_lean', 'fwd_back_sim')}
        for b in bs:
            acc = {k:[] for k in res}
            for r in rr:
                F, P = WIN[r+1, b], WIN[r, b];Fp = F-(F@unit(P))*unit(P);oth = [b2 for b2 in bs if b2 != b]
                acc['fwd'].append(cos(MF[r, b], F));acc['back'].append(cos(MS[r, b], F))
                acc['fwd_other'].append(np.mean([cos(MF[r, b2], F) for b2 in oth]))
                acc['fwd_perp'].append(cos(MF[r, b], Fp));acc['back_perp'].append(cos(MS[r, b], Fp))
                acc['fwd_other_perp'].append(np.mean([cos(MF[r, b2], Fp) for b2 in oth]))
                acc['fwd_lean'].append(cos(MF[r, b], F)-cos(MF[r, b], P));acc['back_lean'].append(cos(MS[r, b], F)-cos(MS[r, b], P))
                acc['fwd_back_sim'].append(cos(MF[r, b], MS[r, b]))
            for k in res:res[k].append(float(np.mean(acc[k])))
        fw[c] = res
    out['forward'] = fw
    out['forward_tests'] = {c:dict(fwd_vs_other=paired(fw[c]['fwd'], fw[c]['fwd_other']),
                                   fwd_vs_back=paired(fw[c]['fwd'], fw[c]['back']),
                                   perp_fwd_vs_other=paired(fw[c]['fwd_perp'], fw[c]['fwd_other_perp']),
                                   perp_fwd_vs_back=paired(fw[c]['fwd_perp'], fw[c]['back_perp']),
                                   lean_fwd_vs_back=paired(fw[c]['fwd_lean'], fw[c]['back_lean'])) for c in conds}
    out['forward_answers'] = [[M['fdesc'][r][b].strip()[:120] for r in range(NR)] for b in range(B)]

# ------------------------------------------------------------------ 3. prediction: does the self-model predict the future?
# For reflection r (made after window r), the future is the next window's mean content F = WIN[r+1].
pred = {}
for c in conds:
    bs = [row[c, s_] for s_ in seeds];res = {k:[] for k in ('own', 'other', 'past', 'inj', 'own_perp', 'other_perp',
                                                              'inj_perp', 'own_retr', 'own_past', 'own_fut_minus_past')}
    rr = [r for r in range(1, NR-1) if not (c in ('perturb', 'ablate') and r >= R_INT)]
    for i, b in enumerate(bs):
        o, ot, p, inj, op, otp, ip, rt, opst, ofmp = ([] for _ in range(10))
        for r in rr:
            F, P = WIN[r+1, b], WIN[r, b]
            Fp = F-(F@unit(P))*unit(P)                                     # the part of the future not in the past
            o.append(cos(MS[r, b], F));p.append(cos(P, F));opst.append(cos(MS[r, b], P))
            others = [b2 for b2 in bs if b2 != b]
            ot.append(np.mean([cos(MS[r, b2], F) for b2 in others]))
            op.append(cos(MS[r, b], Fp));otp.append(np.mean([cos(MS[r, b2], Fp) for b2 in others]))
            if SRC[r, b] != -1:inj.append(cos(INJ[r, b], F));ip.append(cos(INJ[r, b], Fp))
            sc = {b2:cos(MS[r, b], WIN[r+1, b2]) for b2 in bs}             # retrieval of one's own future
            rt.append(float(max(sc, key=sc.get) == b))
            ofmp.append(cos(MS[r, b], F)-cos(MS[r, b], P))
        for k, v in zip(('own', 'other', 'past', 'own_perp', 'other_perp', 'own_retr', 'own_past', 'own_fut_minus_past'),
                        (o, ot, p, op, otp, rt, opst, ofmp)):res[k].append(float(np.mean(v)))
        if inj:res['inj'].append(float(np.mean(inj)));res['inj_perp'].append(float(np.mean(ip)))
    pred[c] = res
out['prediction'] = pred
out['prediction_tests'] = {c:dict(own_vs_other=paired(pred[c]['own'], pred[c]['other']),
                                  perp_own_vs_other=paired(pred[c]['own_perp'], pred[c]['other_perp']),
                                  retrieval=ms(pred[c]['own_retr']),
                                  future_vs_past=paired(pred[c]['own'], pred[c]['own_past']))
                           for c in conds}
if 'yoked' in conds:
    out['prediction_tests']['yoked_own_vs_injected'] = paired(pred['yoked']['own'], pred['yoked']['inj'])
    out['prediction_tests']['yoked_own_vs_injected_perp'] = paired(pred['yoked']['own_perp'], pred['yoked']['inj_perp'])
if 'avg' in conds:
    out['prediction_tests']['avg_own_vs_injected'] = paired(pred['avg']['own'], pred['avg']['inj'])

# fidelity: does the self-model match its own past window better than the other rows' windows?
fid = {}
for name in ['m']+[k for k in Z.files if k.startswith('fid_')]:
    Mx = Z[name].astype(np.float32);own, oth, hit = [], [], []
    for c in conds:
        bs = [row[c, s_] for s_ in seeds]
        for r in range(NR):
            for b in bs:
                sc = {b2:cos(Mx[r, b], WIN[r, b2]) for b2 in bs}
                own.append(sc[b]);oth.append(np.mean([v for k, v in sc.items() if k != b]));hit.append(max(sc, key=sc.get) == b)
    fid[name] = dict(own=float(np.mean(own)), other=float(np.mean(oth)), retrieval=float(np.mean(hit)), chance=1/len(seeds))
out['fidelity'] = fid

# ------------------------------------------------------------------ 4. perturbation: replace the self-model once
if 'perturb' in conds:
    per = dict(traj_to_foreign=[], twin_traj_to_foreign=[], twin_div=[], self_to_old=[], self_to_foreign=[],
               twin_self_to_old=[], twin_self_to_foreign=[])
    for s_ in seeds:
        bp, bl = row['perturb', s_], row['loop', s_]
        mstar = INJ[R_INT, bp];mold = MS[R_INT, bl]
        per['traj_to_foreign'].append(cos(C[T_INT-40:N, bp], mstar).tolist())
        per['twin_traj_to_foreign'].append(cos(C[T_INT-40:N, bl], mstar).tolist())
        per['twin_div'].append(cos(C[T_INT-40:N, bp], C[T_INT-40:N, bl]).tolist())
        per['self_to_old'].append([cos(MS[k, bp], mold) for k in range(R_INT-4, NR)])
        per['self_to_foreign'].append([cos(MS[k, bp], mstar) for k in range(R_INT-4, NR)])
        per['twin_self_to_old'].append([cos(MS[k, bl], mold) for k in range(R_INT-4, NR)])
        per['twin_self_to_foreign'].append([cos(MS[k, bl], mstar) for k in range(R_INT-4, NR)])
    agg = {k:dict(mean=np.mean(v, 0).tolist(), se=(np.std(v, 0, ddof=1)/np.sqrt(len(seeds))).tolist()) for k, v in per.items()}
    sl = lambda v, i0, i1:np.array(v)[:, i0:i1].mean(1)
    agg['tests'] = dict(
        traj_reorganises=paired(sl(per['traj_to_foreign'], 40, 40+K*3), sl(per['twin_traj_to_foreign'], 40, 40+K*3)),
        traj_late=paired(sl(per['traj_to_foreign'], 40+K*5, None), sl(per['twin_traj_to_foreign'], 40+K*5, None)),
        self_next_to_foreign=paired(sl(per['self_to_foreign'], 5, 6), sl(per['twin_self_to_foreign'], 5, 6)),
        self_late_to_foreign=paired(sl(per['self_to_foreign'], 8, None), sl(per['twin_self_to_foreign'], 8, None)),
        self_next_to_old=paired(sl(per['self_to_old'], 5, 6), sl(per['twin_self_to_old'], 5, 6)),
        self_late_to_old=paired(sl(per['self_to_old'], 8, None), sl(per['twin_self_to_old'], 8, None)))
    agg['t0'] = T_INT-40;agg['r0'] = R_INT-4
    out['perturb'] = agg

# ------------------------------------------------------------------ 5. ablation: remove the self-model
if 'ablate' in conds:
    ab = dict(div=[]);L = lambda c, ph:np.array(out['acf'][f'{c}/{ph}']['long'])
    for s_ in seeds:ab['div'].append(cos(C[T_INT-40:N, row['ablate', s_]], C[T_INT-40:N, row['loop', s_]]).tolist())
    ab['div'] = dict(mean=np.mean(ab['div'], 0).tolist(), se=(np.std(ab['div'], 0, ddof=1)/np.sqrt(len(seeds))).tolist())
    ab['tests'] = dict(long_loop_vs_ablate=paired(L('loop', 'post'), L('ablate', 'post')),
                       long_ablate_post_vs_pre=paired(L('ablate', 'post'), L('ablate', 'pre')),
                       long_loop_post_vs_pre=paired(L('loop', 'post'), L('loop', 'pre')),
                       long_ablate_vs_free_post=paired(L('ablate', 'post'), L('free', 'post')))
    out['ablate'] = ab
out['coherence_tests'] = {f'{c}_vs_free':paired(out['acf'][f'{c}/pre']['long'], out['acf']['free/pre']['long'])
                          for c in conds if c != 'free'}
out['coherence_tests'].update({f'{c}_vs_free_proj':paired(out['acf'][f'{c}/pre']['long_proj'], out['acf']['free/pre']['long_proj'])
                               for c in conds if c != 'free'})
if 'yoked' in conds:
    out['coherence_tests']['loop_vs_yoked'] = paired(out['acf']['loop/pre']['long'], out['acf']['yoked/pre']['long'])
    out['coherence_tests']['loop_vs_yoked_proj'] = paired(out['acf']['loop/pre']['long_proj'], out['acf']['yoked/pre']['long_proj'])
out.setdefault('selfmodel_tests', {})
for c in conds:
    out['selfmodel_tests'][f'{c}_self_vs_content_stability'] = paired(sm[c]['cons_pre'], sm[c]['wcons_pre'])
    out['selfmodel_tests'][f'{c}_own_vs_other_symbol'] = paired(sm[c]['own_symbol'], sm[c]['other_symbol'])
    if c != 'free':out['selfmodel_tests'][f'{c}_vs_free_cons'] = paired(sm[c]['cons_pre'], sm['free']['cons_pre'])
if 'yoked' in conds:
    out['selfmodel_tests']['loop_vs_yoked_cons'] = paired(sm['loop']['cons_pre'], sm['yoked']['cons_pre'])
    out['selfmodel_tests']['loop_vs_yoked_recon'] = paired(sm['loop']['recon'], sm['yoked']['recon'])
    out['selfmodel_tests']['yoked_recon_vs_own_cons'] = paired(sm['yoked']['recon'], sm['yoked']['cons_pre'])
if 'avg' in conds:out['selfmodel_tests']['loop_vs_avg_recon'] = paired(sm['loop']['recon'], sm['avg']['recon'])

# ------------------------------------------------------------------ 4b. clamp and release (run C)
if 'clamp' in conds:
    CL = a.get('clamp_len', 50)//K;cl = dict(self_to_old=[], self_to_foreign=[], twin_self_to_old=[], twin_self_to_foreign=[],
                                             traj_to_foreign=[], twin_traj_to_foreign=[], twin_div=[])
    for s_ in seeds:
        bc, bl = row['clamp', s_], row['loop', s_]
        fvec = MSI[R_INT, SRC[R_INT, bc]];old = MSI[R_INT, bl]               # the clamped foreign self; the own self then
        for k_, (bb, key) in enumerate([(bc, ''), (bl, 'twin_')]):
            cl[key+'self_to_old'].append([cos(MSI[k, bb], old) for k in range(R_INT-4, NR)])
            cl[key+'self_to_foreign'].append([cos(MSI[k, bb], fvec) for k in range(R_INT-4, NR)])
            cl[key+'traj_to_foreign'].append(cos(C[T_INT-40:N, bb], fvec).tolist())
        cl['twin_div'].append(cos(C[T_INT-40:N, bc], C[T_INT-40:N, bl]).tolist())
    agg = {k:dict(mean=np.mean(v, 0).tolist(), se=(np.std(v, 0, ddof=1)/np.sqrt(len(seeds))).tolist()) for k, v in cl.items()}
    sl = lambda v, i0, i1:np.array(v)[:, i0:i1].mean(1)
    i_rel = 4+CL                                                          # first reflection after release (relative)
    agg['tests'] = dict(
        during_traj_to_foreign=paired(sl(cl['traj_to_foreign'], 40, 40+CL*K), sl(cl['twin_traj_to_foreign'], 40, 40+CL*K)),
        during_self_to_foreign=paired(sl(cl['self_to_foreign'], 5, 4+CL), sl(cl['twin_self_to_foreign'], 5, 4+CL)),
        during_self_to_old=paired(sl(cl['self_to_old'], 5, 4+CL), sl(cl['twin_self_to_old'], 5, 4+CL)),
        after_self_to_foreign=paired(sl(cl['self_to_foreign'], i_rel+1, i_rel+6), sl(cl['twin_self_to_foreign'], i_rel+1, i_rel+6)),
        after_self_to_old=paired(sl(cl['self_to_old'], i_rel+1, i_rel+6), sl(cl['twin_self_to_old'], i_rel+1, i_rel+6)),
        late_self_to_foreign=paired(sl(cl['self_to_foreign'], i_rel+6, None), sl(cl['twin_self_to_foreign'], i_rel+6, None)),
        after_traj_to_foreign=paired(sl(cl['traj_to_foreign'], 40+CL*K, 40+CL*K+50), sl(cl['twin_traj_to_foreign'], 40+CL*K, 40+CL*K+50)))
    agg['t0'] = T_INT-40;agg['r0'] = R_INT-4;agg['release_r'] = R_INT+CL
    out['clamp'] = agg

# ------------------------------------------------------------------ 6. what the report draws: the path through J-space
X = C.reshape(-1, C.shape[-1]);mu = X.mean(0)
U, S_, Vt = np.linalg.svd((X-mu)[::3], full_matrices=False)
P = Vt[:3];out['pca_explained'] = (S_[:3]**2/(S_**2).sum()).tolist()
proj = lambda v:((v-mu)@P.T)
out['path'] = np.round(proj(C), 3).transpose(1, 0, 2).tolist()             # (B, N, 3)
out['path_z'] = np.round(proj(Zs), 3).transpose(1, 0, 2).tolist()
out['path_m'] = np.round(proj(MS), 3).transpose(1, 0, 2).tolist()          # (B, NR, 3)

# the words that make a thought distinctive: its J-space coefficients minus each word's average coefficient
def wordvec(idx, coef):
    W = {}
    for i, c_ in zip(idx, coef):W[int(i)] = W.get(int(i), 0.)+float(c_)
    return W
def mean_words(I, Cf):
    acc = {};n = I.shape[0]                                                # per thought (I: (n_thoughts, k))
    for i, c_ in zip(I.reshape(-1), Cf.reshape(-1).astype(float)):acc[int(i)] = acc.get(int(i), 0.)+c_
    return {k:v/n for k, v in acc.items()}
def distinct(I, Cf, base, n=5):
    W = wordvec(I, Cf);sc = sorted(((v-base.get(k, 0.)*1.0, k) for k, v in W.items()), reverse=True)
    return [str(words[k]) for v, k in sc[:n] if v > 0]
jb = mean_words(Z['j_idx'].reshape(N*B, -1), Z['j_coef'].reshape(N*B, -1).astype(np.float32))
mb = mean_words(Z['m_idx'].reshape(NR*B, -1), Z['m_coef'].reshape(NR*B, -1).astype(np.float32))
out['thought_words'] = [[distinct(Z['j_idx'][t, b], Z['j_coef'][t, b], jb) for t in range(N)] for b in range(B)]
out['self_words'] = [[distinct(Z['m_idx'][r, b], Z['m_coef'][r, b], mb, 8) for r in range(NR)] for b in range(B)]
out['texts'] = [[M['texts'][t][b].strip()[:220] for t in range(N)] for b in range(B)]
out['answers'] = [[M['desc'][r][b].strip()[:200] for r in range(NR)] for b in range(B)]
out['rows'] = rows;out['inj_src'] = SRC.T.tolist()

# themes: k-means on the contents, named by their members' distinctive words
rng = np.random.default_rng(0);Xu = unit(X);kk = 14
cent = Xu[rng.choice(len(Xu), kk, replace=False)]
for _ in range(30):
    lab = (Xu@cent.T).argmax(1)
    cent = unit(np.stack([Xu[lab == i].mean(0) if (lab == i).any() else cent[i] for i in range(kk)]))
lab = (Xu@cent.T).argmax(1)
themes = []
JI, JC = Z['j_idx'].reshape(N*B, -1), Z['j_coef'].reshape(N*B, -1).astype(np.float32)
for i in range(kk):
    sel = np.where(lab == i)[0]
    acc = {}
    for q in sel:
        for w_, c_ in zip(JI[q], JC[q]):acc[int(w_)] = acc.get(int(w_), 0.)+float(c_)
    sc = sorted(((v/len(sel)-jb.get(k, 0.), k) for k, v in acc.items()), reverse=True)
    themes.append(dict(words=[str(words[k]) for v, k in sc[:6]], n=int(len(sel)),
                       share={c:float(np.mean([lab.reshape(N, B)[:, row[c, s_]] == i for s_ in seeds])) for c in conds}))
out['themes'] = themes;out['theme_of'] = lab.reshape(N, B).T.tolist()

with open(f'{d}/results.json', 'w', encoding='utf-8') as f:json.dump(out, f, ensure_ascii=False, default=lambda o:o.item() if hasattr(o, 'item') else str(o))
# --------------------------------------------------------------- console summary
print('PCA explained', np.round(out['pca_explained'], 3))
print('\nwandering (mean cos at lags 1,5,20,60 | chance)')
for k, v in out['acf'].items():print(f'  {k:14s}', ' '.join(f"{v['mean'][L-1]:.2f}" for L in (1, 5, 20, 60)), f"| {v['chance']:.2f}")
print('\nfidelity', {k:(round(v['own'], 3), round(v['other'], 3), round(v['retrieval'], 2)) for k, v in fid.items()})
print('\nself-model stability (consecutive cos, pre) vs content windows; cross-seed')
for c in conds:print(f"  {c:8s} self {np.mean(sm[c]['cons_pre']):.3f}  content {np.mean(sm[c]['wcons_pre']):.3f}  cross {sm[c]['cross_pre']:.3f}"
                     f"  own-symbol {np.mean(sm[c]['own_symbol']):.3f} vs others {np.mean(sm[c]['other_symbol']):.3f}"
                     f"  recon {np.mean(sm[c]['recon']):.3f}  generic {np.mean(sm[c]['generic_share']):.3f}")
print('\nprediction of the next window: own self-model | other row\'s | past window | injected  (|| beyond past: own, other)')
for c in conds:
    p_ = pred[c];print(f"  {c:8s} {np.mean(p_['own']):.3f} | {np.mean(p_['other']):.3f} | {np.mean(p_['past']):.3f} | "
                       f"{np.mean(p_['inj']) if p_['inj'] else float('nan'):.3f}  || {np.mean(p_['own_perp']):.3f}, {np.mean(p_['other_perp']):.3f}"
                       f"  retrieval {np.mean(p_['own_retr']):.2f}")
for grp in ('prediction_tests', 'coherence_tests', 'selfmodel_tests'):
    print(f'\n{grp}')
    for k, v in out[grp].items():
        if 'own_vs_other' in v:
            print(f"  {k:40s} " + '  '.join(f"{kk} {vv.get('diff', vv.get('mean')):+.3f}±{vv['se']:.3f}" for kk, vv in v.items()))
        elif 'diff' in v:print(f"  {k:40s} diff {v['diff']:+.3f} ± {v['se']:.3f}  t {v['t']:+.1f}  ({v['pos']}/{v['n']} seeds >0)")
        else:print(f"  {k:40s} {v}")
print('\ngeneric share of self-model norm', round(out['generic_norm_share'], 3))
for c in conds:print(f"  {c:8s} individuated: consecutive {np.mean(sm[c]['cons_ind_pre']):.3f} cross-seed {sm[c]['cross_ind_pre']:.3f}"
                     f" recon {np.nanmean(sm[c]['recon_ind']) if c not in ('free', 'avg') else float('nan'):.3f}")
if 'forward' in out:
    print('\nself-prediction (forward question): cos with next window  fwd | back | fwd other || beyond past fwd | back | other || lean fwd | back')
    for c in conds:
        f_ = out['forward'][c];print(f"  {c:8s} {np.mean(f_['fwd']):.3f} | {np.mean(f_['back']):.3f} | {np.mean(f_['fwd_other']):.3f} || "
                                     f"{np.mean(f_['fwd_perp']):.3f} | {np.mean(f_['back_perp']):.3f} | {np.mean(f_['fwd_other_perp']):.3f} || "
                                     f"{np.mean(f_['fwd_lean']):+.3f} | {np.mean(f_['back_lean']):+.3f}  fwd~back {np.mean(f_['fwd_back_sim']):.2f}")
    for c, v in out['forward_tests'].items():
        print(f"  {c:8s} " + '  '.join(f"{kk} {vv['diff']:+.3f}±{vv['se']:.3f} ({vv['pos']}/{vv['n']})" for kk, vv in v.items()))
for grp in ('perturb', 'ablate', 'clamp'):
    if grp in out:
        print(f'\n{grp}')
        for k, v in out[grp]['tests'].items():print(f"  {k:30s} diff {v['diff']:+.3f} ± {v['se']:.3f}  t {v['t']:+.1f}  ({v['pos']}/{v['n']})")
print('\nthemes');[print(f"  {i:2d} n={t['n']:5d}", ' '.join(t['words']), {c:round(v, 2) for c, v in t['share'].items()}) for i, t in enumerate(themes)]
