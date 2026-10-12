"""Research v5 Phase Q2: qualification of the MEMORY loop (PROTOCOL_V5 amendment Q-4).

Phase Q showed that the fixed-context loop is memoryless: every step recomputes the decision position from the fixed
prompt and the current write and discards the previous step, so the state is a function of (prompt, last write) and a
compact self-state can carry nothing beyond content and the last action (G2: dR2 = 0.001 / -0.008; G3: the same
machinery detects a hidden state component, dR2 0.25-0.33 on dlo). Here each loop step is kept: the decision token is
appended as a new position (positions L, L+1, ...), the write is added at L22 of that new position, and every later
step attends to all earlier loop positions — the system's own past processing becomes part of its context.

Observables at step t: lo_t at the new position; c^ch_t = lo on the challenged prompt's own memory rollout under the
same writes, minus lo_t. Deltas are taken against the same prompt's NO-WRITE memory rollout at the same step (the
repeated token itself changes the state, so the baseline must have the same number of loop positions).

    python qual_mem.py --stage calib|g0|gain|g2|g3|pilot|report --out /workspace/v5/qm
G2 here has two baselines: the last action (content, step, lo_t, eps_t, eps_t-1) and the full action history
(eps_0 .. eps_t); z must add to the first; the second shows whether a compact self-state summarises history better than
the raw record of the system's own actions (finite data)."""
import argparse, json, math, os, pathlib, sys, time
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import qual as Q                                                       # noqa: E402
from qual import (T, KZ, NF, DEV, CH, LR, LW, GAINS, log, atomic_npz, atomic_json, f2_items, content_onehot, cand_sets,
                  lo_mass, Calib, MLP, fit_mlp, grouped_cv, r2, boot_scen, load_lm)                    # noqa: E402

CNT = Q.CNT


class MemEng(Q.Eng):
    @torch.no_grad()
    def prefix(self, texts):
        P = super().prefix(texts);P['t'] = 0;return P

    @torch.no_grad()
    def step(self, P, w, read=(LR,)):
        """Append the decision token at the next loop position with w added at L_w; keep its KV (no crop)."""
        lm = self.lm;lm.clear()
        for l in read:lm.read[l] = None
        if w is not None:
            ww = w.to(lm.model.dtype);lm.edit[LW] = lambda h, ww=ww:h+ww[:, None, :]
        B = P['last'].shape[0];t = P['t']
        att = torch.cat([P['att'][:, :-1], torch.ones(B, t+1, dtype=P['att'].dtype, device=DEV)], 1)
        torch.cuda.synchronize();t0 = time.time()
        o = lm.model(input_ids=P['last'], attention_mask=att, position_ids=P['pos']+t, past_key_values=P['cache'],
                     use_cache=True, logits_to_keep=1)
        out = {l:lm.store[l][:, -1].float() for l in read};out['logits'] = o.logits[:, -1].float()
        torch.cuda.synchronize();CNT['passes'] += B;CNT['t_gpu'] += time.time()-t0
        P['t'] = t+1;lm.clear();return out

    @torch.no_grad()
    def full(self, P, ws):
        """Whole-sequence recomputation: prompt + (len(ws)) copies of the decision token, write ws[j] at position j."""
        lm = self.lm;lm.clear();n = len(ws);ids = torch.cat([P['ids']]+[P['last']]*(n-1), 1)
        att = torch.cat([P['full_att']]+[torch.ones_like(P['last'])]*(n-1), 1)
        Wt = torch.stack([w if w is not None else torch.zeros(ids.shape[0], lm.d, device=DEV) for w in ws], 1).to(lm.model.dtype)

        def ed(h, Wt=Wt):
            h = h.clone();h[:, -n:] = h[:, -n:]+Wt;return h
        lm.edit[LW] = ed;posid = (att.long().cumsum(-1)-1).clamp_min(0)
        o = lm.model(input_ids=ids, attention_mask=att, position_ids=posid, use_cache=False, logits_to_keep=1)
        lm.clear();return o.logits[:, -1].float()


@torch.no_grad()
def rollout(eng, C, items, mode='none', g=0., seed=0, writer=1, F=None, cand='R', hidden=None, corrupt=None, Tn=T, ffun=None):
    """As qual.rollout, with memory. ffun(z, t, rows) overrides F (used by Phase P branches)."""
    lm = eng.lm;B = len(items);sets = cand_sets(lm, items)
    P = {ch:eng.prefix([it['text'][ch] for it in items]) for ch in ['none']+CH}
    gen = torch.Generator(device=DEV);gen.manual_seed(seed);Mz, Mf = C.M(writer);vh = C.t['v_h']
    R = dict(zR=np.zeros((Tn+1, B, KZ), np.float32), zJ=np.zeros((Tn+1, B, KZ), np.float32), lo=np.zeros((Tn+1, B), np.float32),
             c=np.zeros((Tn+1, B, len(CH)), np.float32), mass=np.zeros((Tn+1, B), np.float32),
             eps=np.zeros((Tn, B, KZ+NF), np.float32), f=np.full((Tn, B, NF), np.nan, np.float32), wn=np.zeros((Tn, B), np.float32))
    hid = None
    if hidden is not None:
        sig = torch.tensor(hidden[0], device=DEV, dtype=torch.float32);hid = sig[:, None]*hidden[1]*float(C.np['n_J'])*vh[None]
    w = hid
    for t in range(Tn+1):
        o = eng.step(P['none'], w);lo, m = lo_mass(o['logits'], sets);R['lo'][t] = lo.cpu().numpy();R['mass'][t] = m.cpu().numpy()
        for j, ch in enumerate(CH):
            oc = eng.step(P[ch], w, read=());loc, _ = lo_mass(oc['logits'], sets);R['c'][t, :, j] = (loc-lo).cpu().numpy()
        zR, zJ, _ = C.readers(lm, o[LR]);R['zR'][t] = zR.cpu().numpy();R['zJ'][t] = zJ.cpu().numpy()
        if t == Tn:break
        z = zR if cand == 'R' else zJ
        if mode == 'none':u = None
        elif mode == 'random':
            e = torch.randn(B, KZ+NF, device=DEV, generator=gen);R['eps'][t] = e.cpu().numpy();u = e[:, :KZ]@Mz.T+e[:, KZ:]@Mf.T
        else:
            f = ffun(z, t) if ffun is not None else F(z, t)
            if corrupt is not None and t == corrupt['k']:
                ff = corrupt['f'];ok = torch.isfinite(ff).all(-1);f = torch.where(ok[:, None], ff, f)
            R['f'][t] = f.cpu().numpy();u = z@Mz.T+f@Mf.T
        ww = C.write(u, g) if u is not None else None
        R['wn'][t] = ww.norm(dim=-1).cpu().numpy() if ww is not None else 0.
        w = ww if hid is None else (hid if ww is None else ww+hid)
    del P;torch.cuda.empty_cache();return R


def batched(eng, C, items, **kw):
    out = None;hidden = kw.pop('hidden', None);corrupt = kw.pop('corrupt', None);ffun = kw.pop('ffun', None)
    for s in range(0, len(items), Q.BS):
        sl = slice(s, s+Q.BS);kk = dict(kw);kk['seed'] = kw.get('seed', 0)*1000+s
        if hidden is not None:kk['hidden'] = (hidden[0][sl], hidden[1])
        if corrupt is not None:kk['corrupt'] = dict(k=corrupt['k'], f=torch.tensor(corrupt['f'][sl], device=DEV))
        if ffun is not None:
            rows = np.arange(s, min(len(items), s+Q.BS));kk['ffun'] = (lambda z, t, rows=rows:ffun(z, t, rows))
        r = rollout(eng, C, items[sl], **kk);out = r if out is None else {k:np.concatenate([out[k], r[k]], 1) for k in r}
    return out


def natural(eng, C, items):
    """No-write memory rollout: per step baseline (T+1, B) of lo, (T+1, B, 4) of c, and the candidate mass."""
    r = batched(eng, C, items, mode='none');return r['lo'], r['c'], r['mass']


def deltas(r, nat):
    return r['lo']-nat[0], r['c']-nat[1]


# ----------------------------------------------------------------------------------------------------------------
def stage_calib(out):
    """Readers fitted on no-write memory rollouts of training prompts (all loop positions), writers as in Phase Q."""
    lm = load_lm();eng = MemEng(lm);t0 = time.time();tr = f2_items('train');va = f2_items('val')
    HR, HW = [], []
    for s in range(0, len(tr), Q.BS):
        it = tr[s:s+Q.BS];P = eng.prefix([x['text']['none'] for x in it])
        for t in range(T+1):
            o = eng.step(P, None, read=(LR, LW));HR.append(o[LR].cpu());HW.append(o[LW].cpu())
        del P
    HR, HW = torch.cat(HR).to(DEV), torch.cat(HW).to(DEV)
    mu_r = HR.mean(0);sd = HR.std(0);sd_r = torch.maximum(sd, .1*sd.median())
    Y = (HR-mu_r)/sd_r;_, _, V = torch.linalg.svd(Y-Y.mean(0), full_matrices=False);V_R = V[:KZ];zs_R = (Y@V_R.T).std(0).clamp_min(1e-6)
    recs = torch.cat([lm.lens.decompose(LR, HR[s:s+256]-mu_r, 25)[2] for s in range(0, len(HR), 256)])
    rec_mu = recs.mean(0);_, _, V2 = torch.linalg.svd(recs-rec_mu, full_matrices=False);V_J = V2[:KZ]
    zs_J = ((recs-rec_mu)@V_J.T).std(0).clamp_min(1e-6);n_J = recs.norm(dim=-1).median()
    mu_w = HW.mean(0);sd_w = HW.std(0);massive = sd_w > 20*sd_w.median()
    Xw = (HW-mu_w);Xw[:, massive] = 0;_, Sw, Vw = torch.linalg.svd(Xw, full_matrices=False);P_w = Vw[:256]
    Cd = dict(mu_r=mu_r, sd_r=sd_r, V_R=V_R, zs_R=zs_R, rec_mu=rec_mu, V_J=V_J, zs_J=zs_J, n_J=n_J, mu_w=mu_w, sd_w=sd_w,
              massive=massive.float(), P_w=P_w, pw_var=(Sw[:256]**2).sum()/(Sw**2).sum())
    for sd_ in (1, 2, 3):
        gq = torch.Generator(device=DEV);gq.manual_seed(1000+sd_);npw = P_w.shape[0]
        Mz = P_w.T@torch.randn(npw, KZ, device=DEV, generator=gq);Mz = Mz/Mz.norm(dim=0, keepdim=True)
        Mf = P_w.T@torch.randn(npw, NF, device=DEV, generator=gq);Mf = Mf/Mf.norm(dim=0, keepdim=True)*math.sqrt(KZ/NF)
        Cd[f'Mz{sd_}'], Cd[f'Mf{sd_}'] = Mz, Mf
    # natural memory rollouts on validation (no write): observables per step; hidden-mode direction from step-0 L_w states
    Cd['v_h'] = torch.zeros_like(mu_w)
    atomic_npz(out/'calib.npz', **{k:(v.float().cpu().numpy() if torch.is_tensor(v) else np.array(v, np.float32)) for k, v in Cd.items()})
    C = Calib(out/'calib.npz');nat = natural(eng, C, va)
    hw = []
    for s in range(0, len(va), Q.BS):
        it = va[s:s+Q.BS];P = eng.prefix([x['text']['none'] for x in it]);o = eng.step(P, None, read=(LW,));hw.append(o[LW]);del P
    hw = torch.cat(hw);c0 = torch.tensor(nat[1][0, :, 0], device=DEV);q1, q3 = torch.quantile(c0, .25), torch.quantile(c0, .75)
    X = hw-mu_w;X[:, massive] = 0;vh = X[c0 >= q3].mean(0)-X[c0 <= q1].mean(0);Cd['v_h'] = vh/vh.norm()
    atomic_npz(out/'calib.npz', **{k:(v.float().cpu().numpy() if torch.is_tensor(v) else np.array(v, np.float32)) for k, v in Cd.items()},
               nat_val_lo=nat[0], nat_val_c=nat[1])
    drift = {t:float(np.sqrt(np.mean((nat[0][t]-nat[0][0])**2))) for t in range(T+1)}
    info = dict(n_train=len(tr), n_val=len(va), n_J=float(n_J), massive_dims=int(massive.sum()),
                nat_val=dict(mass_frac_ge_0_5=float(np.mean(nat[2] >= .5)), mean_abs_c_t0={ch:float(np.abs(nat[1][0, :, j]).mean()) for j, ch in enumerate(CH)},
                             rms_lo_drift_without_writes=drift), seconds=time.time()-t0)
    atomic_json(out/'calib.json', info);log('calib', info)


def stage_g0(out):
    lm = load_lm();eng = MemEng(lm);C = Calib(out/'calib.npz');va = f2_items('val')[:64];res = {}
    a = batched(eng, C, va, mode='random', g=1., seed=3);b = batched(eng, C, va, mode='random', g=1., seed=3)
    res['a_replay_bit_identical'] = bool(all(np.array_equal(a[k], b[k], equal_nan=True) for k in a))
    n0 = batched(eng, C, va, mode='none');n1 = batched(eng, C, va[::-1], mode='none')
    res['c_branch_order_maxdiff'] = float(max(np.abs(n0[k]-n1[k][:, ::-1]).max() for k in ('lo', 'c', 'zR', 'zJ')))
    # cached memory path vs whole-sequence recomputation after 3 loop steps with writes
    it = va[:32];sets = cand_sets(lm, it);P = eng.prefix([x['text']['none'] for x in it]);g = torch.Generator(device=DEV);g.manual_seed(5)
    Mz, Mf = C.M(1);ws = []
    for s in range(3):
        e = torch.randn(32, KZ+NF, device=DEV, generator=g);ws.append(C.write(e[:, :KZ]@Mz.T+e[:, KZ:]@Mf.T, 1.))
    for w in [None]+ws[:-1]:eng.step(P, w, read=())                    # positions 0..2 with writes None, w0, w1
    lc, _ = lo_mass(eng.step(P, ws[-1], read=())['logits'], sets)      # position 3 with w2
    P2 = eng.prefix([x['text']['none'] for x in it]);lf, _ = lo_mass(eng.full(P2, [None]+ws), sets)
    res['b_cached_vs_full_q95_lo'] = float(np.quantile((lc-lf).abs().cpu().numpy(), .95))
    res['a_pass'] = res['a_replay_bit_identical'];res['c_pass'] = res['c_branch_order_maxdiff'] <= 1e-3
    atomic_json(out/'g0.json', res);log('g0', res)


def noise_floor(out):
    try:return float(json.loads((out/'g0.json').read_text())['b_cached_vs_full_q95_lo'])
    except Exception:return .6


def stage_gain(out):
    lm = load_lm();eng = MemEng(lm);C = Calib(out/'calib.npz');it = f2_items('val', 50);nat = natural(eng, C, it);res = {}
    for g in GAINS:
        r = batched(eng, C, it, mode='random', g=g, seed=11);dlo, dc = deltas(r, nat)
        fin = np.isfinite(r['lo']).all(0)&np.isfinite(r['c']).all((0, 2))
        res[g] = dict(rms_dc_contra=float(np.sqrt(np.mean(dc[1:, fin, 0]**2))), rms_dlo=float(np.sqrt(np.mean(dlo[1:, fin]**2))),
                      mass_frac_ge_0_5=float(np.mean(r['mass'][1:, fin] >= .5)), diverged=int((~fin).sum()),
                      persistence=float(np.sqrt(np.mean(dc[-1, fin, 0]**2))/max(np.sqrt(np.mean(dc[1, fin, 0]**2)), 1e-9)))
        log('gain', g, res[g])
    nf = max(noise_floor(out), .62);thr = max(1., 5*nf)
    ok = [g for g in GAINS if res[g]['rms_dc_contra'] >= thr and res[g]['mass_frac_ge_0_5'] >= .9 and res[g]['diverged'] == 0]
    atomic_json(out/'gain.json', dict(per_gain=res, g_star=ok[0] if ok else None, threshold=thr, noise_floor=nf));log('g*', ok[0] if ok else None)


def gstar(out):
    g = json.loads((out/'gain.json').read_text())['g_star']
    if g is None:raise SystemExit('no gain qualifies')
    return g


def g2_contrast(items, r, cand, nat, tag):
    """Predictors of (dc, dlo) at t+1: last action (content, step, lo_t, eps_t, eps_t-1) [+ z_t own | + z_t swap];
    full action history (eps_0..eps_t, zero-padded) [+ z_t own]."""
    rng = np.random.default_rng(0);Xc = content_onehot(items);n = len(items);Z = r['z'+cand];dlo, dc = deltas(r, nat);rows = []
    for t in range(T):
        sw = Q.swap_index(items, r['lo'], t, rng)
        for i in range(n):
            if sw[i] < 0:continue
            last = np.concatenate([Xc[i], np.eye(T, dtype=np.float32)[t], [r['lo'][t, i]], r['eps'][t, i],
                                   r['eps'][t-1, i] if t > 0 else np.zeros(KZ+NF, np.float32)])
            hist = np.zeros((T, KZ+NF), np.float32);hist[:t+1] = r['eps'][:t+1, i]
            rows.append((items[i]['sid'], last, hist.ravel(), Z[t, i], Z[t, sw[i]], dc[t+1, i, 0], dlo[t+1, i]))
    g = np.array([x[0] for x in rows]);L_ = np.stack([x[1] for x in rows]);H_ = np.stack([x[2] for x in rows])
    Zo = np.stack([x[3] for x in rows]);Zs = np.stack([x[4] for x in rows]);Y = np.stack([[x[5], x[6]] for x in rows]).astype(np.float32)
    designs = dict(last=L_, last_own=np.concatenate([L_, Zo], 1), last_swap=np.concatenate([L_, Zs], 1),
                   hist=np.concatenate([L_, H_], 1), hist_own=np.concatenate([L_, H_, Zo], 1))
    Pd = {k:grouped_cv(v.astype(np.float32), Y, g) for k, v in designs.items()};res = {}
    for j, nm in enumerate(('dc_contra', 'dlo')):
        y = Y[:, j];r2s = {k:r2(y, Pd[k][:, j]) for k in Pd}
        d1 = r2s['last_own']-r2s['last_swap'];ci1 = boot_scen(g, lambda ix:r2(y[ix], Pd['last_own'][ix, j])-r2(y[ix], Pd['last_swap'][ix, j]), n=2000)
        d2 = r2s['hist_own']-r2s['hist'];ci2 = boot_scen(g, lambda ix:r2(y[ix], Pd['hist_own'][ix, j])-r2(y[ix], Pd['hist'][ix, j]), n=2000)
        res[nm] = dict(r2=r2s, delta_vs_swap=d1, ci_vs_swap=ci1, pass_vs_swap=bool(d1 >= .05 and ci1[0] > 0), delta_beyond_history=d2,
                       ci_beyond_history=ci2)
    res['n_rows'] = int(len(g));log(tag, cand, res);return res


def stage_g2(out):
    lm = load_lm();eng = MemEng(lm);C = Calib(out/'calib.npz');g = gstar(out);it = f2_items('val');nat = natural(eng, C, it)
    r = batched(eng, C, it, mode='random', g=g, seed=21);atomic_npz(out/'g2_rollouts.npz', nat_lo=nat[0], nat_c=nat[1], **r)
    dlo, dc = deltas(r, nat);snr = float(np.sqrt(np.mean(dc[1:, :, 0]**2))/max(noise_floor(out), .62))
    # memory check: does a write at step t still change the outcome at t+2, t+3 when later writes are held fixed? (lag regression)
    res = dict(g_star=g, G1c_snr=snr, G1c_pass=snr >= 5)
    for cand in ('R', 'J'):res[cand] = g2_contrast(it, r, cand, nat, 'G2mem')
    res['pass_any'] = bool(res['R']['dc_contra']['pass_vs_swap'] or res['J']['dc_contra']['pass_vs_swap'])
    atomic_json(out/'g2.json', res);log('g2', {k:v for k, v in res.items() if k not in ('R', 'J')})


def stage_g3(out):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    lm = load_lm();eng = MemEng(lm);C = Calib(out/'calib.npz');g = gstar(out);it = f2_items('val');nat = natural(eng, C, it)
    sig = np.where(np.random.default_rng(31).random(len(it)) < .5, -1., 1.)
    r = batched(eng, C, it, mode='random', g=g, seed=31, hidden=(sig, .5));res = dict(g_star=g)
    groups = np.array([x['sid'] for x in it])
    for cand in ('R', 'J'):
        Z = r['z'+cand][1:].transpose(1, 0, 2).reshape(len(it), -1);acc = []
        for tr_, te in GroupKFold(5).split(Z, sig, groups):
            m = LogisticRegression(C=1., max_iter=3000).fit(Z[tr_], sig[tr_]);acc.append(float((m.predict(Z[te]) == sig[te]).mean()))
        res[cand] = dict(sigma_decode=float(np.mean(acc)), contrast=g2_contrast(it, r, cand, nat, 'G3mem'))
    atomic_json(out/'g3.json', res);log('g3', {c:res[c]['sigma_decode'] for c in ('R', 'J')})


def stage_report(out):
    J = {n:json.loads((out/f'{n}.json').read_text()) for n in ('calib', 'g0', 'gain', 'g2', 'g3') if (out/f'{n}.json').exists()}
    rep = dict(g_star=J.get('gain', {}).get('g_star'), G0=J.get('g0'), G2_pass_any=J.get('g2', {}).get('pass_any'),
               G2_R=J.get('g2', {}).get('R', {}).get('dc_contra'), G2_J=J.get('g2', {}).get('J', {}).get('dc_contra'),
               G3={c:J.get('g3', {}).get(c, {}).get('sigma_decode') for c in ('R', 'J')})
    atomic_json(out/'q_report.json', rep);log('report', rep)


def main():
    p = argparse.ArgumentParser();p.add_argument('--stage', required=True);p.add_argument('--out', required=True)
    a = p.parse_args();out = pathlib.Path(a.out);out.mkdir(parents=True, exist_ok=True);t0 = time.time()
    {'calib':stage_calib, 'g0':stage_g0, 'gain':stage_gain, 'g2':stage_g2, 'g3':stage_g3, 'report':stage_report}[a.stage](out)
    tm = dict(stage=a.stage, seconds=time.time()-t0, passes=CNT['passes'], ms_per_pass=1000*CNT['t_gpu']/max(CNT['passes'], 1),
              peak_gb=float(torch.cuda.max_memory_allocated()/1e9))
    with open(out/'timing.jsonl', 'a') as f:f.write(json.dumps(tm)+'\n')
    log('timing', tm)


if __name__ == '__main__':
    main()
