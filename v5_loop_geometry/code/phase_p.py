"""Research v5 Phase P (PROTOCOL_V5 §6): the self-forecast loop with the B0 writer — does the forecast keep up with a
change its own channel causes? Reuses qual.py's engine (fixed prompt KV, write at L22, read at L29, lo and c).
All observables are write-induced changes relative to the prompt's own unwritten values (Amendment Q-1).

    python phase_p.py --stage train --q /workspace/v5/q --out /workspace/v5/p [--writers 1,2,3] [--cands R,J]
    python phase_p.py --stage test  --q /workspace/v5/q --out /workspace/v5/p

train: policy iteration of F (round 0 with forecasts 0, rounds 1-3 with F_r; F_{r+1} on rounds 0..r weighted 1,1,2,4),
       F_swap on the same transitions with content-matched z_swap, own-policy calibration of F_4 on validation scenarios.
test:  test scenarios; systems S-own, S-swap, S-zero; branches own / restored / continued / sham / dose at k = 2;
       donor rule §6.3 on deltas; estimands E1-E5 (§6.4) with scenario-grouped bootstrap; every array saved."""
import argparse, json, pathlib, sys, time
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import os
import qual as Q                                                  # noqa: E402
from qual import T, KZ, NF, DEV, log, atomic_npz, atomic_json      # noqa: E402
MEM = int(os.environ.get('P_MEM', 0))                             # 1: the memory loop of qual_mem.py (amendment Q-4)
if MEM:
    import qual_mem as E                                           # noqa: E402
    ENG, NATURAL, DELTAS, BATCHED = E.MemEng, E.natural, E.deltas, E.batched
else:
    ENG, NATURAL, DELTAS, BATCHED = Q.Eng, Q.natural, Q.deltas, Q.batched

K_FORK = 2


class Fnet(torch.nn.Module):
    def __init__(self, h=128):
        super().__init__();self.net = torch.nn.Sequential(torch.nn.Linear(KZ+T, h), torch.nn.GELU(), torch.nn.Linear(h, h),
                                                          torch.nn.GELU(), torch.nn.Linear(h, NF))

    def forward(self, z, t):
        return self.net(torch.cat([z, torch.eye(T, device=z.device)[t].expand(len(z), T)], 1))


def as_F(net):
    def F(z, t):
        with torch.no_grad():return net(z.float(), t)
    return F


def zero_F(z, t):
    return torch.zeros(len(z), NF, device=DEV)


def load_net(path):
    n = Fnet().to(DEV);n.load_state_dict(torch.load(path, map_location=DEV));n.eval();return n


def dtargets(R, nat, ystat):
    """(T, B, NF) standardised forecast targets: (dlo, dc_contra) at t+1, t+2, t+3; NaN beyond the horizon."""
    dlo, dc = DELTAS(R, nat);my, sy = ystat;Tn = dlo.shape[0]-1;B = dlo.shape[1];Y = np.full((Tn, B, NF), np.nan, np.float32)
    for t in range(Tn):
        for h in (1, 2, 3):
            if t+h <= Tn:Y[t, :, 2*(h-1)] = dlo[t+h];Y[t, :, 2*(h-1)+1] = dc[t+h, :, 0]
    return (Y-my)/sy


def ystat_from_q(qdir):
    r = dict(np.load(pathlib.Path(qdir)/'g2_rollouts.npz'));dlo, dc = DELTAS(r, (r['nat_lo'], r['nat_c']));Y = []
    for h in (1, 2, 3):Y += [dlo[h:].ravel(), dc[h:, :, 0].ravel()]
    return np.array([np.mean(y) for y in Y], np.float32), np.array([np.std(y)+1e-6 for y in Y], np.float32)


def transitions(R, cand, Y, items, swap=None):
    Z = R['z'+cand];zs, ts, ys, gs = [], [], [], []
    for t in range(T):
        for i in range(len(items)):
            src = i if swap is None else swap[t][i]
            if src < 0:continue
            zs.append(Z[t, src]);ts.append(t);ys.append(Y[t, i]);gs.append(items[i]['sid'])
    return np.array(zs, np.float32), np.array(ts), np.array(ys, np.float32), np.array(gs)


def swap_table(items, R):
    rng = np.random.default_rng(0);return [Q.swap_index(items, R['lo'], t, rng) for t in range(T)]


def fit_F(Z, Tt, Y, groups, w, seed=0, epochs=200):
    torch.manual_seed(seed);us = np.unique(groups);rng = np.random.default_rng(seed)
    va = set(rng.choice(us, max(1, int(.15*len(us))), replace=False));vm = np.array([g in va for g in groups])
    X = torch.tensor(Z, device=DEV);Tt = torch.tensor(Tt, device=DEV, dtype=torch.long);Yt = torch.tensor(Y, device=DEV)
    W = torch.tensor(w, device=DEV);M = torch.isfinite(Yt).float();Yt = torch.nan_to_num(Yt)
    net = Fnet().to(DEV);opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    tr_i = torch.tensor(np.where(~vm)[0], device=DEV);va_i = torch.tensor(np.where(vm)[0], device=DEV);best, state, bad = 1e9, None, 0

    def loss(ix):
        e = (net(X[ix], Tt[ix])-Yt[ix])**2*M[ix];return (e.sum(1)*W[ix]).sum()/((M[ix].sum(1)*W[ix]).sum()).clamp_min(1)
    for ep in range(epochs):
        for b in torch.split(tr_i[torch.randperm(len(tr_i), device=DEV)], 256):
            l = loss(b);opt.zero_grad();l.backward();opt.step()
        with torch.no_grad():v = float(loss(va_i))
        if v < best-1e-5:best, bad, state = v, 0, {k:x.clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 15:break
    net.load_state_dict(state);net.eval();return net, best


def stage_train(a, out):
    lm = Q.load_lm();eng = ENG(lm);qd = pathlib.Path(a.q);C = Q.Calib(qd/'calib.npz');g = Q.gstar(qd);ystat = ystat_from_q(qd)
    tr = Q.f2_items('train');va = Q.f2_items('val');nat_tr = NATURAL(eng, C, tr);nat_va = NATURAL(eng, C, va);info = dict(g_star=g, mem=MEM)
    for wr in a.writers:
        for cand in a.cands:
            tag = f'w{wr}_{cand}';rounds = [];F = zero_F
            for r in range(4):
                R = BATCHED(eng, C, tr, mode='closed', g=g, writer=wr, F=F, cand=cand, seed=100*wr+r)
                atomic_npz(out/f'train_{tag}_r{r}.npz', **R);rounds.append(R)
                Zs, Ts, Ys, Gs, Ws = [], [], [], [], []
                for j, Rj in enumerate(rounds):
                    z_, t_, y_, g_ = transitions(Rj, cand, dtargets(Rj, nat_tr, ystat), tr)
                    Zs.append(z_);Ts.append(t_);Ys.append(y_);Gs.append(g_);Ws.append(np.full(len(y_), [1, 1, 2, 4][j], np.float32))
                net, vloss = fit_F(*map(np.concatenate, (Zs, Ts, Ys, Gs, Ws)), seed=r);F = as_F(net)
                log('train', tag, 'round', r, 'val loss', round(vloss, 4))
            torch.save(net.state_dict(), out/f'F_{tag}.pt')
            Zs, Ts, Ys, Gs, Ws = [], [], [], [], []
            for j, Rj in enumerate(rounds):
                z_, t_, y_, g_ = transitions(Rj, cand, dtargets(Rj, nat_tr, ystat), tr, swap=swap_table(tr, Rj))
                Zs.append(z_);Ts.append(t_);Ys.append(y_);Gs.append(g_);Ws.append(np.full(len(y_), [1, 1, 2, 4][j], np.float32))
            net_s, vls = fit_F(*map(np.concatenate, (Zs, Ts, Ys, Gs, Ws)), seed=9);torch.save(net_s.state_dict(), out/f'Fswap_{tag}.pt')
            Rv = BATCHED(eng, C, va, mode='closed', g=g, writer=wr, F=F, cand=cand, seed=900+wr)
            Yv = dtargets(Rv, nat_va, ystat)[:, :, 1];fv = Rv['f'][:, :, 1];ok = np.isfinite(Yv)
            info[tag] = dict(own_policy_skill_dc_h1=float(1-np.mean((fv[ok]-Yv[ok])**2)/max(np.var(Yv[ok]), 1e-9)),
                             val_loss_F=vloss, val_loss_Fswap=vls)
            atomic_npz(out/f'calib_{tag}.npz', **Rv);log('calibration', tag, info[tag])
    atomic_json(out/'train.json', info)


def donors(items, R, nat, ystat, k=K_FORK):
    """§6.3 on deltas: two donors per rollout (the second for the sham branch); -1 = none."""
    my, sy = ystat;dlo, _ = DELTAS(R, nat);f = R['f'][k];fl = f[:, 0]*sy[0]+my[0];fc = f[:, 1];sd_c = float(np.nanstd(R['f'][:, :, 1]))
    key = [(x['sub'], x['evidence'], x['edir']) for x in items];sid = [x['sid'] for x in items];uses = {};D = np.full((len(items), 2), -1)
    for i in range(len(items)):
        el = [j for j in range(len(items)) if key[j] == key[i] and sid[j] != sid[i] and abs(dlo[k, j]-dlo[k, i]) <= .5
              and abs(fl[j]-fl[i]) <= .5 and abs(fc[j]-fc[i]) >= sd_c]
        el = sorted(el, key=lambda q:(-abs(fc[q]-fc[i]), q));chosen = []
        for j in el:
            if uses.get(j, 0) >= 3 or (chosen and sid[j] == sid[chosen[0]]):continue
            chosen.append(j);uses[j] = uses.get(j, 0)+1
            if len(chosen) == 2:break
        for c_, j in enumerate(chosen):D[i, c_] = j
    return D


@torch.no_grad()
def rollout_fn(eng, C, items, g, wr, cand, Ffun, extra=None, seed=0):
    """Closed loop where Ffun(z, t, rows) returns the forecasts used (rows = item indices of the batch) and extra(t, rows)
    an optional displacement added to the writer input u (dose control)."""
    lm = eng.lm;out = None
    for s in range(0, len(items), Q.BS):
        rows = np.arange(s, min(len(items), s+Q.BS));it = [items[i] for i in rows];B = len(it);sets = Q.cand_sets(lm, it)
        P = {ch:eng.prefix([x['text'][ch] for x in it]) for ch in ['none']+Q.CH};Mz, Mf = C.M(wr)
        R = dict(zR=np.zeros((T+1, B, KZ), np.float32), zJ=np.zeros((T+1, B, KZ), np.float32), lo=np.zeros((T+1, B), np.float32),
                 c=np.zeros((T+1, B, len(Q.CH)), np.float32), mass=np.zeros((T+1, B), np.float32), f=np.full((T, B, NF), np.nan, np.float32),
                 eps=np.zeros((T, B, KZ+NF), np.float32), wn=np.zeros((T, B), np.float32))
        w = None
        for t in range(T+1):
            o = eng.step(P['none'], w);lo, m = Q.lo_mass(o['logits'], sets);R['lo'][t] = lo.cpu().numpy();R['mass'][t] = m.cpu().numpy()
            for j, ch in enumerate(Q.CH):
                oc = eng.step(P[ch], w, read=());loc, _ = Q.lo_mass(oc['logits'], sets);R['c'][t, :, j] = (loc-lo).cpu().numpy()
            zR, zJ, _ = C.readers(lm, o[Q.LR]);R['zR'][t] = zR.cpu().numpy();R['zJ'][t] = zJ.cpu().numpy()
            if t == T:break
            z = zR if cand == 'R' else zJ;f = Ffun(z, t, rows);R['f'][t] = f.cpu().numpy();u = z@Mz.T+f@Mf.T
            if extra is not None:
                d = extra(t, rows)
                if d is not None:u = u+d
            w = C.write(u, g);R['wn'][t] = w.norm(dim=-1).cpu().numpy()
        del P;torch.cuda.empty_cache()
        out = R if out is None else {kk:np.concatenate([out[kk], R[kk]], 1) for kk in R}
    return out


def stage_test(a, out):
    lm = Q.load_lm();eng = ENG(lm);qd = pathlib.Path(a.q);C = Q.Calib(qd/'calib.npz');g = Q.gstar(qd);ystat = ystat_from_q(qd)
    te = Q.f2_items('test');nat = NATURAL(eng, C, te);res = dict(g_star=g, n_items=len(te), mem=MEM);k = K_FORK
    for wr in a.writers:
        for cand in a.cands:
            tag = f'w{wr}_{cand}';net = load_net(out/f'F_{tag}.pt');F = as_F(net);Fs = load_net(out/f'Fswap_{tag}.pt')
            own = rollout_fn(eng, C, te, g, wr, cand, lambda z, t, rows:F(z, t))
            D = donors(te, own, nat, ystat);has1 = D[:, 0] >= 0
            Fown = torch.tensor(own['f'], device=DEV)                  # own-branch forecasts (T, N, NF), standardised

            def corrupt_at(f, t, rows, which):
                src = which[rows];ok = torch.tensor(src >= 0, device=DEV)
                if ok.any():
                    dc_ = Fown[t, np.maximum(src, 0)];f = f.clone();f[:, 1::2] = torch.where(ok[:, None], dc_[:, 1::2], f[:, 1::2])
                return f
            rest = rollout_fn(eng, C, te, g, wr, cand, lambda z, t, rows:corrupt_at(F(z, t), t, rows, D[:, 0]) if t == k else F(z, t))
            cont = rollout_fn(eng, C, te, g, wr, cand, lambda z, t, rows:corrupt_at(F(z, t), t, rows, D[:, 0]) if t >= k else F(z, t))
            sham = rollout_fn(eng, C, te, g, wr, cand, lambda z, t, rows:(corrupt_at(F(z, t), t, rows, D[:, 0]) if t == k else
                                                                         corrupt_at(F(z, t), t, rows, D[:, 1]) if t > k else F(z, t)))
            Mz, Mf = C.M(wr);gen = torch.Generator(device=DEV);gen.manual_seed(77)
            dfn = np.nan_to_num(rest['f'][k]-own['f'][k]);dn = torch.tensor(np.linalg.norm(dfn@Mf.cpu().numpy().T, axis=1), device=DEV)

            def dose_extra(t, rows):
                if t != k:return None
                e = torch.randn(len(rows), KZ+NF, device=DEV, generator=gen);d = e[:, :KZ]@Mz.T+e[:, KZ:]@Mf.T
                return d/d.norm(dim=1, keepdim=True)*dn[rows][:, None]
            dose = rollout_fn(eng, C, te, g, wr, cand, lambda z, t, rows:F(z, t), extra=dose_extra)
            sw = swap_table(te, own);zsw = np.stack([own['z'+cand][t, [s_ if s_ >= 0 else i for i, s_ in enumerate(sw[t])]] for t in range(T)])
            with torch.no_grad():fsw = torch.tensor(np.stack([Fs(torch.tensor(zsw[t], device=DEV), t).cpu().numpy() for t in range(T)]), device=DEV)
            sswap = rollout_fn(eng, C, te, g, wr, cand, lambda z, t, rows:fsw[t, rows])
            szero = rollout_fn(eng, C, te, g, wr, cand, lambda z, t, rows:zero_F(z, t))
            for nm, R_ in (('own', own), ('restored', rest), ('continued', cont), ('sham', sham), ('dose', dose), ('sswap', sswap), ('szero', szero)):
                atomic_npz(out/f'test_{tag}_{nm}.npz', **R_)
            np.save(out/f'donors_{tag}.npy', D)
            res[tag] = estimands(te, nat, own, rest, cont, sham, dose, D, Fs, cand, ystat, sw)
            log('test', tag, {kk:vv for kk, vv in res[tag].items()})
    atomic_json(out/'test.json', res)


def estimands(items, nat, own, rest, cont, sham, dose, D, Fs, cand, ystat, sw, k=K_FORK, nboot=10000):
    my, sy = ystat;sid = np.array([x['sid'] for x in items]);has = D[:, 0] >= 0
    dcf = lambda R:DELTAS(R, nat)[1][:, :, 0]
    fc = lambda R:R['f'][:, :, 1]*sy[1]+my[1]                           # h = 1 forecast of dc in nats

    def Fs_on(R, ts):
        with torch.no_grad():
            return np.stack([Fs(torch.tensor(R['z'+cand][t, [s_ if s_ >= 0 else i for i, s_ in enumerate(sw[t])]], device=DEV), t).cpu().numpy()
                             for t in ts])[:, :, 1]*sy[1]+my[1]
    y = dcf(own)[1:];eF = (fc(own)-y)**2;eS = (Fs_on(own, range(T))-y)**2                       # E1
    manip = float(np.sqrt(np.mean(((dcf(rest)-dcf(own))[k+1:, has])**2)))                     # E2
    manip_dose = float(np.sqrt(np.mean(((dcf(dose)-dcf(own))[k+1:, has])**2)))
    yr = dcf(rest)[k+2:];yo = dcf(own)[k+2:]                                                 # E3: forecasts t=3..5 -> outcomes 4..6
    e_fresh = (fc(rest)[k+1:T]-yr)**2;e_stale = (fc(own)[k+1:T]-yr)**2;e_own = (fc(own)[k+1:T]-yo)**2
    es_fresh = (Fs_on(rest, range(k+1, T))-yr)**2;es_stale = (Fs_on(own, range(k+1, T))-yr)**2   # E4
    rows = {s:np.where((sid == s)&has)[0] for s in np.unique(sid[has])};rows_all = {s:np.where(sid == s)[0] for s in np.unique(sid)}
    per = lambda A, rr:np.array([A[:, rr[s]].mean() for s in rr])
    P = dict(eF=per(eF, rows_all), eS=per(eS, rows_all), fresh=per(e_fresh, rows), stale=per(e_stale, rows), own=per(e_own, rows),
             sfresh=per(es_fresh, rows), sstale=per(es_stale, rows))
    fns = dict(E1_skill=(lambda ix:1-P['eF'][ix].mean()/P['eS'][ix].mean(), 'eF'),
               E3_TR=(lambda ix:P['fresh'][ix].mean()/P['stale'][ix].mean(), 'fresh'),
               E3_NI=(lambda ix:P['fresh'][ix].mean()/P['own'][ix].mean(), 'fresh'),
               E4_dTR=(lambda ix:P['fresh'][ix].mean()/P['stale'][ix].mean()-P['sfresh'][ix].mean()/P['sstale'][ix].mean(), 'fresh'))
    rng = np.random.default_rng(0);out = dict(coverage=float(has.mean()), coverage_sham=float((D[:, 1] >= 0).mean()), E2_manip_rms=manip,
                                              E2_dose_rms=manip_dose, n_scen_E3=int(len(rows)))
    for nm, (fn, key) in fns.items():
        n_ = len(P[key])
        if n_ < 3:out[nm] = None;continue
        bs = [fn(rng.integers(0, n_, n_)) for _ in range(nboot)]
        out[nm] = float(fn(np.arange(n_)));out[nm+'_ci'] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]
        out[nm+'_ci_holm'] = [float(np.quantile(bs, .05/8/2)), float(np.quantile(bs, 1-.05/8/2))]
    for nm, R in (('restored', rest), ('sham', sham), ('continued', cont)):
        out[f'E5_dist_{nm}'] = float(np.abs(dcf(R)[k+2:]-dcf(own)[k+2:])[:, has].mean())
    return out


def main():
    p = argparse.ArgumentParser();p.add_argument('--stage', required=True);p.add_argument('--q', required=True);p.add_argument('--out', required=True)
    p.add_argument('--writers', default='1,2,3');p.add_argument('--cands', default='R,J')
    a = p.parse_args();a.writers = [int(x) for x in a.writers.split(',')];a.cands = a.cands.split(',')
    out = pathlib.Path(a.out);out.mkdir(parents=True, exist_ok=True);t0 = time.time()
    {'train':stage_train, 'test':stage_test}[a.stage](a, out)
    log('done', a.stage, f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
