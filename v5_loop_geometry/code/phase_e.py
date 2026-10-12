"""Research v5 Phase E (PROTOCOL_V5 §6b): coupled evolution — the self-forecaster keeps learning while its forecasts
drive the writes, so every improvement changes the process it must predict (the author's 2024 tension). Nothing rewards
complexity; the only objective is the forecaster's own prediction error.

Systems (per writer and candidate, F initialised from Phase P's F_4):
  online   F updated after every batch on the transitions that batch produced (window 1024, 20 AdamW steps, lr 3e-4)
  frozen   same loop, F fixed
  swap     forecasts from F_swap(z of a content-matched rollout in P's test pool), F_swap learning online the same way
  observer F learning online, forecasts not written (forecast weight 0)
Per batch: F's error on the outcomes it forecast (before they happened), the loop's modulation of the disposition
(RMS dc), late step size and dimension of z; every 5 batches the error on a fixed probe set (P's validation rollouts).
    python phase_e.py --q /workspace/v5/q --p /workspace/v5/p --out /workspace/v5/e [--writers 1] [--cands R,J] [--batches 60]"""
import argparse, collections, copy, json, pathlib, sys, time
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import qual as Q                                                   # noqa: E402
import phase_p as PP                                                # noqa: E402
from qual import T, KZ, NF, DEV, log, atomic_npz, atomic_json       # noqa: E402


def pr(X):
    X = X-X.mean(0);s = np.linalg.svd(X.astype(np.float64), compute_uv=False)**2;return float(s.sum()**2/max((s**2).sum(), 1e-30))


def main():
    p = argparse.ArgumentParser();p.add_argument('--q', required=True);p.add_argument('--p', required=True);p.add_argument('--out', required=True)
    p.add_argument('--writers', default='1');p.add_argument('--cands', default='R,J');p.add_argument('--batches', type=int, default=60)
    p.add_argument('--bs', type=int, default=64)
    a = p.parse_args();out = pathlib.Path(a.out);out.mkdir(parents=True, exist_ok=True);qd, pd = pathlib.Path(a.q), pathlib.Path(a.p)
    lm = Q.load_lm();eng = PP.ENG(lm);C = Q.Calib(qd/'calib.npz');g = Q.gstar(qd);ystat = PP.ystat_from_q(qd);my, sy = ystat
    te = Q.f2_items('test');va = Q.f2_items('val');nat_te = PP.NATURAL(eng, C, te);nat_va = PP.NATURAL(eng, C, va);res = dict(g_star=g)
    key = [(x['sub'], x['evidence'], x['edir']) for x in te];sid = [x['sid'] for x in te]
    for wr in [int(x) for x in a.writers.split(',')]:
        for cand in a.cands.split(','):
            tag = f'w{wr}_{cand}'
            if not (pd/f'F_{tag}.pt').exists():log('missing', tag);continue
            F0 = PP.load_net(pd/f'F_{tag}.pt');Fs0 = PP.load_net(pd/f'Fswap_{tag}.pt')
            pool = dict(np.load(pd/f'test_{tag}_own.npz'))                       # P's own-branch test rollouts: swap pool
            match = []                                                         # deterministic content-matched partner per item
            for i in range(len(te)):
                c_ = [j for j in range(len(te)) if key[j] == key[i] and sid[j] != sid[i]];match.append(c_[i % len(c_)] if c_ else i)
            match = np.array(match)
            Rv = dict(np.load(pd/f'calib_{tag}.npz'));Yv = PP.dtargets(Rv, nat_va, ystat)                 # fixed probe set
            Zp, Tp, Yp, _ = PP.transitions(Rv, cand, Yv, va);Zp, Tp, Yp = (torch.tensor(Zp, device=DEV), torch.tensor(Tp, device=DEV),
                                                                          torch.tensor(Yp, device=DEV))
            for system in ('online', 'frozen', 'swap', 'observer'):
                net = copy.deepcopy(Fs0 if system == 'swap' else F0);net.train(system != 'frozen')
                opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4);win = collections.deque(maxlen=1024)
                rng = np.random.default_rng(1000*wr+{'online':1, 'frozen':2, 'swap':3, 'observer':4}[system]);curve = [];snaps = {}
                for b in range(a.batches):
                    ix = rng.choice(len(te), a.bs, replace=False);items = [te[i] for i in ix]
                    nat = tuple(x[..., ix] if x.ndim == 2 else x[:, ix] for x in nat_te[:2]) if PP.MEM else (nat_te[0][ix], nat_te[1][ix])

                    def Ffun(z, t, rows, ix=ix):
                        with torch.no_grad():
                            if system == 'swap':
                                zz = torch.tensor(pool['z'+cand][t, match[ix[rows]]], device=DEV);return net(zz, t)
                            f = net(z.float(), t)
                            return f*0 if system == 'observer' else f
                    obs_f = []

                    def Fobs(z, t, rows, ix=ix):                              # the observer's own forecasts (not written)
                        f = Ffun(z, t, rows)
                        if system == 'observer':
                            with torch.no_grad():obs_f.append(net(z.float(), t).cpu().numpy())
                        return f
                    R = PP.rollout_fn(eng, C, items, g, wr, cand, Fobs, seed=10_000*b+wr)
                    dlo, dc = PP.DELTAS(R, nat);Y = PP.dtargets(R, nat, ystat)
                    fcast = np.stack(obs_f) if system == 'observer' else R['f']          # forecasts made before the outcomes
                    err = float(np.nanmean((fcast[:, :, 1]-Y[:, :, 1])**2))
                    zz = R['z'+cand];st = dict(batch=b, err_h1_dc=err, dc_rms=float(np.sqrt(np.mean(dc[1:, :, 0]**2))),
                                               dlo_rms=float(np.sqrt(np.mean(dlo[1:]**2))), late_step=float(np.linalg.norm(zz[-1]-zz[-2], axis=-1).mean()),
                                               pr_z_end=pr(zz[-1]), mass_frac=float(np.mean(R['mass'][1:] >= .5)))
                    if system != 'frozen':
                        for t in range(T):
                            src = zz[t] if system != 'swap' else pool['z'+cand][t, match[ix]]
                            for i in range(len(items)):win.append((src[i], t, Y[t, i]))
                        for _ in range(20):
                            sel = rng.choice(len(win), min(256, len(win)), replace=False)
                            z_ = torch.tensor(np.stack([win[s][0] for s in sel]), device=DEV);t_ = torch.tensor([win[s][1] for s in sel], device=DEV)
                            y_ = torch.tensor(np.stack([win[s][2] for s in sel]), device=DEV);m_ = torch.isfinite(y_).float();y_ = torch.nan_to_num(y_)
                            oh = torch.eye(T, device=DEV)[t_];pred = net.net(torch.cat([z_, oh], 1))
                            loss = (((pred-y_)**2)*m_).sum()/m_.sum().clamp_min(1);opt.zero_grad();loss.backward();opt.step()
                    if b % 5 == 0 or b == a.batches-1:
                        with torch.no_grad():
                            oh = torch.eye(T, device=DEV)[Tp.long()];pp = net.net(torch.cat([Zp, oh], 1));mp = torch.isfinite(Yp)   # own codes, every system
                            st['probe_err_h1_dc'] = float((((pp-torch.nan_to_num(Yp))**2)*mp)[:, 1].sum()/mp[:, 1].sum())
                    if b in (0, 20, 40, a.batches-1):snaps[b] = R
                    curve.append(st)
                    if b % 10 == 0:log('E', tag, system, st)
                res[f'{tag}|{system}'] = curve
                for b_, R_ in snaps.items():atomic_npz(out/f'snap_{tag}_{system}_b{b_}.npz', **R_)
                atomic_json(out/'phase_e.json', res)
    atomic_json(out/'phase_e.json', res);log('done')


if __name__ == '__main__':
    main()
