"""Research v7b: does the state know whether its own read worked, and use that to read again?

Built on focus_sel.py (same streams, records, frozen model and loop). New: every read can fail to deliver (with
probability p_fail the read colour does not arrive; what arrives is the read's own "nothing new" weight with an empty
colour), and each chunk allows a second, optional read. Both reads cost a little (cost x the attention mass on records),
so reading again pays only when the first read failed while the tracked animal's record was there (the cost is off for
the first cost_start steps and ramps in over cost_ramp steps: with it on from the start both two-read arms stopped reading
within 100 steps, before reading could become useful). From scratch, the noisy channel makes learning the focus slow
(0.39 after 600 steps vs 0.88 without failures and noise), so the main runs start from a trained focus_sel.py loop
(--init; every arm from the same checkpoint): selection is already there, and what to do about failed reads and the
second read must still be learned. The read colour
passes a noisy channel (fixed noise, sigma per colour dimension), so a faint read (little attention mass on a record)
delivers mostly noise: reading cheaply by spreading attention does not work.

The two reads differ only in what the query sees:
  retry  the second read's query comes from the state after the first read (its outcome has passed through the
         frozen model into the state);
  blind  control: the second read's query comes from the state before the first read, so it cannot know the outcome;
  single control: one read per chunk (no second chance).
Both reads use the same query map, with a flag saying which read it is. Objective: the right colour at every question
plus the read cost. Nothing rewards reading again, or not, directly.

Measured: accuracy; second-read attention on the tracked record when the first read failed / delivered / when the
tracked animal is absent ("tried and failed" and "chose nothing" deliver the same empty colour; only the read's own
"nothing new" weight tells them apart); whether the intermediate state carries "my read failed" (decoding, grouped by
stream); the answer's confidence when both reads failed (the stale colour is then always wrong); and edits of the
intermediate state along the failed-minus-delivered direction (class means from separate calibration streams), applied
to the whole state, only to the second read's query, or only to the second update, with random directions of the same
size as control. Also: a paired counterfactual — the same stream, history,
first attention and channel noise, branching only on whether the first read is delivered, then exchanging the two
intermediate states for the second query alone; test-time failure rates other than the trained one; and an arm in which
the read's own "nothing new" weight is not delivered (--no_null 1).
    python focus_retry.py train --out <run folder with records.pt> --arm retry --seed 0
    python focus_retry.py eval  --out <run folder> --tag retry_c0.1"""
import argparse, json, math, pathlib, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from focus_sel import DEV, DZ, NPH, RQ, Frozen, Loop, load_rec, log, make_stream, nce, sph, tensors


class Loop2(Loop):
    """The v7 loop with a read flag in the query (first or second read of the chunk)."""
    def __init__(self, d, keys_dim, norm=False):
        super().__init__(d, 'state', keys_dim, norm);self.Wq = nn.Linear(DZ+11+1, RQ)

    def query2(self, z, instr, second):
        return self.Wq(torch.cat([z, instr, torch.full_like(instr[:, :1], float(second))], -1))


def episode2(fm, loop, rec, T, D, arm, mid_fn=None, keep=False, no_null=False):
    """K chunks. D: 'fail' (B, K, 2) bool, the read's delivery fails; 'noise' (B, K, 2, 10), channel noise. mid_fn(k,
    z_mid) -> (z for the second query, z for the second update), for interventions on the intermediate state. Returns
    logits, states, intermediate states, attentions of both reads, read cost and the self-model's loss."""
    B, K, n_upd = T['rid'].shape;keys = rec['keys'][T['rid']];col = rec['colour'][T['rid']]
    z_prev = sph(loop.z0.expand(B, DZ));LG, Z, ZM, A1, A2, C, LS = [], [], [], [], [], [], []
    def state(h):
        return sph(loop.Rl((h-loop.mu_h)/loop.sd_h if loop.norm else h))
    def deliver(att, k, j):
        ok = (~D['fail'][:, k, j]).float()[:, None]
        return torch.cat([(att[:, :-1, None]*col[:, k]).sum(1)*ok+D['noise'][:, k, j], att[:, -1:]*(0. if no_null else 1.)], -1)
    for k in range(K):
        has = (T['iid'][:, k] >= 0).float()[:, None];instr = torch.cat([rec['instr'][T['iid'][:, k].clamp_min(0)]*has, has], -1)
        a1 = loop.select(loop.query2(z_prev, instr, 0), keys[:, k]);cost = 1-a1[:, -1]
        if arm == 'single':
            lg, h = fm.step(loop.write(z_prev, deliver(a1, k, 0), instr));z = state(h);zm = z;a2 = torch.zeros_like(a1)
        else:
            _, h1 = fm.step(loop.write(z_prev, deliver(a1, k, 0), instr));zm = state(h1)
            zq, zu = (zm, zm) if mid_fn is None else mid_fn(k, zm)
            a2 = loop.select(loop.query2(zq if arm == 'retry' else z_prev, instr, 1), keys[:, k]);cost = cost+(1-a2[:, -1])
            lg, h = fm.step(loop.write(zu, deliver(a2, k, 1), instr));z = state(h)
        if k > 0:LS.append(nce(loop.forecast(z_prev.detach()), z.detach()))
        LG.append(lg);Z.append(z);ZM.append(zm);C.append(cost)
        if keep:A1.append(a1.detach());A2.append(a2.detach())
        z_prev = z
    return dict(logits=torch.stack(LG, 1), z=torch.stack(Z, 1), zm=torch.stack(ZM, 1), cost=torch.stack(C, 1),
                a1=torch.stack(A1, 1) if keep else None, a2=torch.stack(A2, 1) if keep else None,
                ls=torch.stack(LS).mean() if LS else torch.zeros((), device=DEV))


def draw(rng, B, K, p, sigma):
    """Delivery failures and channel noise for B streams."""
    return dict(fail=torch.tensor(rng.random((B, K, 2)) < p, device=DEV),
                noise=torch.tensor(rng.normal(0, sigma, (B, K, 2, 10)), dtype=torch.float32, device=DEV))


def cut(D, sl):
    return {k:v[sl] for k, v in D.items()}


def tag_of(a):
    return (f'{a.arm}_c{a.cost:g}'+(f'_p{a.p_fail:g}' if a.p_fail != 0.3 else '')+('_nonull' if a.no_null else '')+('_ws' if a.init else '')
            +(f'_s{a.seed}' if a.seed else ''))


def warm_start(loop, path):
    """Start from a trained focus_sel.py loop (it already selects the tracked animal's record): every weight is copied;
    the query map gets one more input (which read it is), initialised to zero."""
    sd = torch.load(path, map_location=DEV);own = loop.state_dict()
    for k, v in sd.items():
        if k == 'Wq.weight':own[k].zero_();own[k][:, :v.shape[1]] = v
        elif k in own and own[k].shape == v.shape:own[k].copy_(v)
        else:raise ValueError(f'cannot copy {k}')
    loop.load_state_dict(own)


def train(a):
    out = pathlib.Path(a.out);rec = load_rec(out);fm = Frozen(rec['model'], rec['lw'], rec['lr'], a.local_only);tag = tag_of(a)
    torch.manual_seed(a.seed);loop = Loop2(fm.d, rec['keys'].shape[-1], a.norm).to(DEV)
    with torch.no_grad():                                    # write strength as in focus_sel.py
        o = fm.model(input_ids=fm.prompt[None], use_cache=False, output_hidden_states=True);fm.H = None
        loop.n_w.fill_(float(o.hidden_states[fm.lw+1][0, fm.P0+NPH:].float().norm(dim=-1).median()))
        if a.norm:
            kk = rec['keys'].float();loop.mu_k.copy_(kk.mean(0));loop.sd_k.copy_(kk.std(0)+1e-3*kk.std(0).mean())
            Hs = []
            for _ in range(4):
                zr = sph(torch.randn(64, DZ, device=DEV));cr = torch.softmax(torch.randn(64, 11, device=DEV), -1)
                _, h = fm.step(loop.write(zr, cr, torch.cat([torch.softmax(torch.randn(64, 10, device=DEV), -1), torch.ones(64, 1, device=DEV)], -1)))
                Hs.append(h)
            Hs = torch.cat(Hs);loop.mu_h.copy_(Hs.mean(0));loop.sd_h.copy_(Hs.std(0)+1e-3*Hs.std(0).mean())
    if a.init:warm_start(loop, a.init)
    opt = torch.optim.AdamW([p for p in loop.parameters() if p.requires_grad], lr=a.lrate, weight_decay=0.)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s:min(1., (s+1)/50)*(0.1+0.9*0.5*(1+math.cos(math.pi*min(1., s/a.steps)))))
    rng = np.random.default_rng(1000+a.seed);frng = np.random.default_rng(5000+a.seed);vrng = np.random.default_rng(777);hist = [];t0 = time.time()
    VT = tensors([make_stream(vrng, a.K, a.n_upd, a.p_switch, a.p_focus) for _ in range(256)], a.K, a.n_upd);VD = draw(np.random.default_rng(778), 256, a.K, a.p_fail, a.sigma)
    for s in range(a.steps):
        T = tensors([make_stream(rng, a.K, a.n_upd, a.p_switch, a.p_focus) for _ in range(a.bs)], a.K, a.n_upd)
        R = episode2(fm, loop, rec, T, draw(frng, a.bs, a.K, a.p_fail, a.sigma), a.arm, no_null=a.no_null)
        la = F.cross_entropy(R['logits'].reshape(-1, 10), T['gold'].reshape(-1));lc = R['cost'].mean()
        w = a.cost*min(1., max(0., (s-a.cost_start)/max(1, a.cost_ramp)))     # no cost until the read is learned, then ramp in
        loss = la+w*lc+R['ls']
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(loop.parameters(), 1.0);opt.step();sched.step()
        rec_ = dict(step=s, ans=float(la), reads=float(lc), ls=float(R['ls']), t=time.time()-t0)
        if s % 100 == 0 or s == a.steps-1:
            with torch.no_grad():
                Rv = episode2(fm, loop, rec, VT, VD, a.arm, no_null=a.no_null);ok = (Rv['logits'].argmax(-1) == VT['gold']).float()
            rec_['val_acc'] = float(ok.mean());rec_['val_updated'] = float(ok[VT['upd']].mean());rec_['val_kept'] = float(ok[~VT['upd']].mean())
            rec_['val_reads'] = float(Rv['cost'].mean())
            log(tag, f"step {s} answer {la:.3f} reads {lc:.2f} | val acc {rec_['val_acc']:.3f} (updated {rec_['val_updated']:.3f}, "
                     f"kept {rec_['val_kept']:.3f}) reads/chunk {rec_['val_reads']:.2f} | {rec_['t']:.0f}s")
        hist.append(rec_)
    (out/'ckpt').mkdir(exist_ok=True);torch.save(loop.state_dict(), out/'ckpt'/f'{tag}.pt')
    json.dump(dict(cfg=vars(a), tag=tag, hist=hist), open(out/'ckpt'/f'{tag}.json', 'w'));log(tag, 'trained', f'{time.time()-t0:.0f}s')


@torch.no_grad()
def evaluate(a):
    out = pathlib.Path(a.out);rec = load_rec(out);cfg = json.load(open(out/'ckpt'/f'{a.tag}.json'))['cfg'];arm = cfg['arm']
    fm = Frozen(rec['model'], rec['lw'], rec['lr'], a.local_only);loop = Loop2(fm.d, rec['keys'].shape[-1], cfg.get('norm', 0)).to(DEV)
    loop.load_state_dict(torch.load(out/'ckpt'/f'{a.tag}.pt', map_location=DEV));loop.eval()
    K, nu, p, sig, nn_ = cfg['K'], cfg['n_upd'], cfg['p_fail'], cfg['sigma'], bool(cfg.get('no_null', 0))
    def run(T_, D_, mid_fn=None):
        """mid_fn(k, z_mid, batch slice) -> (z for the second query, z for the second update)."""
        Rs = [];B = T_['rid'].shape[0]
        for i in range(0, B, a.bs):
            sl = slice(i, i+a.bs);fn = None if mid_fn is None else (lambda sl_:(lambda k, zm:mid_fn(k, zm, sl_)))(sl)
            Rs.append(episode2(fm, loop, rec, cut(T_, sl), cut(D_, sl), arm, mid_fn=fn, keep=True, no_null=nn_))
        return {k:torch.cat([r[k] for r in Rs]) for k in ('logits', 'z', 'zm', 'cost', 'a1', 'a2')}
    # held-out streams (the same as focus_sel.py's evaluation), their delivery failures and channel noise
    rng = np.random.default_rng(4242);S = [make_stream(rng, K, nu, cfg['p_switch'], cfg['p_focus']) for _ in range(a.n_eval)];T = tensors(S, K, nu)
    Dl = draw(np.random.default_rng(31337), a.n_eval, K, p, sig);Fl = Dl['fail'];R = run(T, Dl)
    pr = torch.softmax(R['logits'], -1);ok = (pr.argmax(-1) == T['gold']).float();upd = T['upd']
    isf = (T['ran'] == T['focus'][:, :, None]).float()
    on1 = (R['a1'][..., :-1]*isf).sum(-1);on2 = (R['a2'][..., :-1]*isf).sum(-1)
    f1, f2 = Fl[..., 0], Fl[..., 1]
    tried = on1 > 0.5                                        # the first read went to the tracked record
    cats = dict(present_first_failed=upd & tried & f1, present_first_delivered=upd & tried & ~f1, absent=~upd)
    res = dict(tag=a.tag, cfg=cfg, acc=float(ok.mean()), acc_by_chunk=ok.mean(0).tolist(), acc_updated=float(ok[upd].mean()),
               acc_kept=float(ok[~upd].mean()), reads_per_chunk=float(R['cost'].mean()),
               first_read_on_tracked_when_present=float(on1[upd].mean()), first_read_nothing_new_when_absent=float(R['a1'][..., -1][~upd].mean()))
    res['present_but_first_read_not_on_tracked'] = float((upd & ~tried).float().sum()/upd.float().sum())
    res['by_first_read'] = {c:dict(n=int(m.sum()), first_read_on_tracked=float(on1[m].mean()), first_read_nothing_new=float(R['a1'][..., -1][m].mean()),
                                   second_read_on_tracked=float(on2[m].mean()), second_read_nothing_new=float(R['a2'][..., -1][m].mean()),
                                   reads=float(R['cost'][m].mean()), acc=float(ok[m].mean())) for c, m in cats.items() if m.any()}
    # knowing that it does not know (operational categories): the update of the tracked animal was missed (first read
    # failed and the second failed or did not go to the tracked record) vs delivered by the first read or by the second.
    # 'previous true colour' = the tracked animal's colour before this chunk (always different from the new one)
    old = torch.full_like(T['gold'], -1);old[:, 1:] = T['gold'][:, :-1]
    same = torch.zeros_like(upd);same[:, 1:] = T['focus'][:, 1:] == T['focus'][:, :-1]
    lost = upd & tried & f1 & (f2 | (on2 < 0.5)) & same;got = upd & tried & ~f1 & same;rescued = upd & tried & f1 & ~f2 & (on2 >= 0.5) & same
    def conf(m):
        if not m.any():return None
        pm = pr[m];o = old[m].clamp_min(0)
        return dict(n=int(m.sum()), acc=float(ok[m].mean()), max_prob=float(pm.max(-1).values.mean()),
                    entropy=float(-(pm*pm.clamp_min(1e-9).log()).sum(-1).mean()), prob_previous_true_colour=float(pm.gather(1, o[:, None]).mean()))
    res['confidence'] = dict(update_not_acquired=conf(lost & (old >= 0)), acquired_by_first_read=conf(got & (old >= 0)),
                             acquired_by_second_read=conf(rescued & (old >= 0)))
    # what the intermediate state carries about its own read: decode 'first read failed' on chunks where it went to the
    # tracked record (grouped by stream), and 'tried and failed' vs 'chose nothing' (the delivered colour is empty in both)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:LogisticRegression = None            # decoding is then done from the saved arrays
    zm = R['zm'].cpu().numpy();gid = np.repeat(np.arange(zm.shape[0]), zm.shape[1]).reshape(zm.shape[:2])
    def dec(m, y):
        m = m.cpu().numpy();y = y.cpu().numpy()[m].astype(int)
        if LogisticRegression is None or min((y == 0).sum(), (y == 1).sum()) < 10:return None
        pr_ = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced')), zm[m], y,
                                cv=GroupKFold(5), groups=gid[m])
        return float(np.mean([(pr_[y == c] == c).mean() for c in (0, 1)]))
    chose_nothing = ~upd & (R['a1'][..., -1] > 0.5)
    res['decode_mid_state'] = dict(first_read_failed=dec(upd & tried, f1),
                                   tried_and_failed_vs_chose_nothing=dec((upd & tried & f1) | chose_nothing, upd))
    # paired counterfactual: same streams, history, first attention and noise; branch only on whether the first
    # read at chunk t is delivered (the second read, if made, is delivered in both branches); then exchange the two
    # intermediate states for the second query only (each branch keeps its own state for the update). Pooled over t = 2-4
    # on (stream, t) where the tracked record is present and the first read went to it.
    if arm != 'single':
        acc_ = {}
        def forced(t, fails):
            D2 = {k:v.clone() for k, v in Dl.items()};D2['fail'][:, t, 0] = fails;D2['fail'][:, t, 1] = False;return D2
        def qswap(src, t):
            return lambda k, zm_, sl:((src[sl], zm_) if k == t else (zm_, zm_))
        for t in (2, 3, 4):
            DD, DF = forced(t, False), forced(t, True);RD, RF = run(T, DD), run(T, DF)
            RDs, RFs = run(T, DD, qswap(RF['zm'][:, t], t)), run(T, DF, qswap(RD['zm'][:, t], t))
            m = (upd & tried)[:, t]
            for c, Rx in (('delivered', RD), ('failed', RF), ('delivered_query_from_failed_state', RDs), ('failed_query_from_delivered_state', RFs)):
                o2 = (Rx['a2'][:, t, :-1]*isf[:, t]).sum(-1);okx = (Rx['logits'].argmax(-1) == T['gold']).float()
                v = torch.stack([o2[m], Rx['a2'][:, t, -1][m], okx[:, t][m], okx[:, t+1][m]], 1)
                acc_[c] = torch.cat([acc_[c], v]) if c in acc_ else v
        res['counterfactual_first_delivery'] = {c:dict(n=int(v.shape[0]), second_read_on_tracked=float(v[:, 0].mean()), second_read_nothing_new=float(v[:, 1].mean()),
                                                       acc=float(v[:, 2].mean()), next_acc=float(v[:, 3].mean())) for c, v in acc_.items()}
    # other failure rates at test time (trained at p): does the second read follow what happened or a fixed habit?
    res['test_failure_rate'] = {}
    for pt in (0.0, 0.6):
        Rt = run(T, draw(np.random.default_rng(31337), a.n_eval, K, pt, sig));okt = (Rt['logits'].argmax(-1) == T['gold']).float()
        on1t = (Rt['a1'][..., :-1]*isf).sum(-1);on2t = (Rt['a2'][..., :-1]*isf).sum(-1)
        Ft = draw(np.random.default_rng(31337), a.n_eval, K, pt, sig)['fail'][..., 0];tr_ = upd & (on1t > 0.5)
        res['test_failure_rate'][f'{pt:g}'] = dict(acc=float(okt.mean()), reads_per_chunk=float(Rt['cost'].mean()),
                                                    second_read_on_tracked_after_failed=float(on2t[tr_ & Ft].mean()) if (tr_ & Ft).any() else None,
                                                    second_read_on_tracked_after_delivered=float(on2t[tr_ & ~Ft].mean()) if (tr_ & ~Ft).any() else None)
    # edits of the intermediate state: direction failed - delivered (class means of normalised intermediate states over
    # separate calibration streams, first read on the tracked record)
    if arm != 'single':
        crng = np.random.default_rng(777);CT = tensors([make_stream(crng, K, nu, cfg['p_switch'], cfg['p_focus']) for _ in range(a.n_eval)], K, nu)
        CD = draw(np.random.default_rng(778), a.n_eval, K, p, sig);CF = CD['fail'];RC = run(CT, CD)
        ci = (CT['ran'] == CT['focus'][:, :, None]).float();ctried = (RC['a1'][..., :-1]*ci).sum(-1) > 0.5
        Zc = F.normalize(RC['zm'], dim=-1);m_fail = Zc[CT['upd'] & ctried & CF[..., 0]].mean(0);m_ok = Zc[CT['upd'] & ctried & ~CF[..., 0]].mean(0)
        dvec = m_fail-m_ok;g = torch.Generator(device='cpu').manual_seed(0)
        rnds = [F.normalize(torch.randn(DZ, generator=g), dim=0).to(DEV)*dvec.norm() for _ in range(3)]
        res['edit_direction_norm'] = float(dvec.norm())
        res['calibration_counts'] = dict(failed=int((CT['upd'] & ctried & CF[..., 0]).sum()), delivered=int((CT['upd'] & ctried & ~CF[..., 0]).sum()))
        def mk(delta, where, t, disp):
            def fn(k, zm_, sl):
                if k != t:return zm_, zm_
                ze = sph(F.normalize(zm_, dim=-1)+delta);disp.append(float((1-F.cosine_similarity(ze, zm_, dim=-1)).mean()))
                return {'both':(ze, ze), 'query':(ze, zm_), 'update':(zm_, ze)}[where]
            return fn
        conds = {'none':(None, None), 'toward_failed':(dvec, 'both'), 'toward_failed_x0.5':(0.5*dvec, 'both'), 'toward_failed_x2':(2*dvec, 'both'),
                 'toward_failed_query_only':(dvec, 'query'), 'toward_failed_update_only':(dvec, 'update'),
                 **{f'random_same_size_r{i}':(r_, 'both') for i, r_ in enumerate(rnds)},
                 'toward_delivered':(-dvec, 'both'), 'toward_delivered_query_only':(-dvec, 'query'), 'toward_delivered_update_only':(-dvec, 'update')}
        E = {}
        for c, (delta, where) in conds.items():
            agg = {};disp = []
            for t in (2, 3, 4):                              # one edited chunk at a time; categories from the unedited run
                Re = R if delta is None else run(T, Dl, mk(delta, where, t, disp))
                on2e = (Re['a2'][..., :-1]*isf).sum(-1);oke = (Re['logits'].argmax(-1) == T['gold']).float()
                for cat, m in cats.items():
                    mt = m[:, t]
                    if not mt.any():continue
                    agg.setdefault(cat, []).append(dict(second_read_on_tracked=float(on2e[:, t][mt].mean()), second_read_nothing_new=float(Re['a2'][:, t, -1][mt].mean()),
                                                        acc=float(oke[:, t][mt].mean()), next_acc=float(oke[:, t+1][mt].mean())))
            E[c] = {cat:{k:float(np.mean([x[k] for x in v])) for k in v[0]} for cat, v in agg.items()}
            E[c]['state_moved_1_minus_cos'] = float(np.mean(disp)) if disp else 0.
        rk = [k for k in E if k.startswith('random_same_size_r')]           # the three random directions, averaged
        E['random_same_size'] = {cat:{k:float(np.mean([E[r][cat][k] for r in rk])) for k in E[rk[0]][cat]} for cat in E[rk[0]] if cat != 'state_moved_1_minus_cos'}
        E['random_same_size']['state_moved_1_minus_cos'] = float(np.mean([E[r]['state_moved_1_minus_cos'] for r in rk]))
        res['mid_state_edits'] = E
    np.savez_compressed(out/f'evalr_{a.tag}.npz', z=R['z'].cpu().numpy(), zm=zm, a1=R['a1'].cpu().numpy(), a2=R['a2'].cpu().numpy(),
                        fail=Fl.cpu().numpy(), ran=T['ran'].cpu().numpy(), focus=T['focus'].cpu().numpy(), gold=T['gold'].cpu().numpy(), upd=upd.cpu().numpy(),
                        ok=ok.cpu().numpy(), probs=pr.cpu().numpy())
    json.dump(res, open(out/f'evalr_{a.tag}.json', 'w'), indent=1)
    log(a.tag, json.dumps({k:v for k, v in res.items() if k != 'cfg'}))


def main():
    p = argparse.ArgumentParser();p.add_argument('stage', choices=['train', 'eval']);p.add_argument('--out', required=True)
    p.add_argument('--local_only', type=int, default=0);p.add_argument('--arm', default='retry', choices=['retry', 'blind', 'single'])
    p.add_argument('--cost', type=float, default=0.1);p.add_argument('--p_fail', type=float, default=0.3);p.add_argument('--sigma', type=float, default=0.1)
    p.add_argument('--no_null', type=int, default=0);p.add_argument('--cost_start', type=int, default=300);p.add_argument('--cost_ramp', type=int, default=300)
    p.add_argument('--init', default='', help='start from a trained focus_sel.py checkpoint (warm start)')
    p.add_argument('--steps', type=int, default=1200);p.add_argument('--bs', type=int, default=64);p.add_argument('--lrate', type=float, default=2e-3)
    p.add_argument('--seed', type=int, default=0);p.add_argument('--tag', default=None)
    p.add_argument('--K', type=int, default=6);p.add_argument('--n_upd', type=int, default=4);p.add_argument('--p_switch', type=float, default=0.3)
    p.add_argument('--p_focus', type=float, default=0.5);p.add_argument('--n_eval', type=int, default=512);p.add_argument('--norm', type=int, default=0)
    a = p.parse_args()
    if a.stage == 'train':train(a)
    else:evaluate(a)


if __name__ == '__main__':
    main()
