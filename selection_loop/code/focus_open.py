"""Research v8 (open access): does selective reading emerge when nothing forces it? (DESIGN_open.md)

Same task as focus_sel.py, but the frozen model reads every chunk as ordinary text — instruction (if any) and all four
records together, its own attention free. Per chunk the loop writes its state (and its self-model's forecast) into
placeholders at block L_w; the next state is read from the model's own representation of the chunk's last token at block
L_r; the question is asked with the new state written in. Any selection has to happen inside the frozen model's
attention, steered by the written state. Training: the right colour at every question (plus the self-model's own loss).

Evaluation: accuracy (and with no memory); what the state holds (saved for decoding: the tracked animal's colour vs the
other animals updated in the same chunk); focus-code edits on twin streams before the chunk is read (intake) and after
(the question only), with random, half-strength and absent-animal controls; and the frozen model's own attention from
the chunk's last token to each record, with the trained state, without memory, with no write, and under the edit.
    python focus_open.py train --out <dir> --model Qwen/Qwen2.5-0.5B-Instruct --lw 8 --lr 16 --dz 32
    python focus_open.py eval  --out <dir> --model Qwen/Qwen2.5-0.5B-Instruct --lw 8 --lr 16 --tag open_z32"""
import argparse, json, math, pathlib, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
import focus_loop as FL
from focus_loop import ANIMALS, NF, NPH, NZ, Frozen, log, make_stream, rendered, stats, task_args

DEV = FL.DEV


class Loop(nn.Module):
    def __init__(self, d, dz, mu, sd, n_w):
        super().__init__();self.d, self.dz, self.n_w = d, dz, n_w
        self.register_buffer('mu', mu);self.register_buffer('sd', sd)
        self.Wz = nn.Linear(dz, NZ*d);self.Wf = nn.Linear(dz, NF*d);self.Rl = nn.Linear(d, dz)
        self.S = nn.Sequential(nn.Linear(dz, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, dz))
        self.z0 = nn.Parameter(torch.randn(dz))

    def sph(self, x):
        return math.sqrt(self.dz)*F.normalize(x, dim=-1)

    def init(self, B):
        return self.sph(self.z0.expand(B, self.dz))

    def forecast(self, z):
        return self.sph(self.S(z))

    def forecast_frozen(self, z):
        return self.sph(functional_call(self.S, {n:p.detach() for n, p in self.S.named_parameters()}, (z,)))

    def write(self, z):
        """State (4 slots) and the self-model's forecast of the next state (2 slots), each slot at the model's typical size."""
        w = torch.cat([self.Wz(z).view(-1, NZ, self.d), self.Wf(self.forecast_frozen(z)).view(-1, NF, self.d)], 1)
        return self.n_w*w/w.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def read(self, h):
        return self.sph(self.Rl((h.float()-self.mu)/self.sd))


def nce(pred, tgt, tau=0.1):
    p = F.normalize(pred, dim=-1);t = F.normalize(tgt, dim=-1)
    return F.cross_entropy(p@t.T/tau, torch.arange(len(p), device=p.device))


def episode(fm, loop, sts, z_fn=None, za_fn=None, w_none_at=None, probe=None):
    """K chunks. z_fn(k, z): the state written before chunk k; za_fn(k, z): the state the question sees after chunk k
    (the carried state stays unedited); w_none_at: chunk with no write at all; probe: dict(k=chunk, store=list) to
    capture the model's attention while it reads that chunk."""
    B = len(sts);K = len(sts[0]['ids']);z = loop.init(B);LG, Z, LS = [], [], []
    for k in range(K):
        zin = z if z_fn is None else z_fn(k, z)
        W = None if w_none_at == k else loop.write(zin)
        if probe is not None:probe['on'] = probe['k'] == k
        H, att, ln = fm.read_chunk([s['ids'][k] for s in sts], W)
        if probe is not None:probe['on'] = False
        zn = loop.read(H[torch.arange(B, device=H.device), ln-1])
        zq = zn if za_fn is None else za_fn(k, zn)
        LG.append(fm.answer(B, loop.write(zq)))
        if k > 0:LS.append(nce(loop.forecast(z.detach()), zn.detach()))
        Z.append(zn);z = zn
    return dict(logits=torch.stack(LG, 1), z=torch.stack(Z, 1), ls=torch.stack(LS).mean() if LS else torch.zeros((), device=DEV))


def info(st):
    """Per chunk: was the tracked animal updated; the records (animal, colour); spans (relative to the chunk text)."""
    K = len(st['chunks']);upd = [any(a == st['focus'][k] for a, c, t in st['chunks'][k][1]) for k in range(K)]
    rec = [[(a, c) for a, c, t in st['chunks'][k][1]] for k in range(K)]
    return upd, rec


def tag_of(a):
    return f'open_z{a.dz}'+a.suffix+(f'_s{a.seed}' if a.seed else '')


def build(fm, a, out):
    sp = out/'stats.pt'
    if sp.exists():mu, sd, n_w = torch.load(sp)
    else:mu, sd, n_w = stats(fm, a);torch.save((mu, sd, n_w), sp)
    return Loop(fm.d, a.dz, mu.to(DEV), sd.to(DEV), n_w).to(DEV)


def train(a):
    out = pathlib.Path(a.out);out.mkdir(parents=True, exist_ok=True);tag = tag_of(a);(out/'ckpt').mkdir(exist_ok=True)
    fm = Frozen(a.model, a.lw, a.lr, bool(a.local_only));torch.manual_seed(a.seed);loop = build(fm, a, out)
    if a.init:loop.load_state_dict(torch.load(a.init, map_location=DEV));log(tag, 'continuing from', a.init)
    opt = torch.optim.AdamW([p for p in loop.parameters() if p.requires_grad], lr=a.lrate, weight_decay=0.)
    hz = dict(n=a.steps)                                     # schedule horizon; shortened to fit --max_minutes
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s:min(1., (s+1)/50)*(0.1+0.9*0.5*(1+math.cos(math.pi*min(1., s/hz['n'])))))
    rng = np.random.default_rng(1000+a.seed);vrng = np.random.default_rng(777);hist = [];t0 = time.time()
    val = [rendered(fm, make_stream(vrng, **task_args(a))) for _ in range(128)]
    s = -1
    while s+1 < hz['n']:
        s += 1
        if s == 20 and a.max_minutes > 0:                    # measured speed -> the number of steps that fits the budget
            hz['n'] = max(21, min(a.steps, int((a.max_minutes*60-(time.time()-t0))/((time.time()-t0)/20))+20));log(tag, 'steps', hz['n'])
        sts = [rendered(fm, make_stream(rng, **task_args(a))) for _ in range(a.bs)];gold = torch.tensor([x['gold'] for x in sts], device=DEV)
        R = episode(fm, loop, sts);la = F.cross_entropy(R['logits'].reshape(-1, a.n_col), gold.reshape(-1));loss = la+R['ls']
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(loop.parameters(), 1.0);opt.step();sched.step()
        rec = dict(step=s, ans=float(la), ls=float(R['ls']), t=time.time()-t0)
        if s % 100 == 0 or s == hz['n']-1:
            with torch.no_grad():
                ok, up = [], []
                for i in range(0, len(val), a.bs):
                    vb = val[i:i+a.bs];Rv = episode(fm, loop, vb);gv = torch.tensor([x['gold'] for x in vb], device=DEV)
                    ok.append((Rv['logits'].argmax(-1) == gv).float().cpu());up.append(torch.tensor([info(x)[0] for x in vb]))
                ok, up = torch.cat(ok), torch.cat(up)
            rec.update(val_acc=float(ok.mean()), val_updated=float(ok[up].mean()), val_kept=float(ok[~up].mean()))
            log(tag, f"step {s} answer {la:.3f} | val acc {rec['val_acc']:.3f} (updated {rec['val_updated']:.3f}, kept {rec['val_kept']:.3f}) "
                     f"by chunk {np.round(ok.mean(0).tolist(), 2).tolist()} | ls {rec['ls']:.3f} | {rec['t']:.0f}s")
        hist.append(rec)
    torch.save(loop.state_dict(), out/'ckpt'/f'{tag}.pt')
    json.dump(dict(cfg=vars(a), tag=tag, hist=hist), open(out/'ckpt'/f'{tag}.json', 'w'));log(tag, 'trained', f'{time.time()-t0:.0f}s')


class AttnProbe:
    """Captures the frozen model's attention weights (layers L_w+1..L_r) while it reads one chunk (eager attention)."""
    def __init__(self, fm):
        self.fm, self.on, self.k, self.store = fm, False, None, []
        for i in range(fm.lw+1, fm.lr+1):fm.layers[i].self_attn.register_forward_hook(self._hook(i))

    def _hook(self, i):
        def fn(mod, inp, out):
            if self.on and isinstance(out, tuple) and len(out) > 1 and out[1] is not None:self.store.append((i, out[1].float().mean(1).cpu()))
        return fn

    def __getitem__(self, key):
        return {'on':self.on, 'k':self.k}[key]

    def __setitem__(self, key, v):
        setattr(self, key, v)


def set_eager(fm):
    try:fm.model.set_attn_implementation('eager')
    except Exception:fm.model.config._attn_implementation = 'eager'


@torch.no_grad()
def evaluate(a):
    out = pathlib.Path(a.out);tag = a.tag;cfg = json.load(open(out/'ckpt'/f'{tag}.json'))['cfg']
    for k in ('dz', 'lw', 'lr', 'model'):setattr(a, k, cfg[k])
    fm = Frozen(a.model, a.lw, a.lr, bool(a.local_only));loop = build(fm, a, out)
    loop.load_state_dict(torch.load(out/'ckpt'/f'{tag}.pt', map_location=DEV));loop.eval();K = a.K;bs = a.bs;res = dict(tag=tag, cfg=cfg, n_pairs=a.n_pairs, n_eval=a.n_eval)
    def run(S, z_fn=None, za_fn=None, **kw):
        Rs = []
        for i in range(0, len(S), bs):
            sl = slice(i, min(i+bs, len(S)))
            zf = None if z_fn is None else (lambda sl_:(lambda k, z:z_fn(k, z, sl_)))(sl)
            zaf = None if za_fn is None else (lambda sl_:(lambda k, z:za_fn(k, z, sl_)))(sl)
            Rs.append(episode(fm, loop, S[sl], z_fn=zf, za_fn=zaf, **kw))
        return dict(logits=torch.cat([r['logits'] for r in Rs]), z=torch.cat([r['z'] for r in Rs]))
    # accuracy on held-out streams; no memory
    S = [rendered(fm, make_stream(np.random.default_rng(4242+i), **task_args(a))) for i in range(a.n_eval)]
    gold = torch.tensor([x['gold'] for x in S], device=DEV);I = [info(x) for x in S];upd = torch.tensor([i[0] for i in I], device=DEV)
    sw = torch.tensor([x['switch'] for x in S], device=DEV);free = torch.ones_like(upd);free[:, 0] = False
    for b, k in enumerate(sw.tolist()):
        if k >= 0:free[b, k] = False
    R0 = run(S);ok = (R0['logits'].argmax(-1) == gold).float()
    Rn = run(S, z_fn=lambda k, z, sl:loop.init(z.shape[0]));okn = (Rn['logits'].argmax(-1) == gold).float()
    res.update(acc=float(ok.mean()), acc_by_chunk=ok.mean(0).tolist(), acc_updated=float(ok[upd].mean()), acc_kept=float(ok[~upd].mean()),
               acc_instruction_free=float(ok[free].mean()), no_memory_acc_by_chunk=okn.mean(0).tolist(), no_memory_instruction_free=float(okn[free].mean()))
    log(tag, 'acc', round(res['acc'], 3), '| updated', round(res['acc_updated'], 3), '| kept', round(res['acc_kept'], 3),
        '| instruction-free', round(res['acc_instruction_free'], 3), '| no memory', round(res['no_memory_instruction_free'], 3))
    rec_an = np.array([[[a_ for a_, c_ in i[1][k]] for k in range(K)] for i in I]);rec_col = np.array([[[c_ for a_, c_ in i[1][k]] for k in range(K)] for i in I])
    np.savez_compressed(out/f'eval_{tag}.npz', z=R0['z'].cpu().numpy(), focus=np.array([x['focus'] for x in S]), gold=gold.cpu().numpy(),
                        switch=sw.cpu().numpy(), upd=upd.cpu().numpy(), ok=ok.cpu().numpy(), ok_no_memory=okn.cpu().numpy(), rec_an=rec_an, rec_col=rec_col)
    # focus class means from separate calibration streams (by tracked animal, averaged over colours)
    C = [rendered(fm, make_stream(np.random.default_rng(777+i), **task_args(a))) for i in range(a.n_eval)];RC = run(C)
    Zu = F.normalize(RC['z'], dim=-1).reshape(-1, loop.dz);fo = torch.tensor([x['focus'] for x in C], device=DEV).reshape(-1)
    go = torch.tensor([x['gold'] for x in C], device=DEV).reshape(-1)
    cell = torch.zeros(a.n_an, a.n_col, loop.dz, device=DEV);cnt = torch.zeros(a.n_an, a.n_col, device=DEV)
    cell.index_put_((fo, go), Zu, accumulate=True);cnt.index_put_((fo, go), torch.ones_like(fo, dtype=torch.float), accumulate=True)
    okc = cnt > 0;cm = cell/cnt.clamp_min(1)[..., None];m_f = torch.stack([cm[f][okc[f]].mean(0) for f in range(a.n_an)])
    # twins: same records, tracked A or B (both updated at chunk t with different colours, B also in chunk 0), no switch
    def twins(t, n, seed):
        rng = np.random.default_rng(seed);A_, B_, D_ = [], [], []
        ta = dict(task_args(a), p_switch=0.)
        while len(A_) < n:
            st = make_stream(rng, **ta);fa = st['focus'][0];an_t = {x for x, c, _ in st['chunks'][t][1]};an_0 = {x for x, c, _ in st['chunks'][0][1]}
            cand = [x for x in an_t & an_0 if x != fa]
            if fa not in an_t or not cand:continue
            fb = int(rng.choice(cand));col = {};gb = []
            for h, Sx in st['chunks']:
                for x, c, _ in Sx:col[x] = c
                gb.append(col.get(fb, -1))
            if gb[t] == st['gold'][t] or gb[0] < 0:continue
            tw = dict(st, chunks=[(f'Keep track of the {ANIMALS[fb]}.', st['chunks'][0][1])]+st['chunks'][1:], focus=[fb]*K, gold=gb)
            absent = [x for x in range(a.n_an) if x not in an_t and x not in (fa, fb)]
            A_.append(rendered(fm, st));B_.append(rendered(fm, tw));D_.append(int(rng.choice(absent)))
        return A_, B_, D_
    E = {}
    g = torch.Generator(device='cpu').manual_seed(0)
    for t in (2, 3, 4):
        A_, B_, D_ = twins(t, a.n_pairs, 99+t)
        fa = torch.tensor([x['focus'][0] for x in A_], device=DEV);fb = torch.tensor([x['focus'][0] for x in B_], device=DEV);fd = torch.tensor(D_, device=DEV)
        gA = torch.tensor([x['gold'] for x in A_], device=DEV);gB = torch.tensor([x['gold'] for x in B_], device=DEV)
        dAB = m_f[fb]-m_f[fa];rnd = F.normalize(torch.randn(dAB.shape, generator=g), dim=-1).to(DEV)*dAB.norm(dim=-1, keepdim=True)
        def ed(delta, after=False, tt=t):
            def fn(k, z, sl):
                return loop.sph(F.normalize(z, dim=-1)+delta[sl]) if k == tt else z
            return dict(za_fn=fn) if after else dict(z_fn=fn)
        conds = {'none':{}, 'focus_A_to_B':ed(dAB), 'focus_A_to_B_after_read':ed(dAB, True), 'focus_A_to_B_x0.5':ed(0.5*dAB),
                 'random_same_size':ed(rnd), 'toward_absent_D':ed(m_f[fd]-m_f[fa])}
        for c, kw in conds.items():
            R = run(A_, **kw);p = R['logits'].argmax(-1)
            E.setdefault(c, []).append(dict(answers_A_new=float((p[:, t] == gA[:, t]).float().mean()), answers_B_new=float((p[:, t] == gB[:, t]).float().mean()),
                                            answers_A_stored=float((p[:, t] == gA[:, t-1]).float().mean()),
                                            next_answers_A=float((p[:, t+1] == gA[:, t+1]).float().mean()), next_answers_B=float((p[:, t+1] == gB[:, t+1]).float().mean())))
        if t == 3:twin3 = (A_, B_, ed(dAB))
    res['focus_edits'] = {c:{k:float(np.mean([x[k] for x in v])) for k in v[0]} for c, v in E.items()}
    for c, v in res['focus_edits'].items():log(tag, 'edit', c, json.dumps({k:round(x, 3) for k, x in v.items()}))
    # the model's own attention while it reads chunk 3 of the twin streams: from the last token to A's record, B's record,
    # the other records, the placeholders (mean over heads; layers L_w+1..L_r)
    set_eager(fm);pr = AttnProbe(fm);A_, B_, kwAB = twins3 = twin3;t = 3;n_p = min(len(A_), a.n_probe)
    def masses(cond_kw, S_=None, tt=3):
        S_ = A_ if S_ is None else S_;pr.store, pr.k = [], tt;Rs = []
        for i in range(0, n_p, bs):
            sl = slice(i, min(i+bs, n_p));kw = {}
            if 'z_fn' in cond_kw:kw['z_fn'] = (lambda sl_:(lambda k, z:cond_kw['z_fn'](k, z, sl_)))(sl)
            if 'nomem' in cond_kw:kw['z_fn'] = lambda k, z:loop.init(z.shape[0])
            if 'nowrite' in cond_kw:kw['w_none_at'] = tt
            pr.store = [];episode(fm, loop, S_[sl], probe=pr, **kw);Rs.append(list(pr.store))
        out_ = {'A':[], 'B':[], 'other':[], 'placeholders':[]};per_layer = {}
        for bi, store in enumerate(Rs):
            i0 = bi*bs
            for (layer, w) in store:
                for j in range(w.shape[0]):
                    st = S_[i0+j];ln = len(fm.pre)+NPH+len(st['ids'][tt]);row = w[j, ln-1];base = fm.P0+NPH
                    fa_, fb_ = A_[i0+j]['focus'][0], B_[i0+j]['focus'][0];m = {'A':0., 'B':0., 'other':0.}
                    for (an, c, s0, s1) in st['spans'][tt]:
                        key = 'A' if an == fa_ else ('B' if an == fb_ else 'other');m[key] += float(row[base+s0:base+s1].sum())
                    m['other'] /= max(1, len(st['spans'][tt])-2);m['placeholders'] = float(row[fm.P0:fm.P0+NPH].sum())
                    for k_ in out_:out_[k_].append(m[k_]);per_layer.setdefault(layer, {'A':[], 'B':[]})
                    per_layer[layer]['A'].append(m['A']);per_layer[layer]['B'].append(m['B'])
        return dict({k_:float(np.mean(v)) for k_, v in out_.items()}, per_layer={int(l):{k_:float(np.mean(v)) for k_, v in d.items()} for l, d in per_layer.items()})
    res['attention_chunk3'] = {'own':masses({}), 'focus_A_to_B':masses(kwAB), 'no_memory':masses({'nomem':1}), 'no_write':masses({'nowrite':1})}
    for c, v in res['attention_chunk3'].items():log(tag, 'attention', c, json.dumps({k:round(x, 4) for k, x in v.items() if k != 'per_layer'}))
    json.dump(res, open(out/f'eval_{tag}.json', 'w'), indent=1);log(tag, 'evaluated')
    # chunk 0, with the instruction in view (A's stream says A, B's twin says B; both records are in the chunk): does the
    # learned write change the model's own attention to the instructed animal's record, compared with writing nothing?
    try:
        res['attention_chunk0'] = {f'{nm}|{wn}':masses(kw, S_, 0) for nm, S_ in (('instruction_A', A_), ('instruction_B', B_))
                                   for wn, kw in (('learned_write', {}), ('nothing_written', {'nowrite':1}))}
        for c, v in res['attention_chunk0'].items():log(tag, 'attention chunk 0', c, json.dumps({k:round(x, 4) for k, x in v.items() if k != 'per_layer'}))
        json.dump(res, open(out/f'eval_{tag}.json', 'w'), indent=1)
    except Exception as e:log(tag, 'chunk-0 probe failed', repr(e))


def main():
    p = argparse.ArgumentParser();p.add_argument('stage', choices=['train', 'eval']);p.add_argument('--out', required=True)
    p.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct');p.add_argument('--local_only', type=int, default=0)
    p.add_argument('--lw', type=int, default=8);p.add_argument('--lr', type=int, default=16);p.add_argument('--dz', type=int, default=32)
    p.add_argument('--steps', type=int, default=1500);p.add_argument('--bs', type=int, default=16);p.add_argument('--lrate', type=float, default=2e-3)
    p.add_argument('--seed', type=int, default=0);p.add_argument('--tag', default=None)
    p.add_argument('--n_an', type=int, default=10);p.add_argument('--n_col', type=int, default=10);p.add_argument('--K', type=int, default=6)
    p.add_argument('--n_upd', type=int, default=4);p.add_argument('--p_switch', type=float, default=0.3);p.add_argument('--p_focus', type=float, default=0.5)
    p.add_argument('--n_eval', type=int, default=512);p.add_argument('--n_pairs', type=int, default=256);p.add_argument('--n_probe', type=int, default=128)
    p.add_argument('--max_minutes', type=float, default=0, help='training time budget (0: none)')
    p.add_argument('--init', default='', help='continue from a loop checkpoint');p.add_argument('--suffix', default='', help='tag suffix')
    a = p.parse_args()
    if a.stage == 'train':train(a)
    else:evaluate(a)


if __name__ == '__main__':
    main()
