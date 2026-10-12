"""Huginn-0125 natural recurrence (protocol section 6). Same contexts as Experiment 2 (A / B / N evidence variants).
Readout: coda exit lens (ln_f -> coda -> ln_f -> lm_head applied to the latent state after recurrence k); the model was
trained with randomly sampled recurrence counts, so intermediate states are in the coda's training distribution.
    python huginn.py --n 100 --K 64"""
import argparse, json, time, pathlib
import numpy as np, torch
from common import OUT, log, atomic_npz, atomic_json, seed_all
from e2 import make_contexts

REPO, REV = 'tomg-group-umd/huginn-0125', 'bb6621b65e90b6a4b9b29ef88dc83866d450470c'


class Huginn:
    def __init__(self, dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
        kw = dict(revision=REV, trust_remote_code=True)
        try:self.m = AutoModelForCausalLM.from_pretrained(REPO, dtype=dtype, **kw)
        except TypeError:self.m = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=dtype, **kw)
        self.m = self.m.cuda().eval()
        for p in self.m.parameters():p.requires_grad_(False)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self.d = self.m.config.n_embd

    def fmt(self, text):
        msgs = [{'role':'system', 'content':'You are a helpful assistant.'}, {'role':'user', 'content':text}]
        try:return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:return f'{text}\nAnswer:'

    def encode(self, texts):
        ids = [self.tok(self.fmt(t), add_special_tokens=False)['input_ids'] for t in texts]
        T = max(map(len, ids));arr = torch.full((len(ids), T), self.pad, dtype=torch.long)
        for i, x in enumerate(ids):arr[i, :len(x)] = torch.tensor(x)              # right padding: causal attention never sees pads
        return arr.cuda(), torch.tensor([len(x)-1 for x in ids]).cuda()

    @torch.no_grad()
    def prelude(self, ids):
        m = self.m;T = ids.shape[1];fc = m.freqs_cis[:, :T]
        e = m.transformer.wte(ids)*m.emb_scale;bi = torch.tensor(-1, dtype=torch.long)
        for blk in m.transformer.prelude:
            bi += 1;e = blk(e, fc, bi, None, None)
        return e, fc

    @torch.no_grad()
    def step(self, x, e, fc, k):
        bi = torch.tensor(self.m.config.n_layers_in_prelude-1+4*k, dtype=torch.long)
        x, _ = self.m.core_block_forward(x, e, fc, None, None, bi, k)
        return x

    @torch.no_grad()
    def exit(self, x, fc, last):
        m = self.m;y = m.transformer.ln_f(x);bi = torch.tensor(0, dtype=torch.long)
        for blk in m.transformer.coda:
            bi -= 1;y = blk(y, fc, bi, None, None)
        y = m.transformer.ln_f(y);r = torch.arange(x.shape[0], device=x.device)
        return m.lm_head(y[r, last]).float()

    def init(self, e, seed):
        g = torch.Generator(device='cuda');g.manual_seed(seed)
        std = self.m.config.init_values['std']
        x = torch.empty(e.shape, device='cuda', dtype=torch.float32)
        x = torch.nn.init.trunc_normal_(x, 0., std, -3*std, 3*std, generator=g)*self.m.emb_scale   # seeded: twins share it
        return x.to(e.dtype)

    def variant_ids(self, w):
        ids = set()
        for v in (w, ' '+w, w.capitalize(), ' '+w.capitalize()):
            t = self.tok.encode(v, add_special_tokens=False)
            if t:ids.add(t[0])
        return sorted(ids)

    def logodds(self, logits, cands):
        lp = torch.log_softmax(logits, -1);out = []
        for b, (a, c) in enumerate(cands):
            ia, ic = self.variant_ids(a), self.variant_ids(c);sa, sc = set(ia), set(ic)
            ia = [i for i in ia if i not in sc];ic = [i for i in ic if i not in sa]
            out.append((torch.logsumexp(lp[b, ia], 0)-torch.logsumexp(lp[b, ic], 0)).item() if ia and ic else np.nan)
        return np.array(out, np.float32)


@torch.no_grad()
def trajectory(H, texts, cands, K, seed=0, edit=None, save_state=False):
    """Run the recurrence to K steps; at every step record exit-lens log-odds, top token, decision-state norm/change.
    edit(k, x, last) -> x may modify the latent state before step k (perturbation / transplant)."""
    ids, last = H.encode(texts);e, fc = H.prelude(ids);x = H.init(e, seed);B = len(texts)
    r = torch.arange(B, device='cuda')
    lo = np.zeros((K+1, B), np.float32);top = np.zeros((K+1, B), np.int32);dlt = np.zeros((K, B), np.float32)
    xs = np.zeros((K+1, B, H.d), np.float16) if save_state else None
    lg = H.exit(x, fc, last);lo[0] = H.logodds(lg, cands);top[0] = lg.argmax(-1).cpu().numpy()
    if save_state:xs[0] = x[r, last].float().cpu().numpy()
    for k in range(K):
        if edit is not None:x = edit(k, x, last, e)
        xn = H.step(x, e, fc, k)
        dlt[k] = ((xn[r, last]-x[r, last]).float().norm(dim=-1)/x[r, last].float().norm(dim=-1).clamp_min(1e-6)).cpu().numpy()
        x = xn;lg = H.exit(x, fc, last);lo[k+1] = H.logodds(lg, cands);top[k+1] = lg.argmax(-1).cpu().numpy()
        if save_state:xs[k+1] = x[r, last].float().cpu().numpy()
    return dict(lo=lo, top=top, dlt=dlt, xs=xs, x=x, last=last, e=e, fc=fc)


def main():
    p = argparse.ArgumentParser();p.add_argument('--n', type=int, default=100);p.add_argument('--K', type=int, default=64)
    p.add_argument('--bs', type=int, default=48);p.add_argument('--tag', default='main')
    a = p.parse_args();out = OUT/'huginn'/a.tag;out.mkdir(parents=True, exist_ok=True)
    seed_all(0);H = Huginn();ctxs = make_contexts(a.n);K = a.K
    atomic_json(out/'contexts.json', ctxs)
    atomic_json(out/'provenance.json', dict(model=REPO, revision=REV, dtype='bfloat16', K=K, readout='coda exit lens',
                                            example_prompt=H.fmt(ctxs[0]['text'])))
    texts = [c['text'] for c in ctxs];cands = [c['cands'] for c in ctxs];N = len(ctxs)
    # 1. natural trajectories, two random initial states
    if not (out/'natural.npz').exists():
        R = {s:dict(lo=[], top=[], dlt=[], xs=[]) for s in (0, 1)}
        for s in (0, 1):
            for i in range(0, N, a.bs):
                t = trajectory(H, texts[i:i+a.bs], cands[i:i+a.bs], K, seed=1000+s, save_state=(s == 0))
                for k in ('lo', 'top', 'dlt'):R[s][k].append(t[k])
                if s == 0:R[s]['xs'].append(t['xs'])
            log('natural seed', s, 'done')
        atomic_npz(out/'natural.npz', lo0=np.concatenate(R[0]['lo'], 1), lo1=np.concatenate(R[1]['lo'], 1),
                   top0=np.concatenate(R[0]['top'], 1), dlt0=np.concatenate(R[0]['dlt'], 1), dlt1=np.concatenate(R[1]['dlt'], 1),
                   xs0=np.concatenate(R[0]['xs'], 1))
    # 2. memory tests on F2/F3 scenarios: decision-state transplant from A or B into N, and full-state swap A <-> B
    by = {}
    for i, c in enumerate(ctxs):by.setdefault((c['fam'], c['sid']), {})[c['tag']] = i
    trip = [(d['A'], d['B'], d['N']) for d in by.values() if all(k in d for k in 'ABN')]
    if not (out/'memory.npz').exists():
        res = {}
        for k0 in [k for k in (4, 16, 32) if k < K]:
            LA, LB, SW_A, SW_B, PA = [], [], [], [], []
            for i in range(0, len(trip), a.bs//2):
                tt = trip[i:i+a.bs//2]
                src = trajectory(H, [texts[x] for t in tt for x in t[:2]], [cands[x] for t in tt for x in t[:2]], k0,
                                 seed=1000, save_state=False)
                xa = src['x'][0::2];xb = src['x'][1::2];la = src['last'][0::2];lb = src['last'][1::2]
                for which, store in ((0, LA), (1, LB)):
                    xsrc, lsrc = (xa, la) if which == 0 else (xb, lb)

                    def tr_edit(k, x, last, e, xsrc=xsrc, lsrc=lsrc):
                        if k == k0:
                            x = x.clone();r = torch.arange(x.shape[0], device=x.device);x[r, last] = xsrc[r, lsrc]
                        return x
                    t = trajectory(H, [texts[t_[2]] for t_ in tt], [cands[t_[2]] for t_ in tt], K, seed=1000, edit=tr_edit)
                    store.append(t['lo'])
                # full-state swap between A and B (same length only), keeping each run's own input injection
                same = [j for j, t_ in enumerate(tt) if src['last'][2*j] == src['last'][2*j+1]]
                if same:
                    sel = [tt[j] for j in same];xa_s, xb_s = xa[same], xb[same]
                    for which, store in ((0, SW_A), (1, SW_B)):
                        tgt = [s_[which] for s_ in sel];xsrc = xb_s if which == 0 else xa_s

                        def sw_edit(k, x, last, e, xsrc=xsrc):
                            if k == k0:
                                T = min(x.shape[1], xsrc.shape[1]);x = x.clone();x[:, :T] = xsrc[:, :T]
                            return x
                        t = trajectory(H, [texts[x_] for x_ in tgt], [cands[x_] for x_ in tgt], K, seed=1000, edit=sw_edit)
                        store.append(np.pad(t['lo'], ((0, 0), (0, 0))) if True else None)
            res[f'k{k0}_N_from_A'] = np.concatenate(LA, 1);res[f'k{k0}_N_from_B'] = np.concatenate(LB, 1)
            if SW_A:res[f'k{k0}_A_with_B_state'] = np.concatenate(SW_A, 1);res[f'k{k0}_B_with_A_state'] = np.concatenate(SW_B, 1)
            log('memory k0', k0, 'gap at k0+1, +8, K:', [float(np.nanmean(res[f'k{k0}_N_from_A'][t]-res[f'k{k0}_N_from_B'][t]))
                                                          for t in (min(K, k0+1), min(K, k0+8), K)])
        atomic_npz(out/'memory.npz', trip=np.array(trip), **res)
    # 3. sensitivity: 1 % noise on the decision state at k0 = 16, twin divergence
    if not (out/'sensitivity.npz').exists():
        D, L = [], []
        for i in range(0, N, a.bs):
            tx, cd = texts[i:i+a.bs], cands[i:i+a.bs]
            base = trajectory(H, tx, cd, K, seed=1000, save_state=True)
            g = torch.Generator(device='cuda');g.manual_seed(7)

            def eps_edit(k, x, last, e):
                if k == min(16, K//2):
                    x = x.clone();r = torch.arange(x.shape[0], device=x.device);v = x[r, last].float()
                    n = torch.randn(v.shape, device=v.device, generator=g);x[r, last] = (v+0.01*v.norm(dim=-1, keepdim=True)*n/n.norm(dim=-1, keepdim=True)).to(x.dtype)
                return x
            pert = trajectory(H, tx, cd, K, seed=1000, edit=eps_edit, save_state=True)
            d = np.linalg.norm(pert['xs'].astype(np.float32)-base['xs'].astype(np.float32), axis=-1)/(np.linalg.norm(base['xs'].astype(np.float32), axis=-1)+1e-6)
            D.append(d);L.append(pert['lo']-base['lo'])
        atomic_npz(out/'sensitivity.npz', dist=np.concatenate(D, 1), dlo=np.concatenate(L, 1))
        log('sensitivity: median relative distance at k=17, 24, 40, K', [float(np.median(np.concatenate(D, 1)[min(t, K)])) for t in (17, 24, 40, K)])
    (out/'DONE').write_text('ok')


if __name__ == '__main__':
    main()
