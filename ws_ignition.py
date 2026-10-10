"""Ignition in the J-space: Dehaene & Naccache's "decisive experiment" on open models.

GNW theory says conscious access is all-or-none. A stimulus at graded strength should enter the workspace with a
threshold-like jump, while earlier processing grows smoothly. At threshold the same stimulus should tip one way or
the other on different trials (a bimodal distribution). The workspace is a bottleneck, so holding one content should
make it harder for another to get in (Dehaene & Naccache 2026, commentary on Gurnee et al.). Gurnee et al. only
tested a blend of two country tokens.

Stimulus: one concept word in a carrier sentence. Its input embedding is e_base + a (e_concept - e_base) + noise,
where e_base is the mean embedding of the dictionary words ("some word") and the noise norm is 0.3 |e_concept -
e_base|. There are 31 strengths a in [0, 1] and 16 noise draws per strength, plus two noise-free anchors (a = 0, 1).
Readouts at every layer, at the concept's own position and at the end of the sentence (persistence):
  lin    share of the way from the a=0 to the a=1 residual (full residual, linear)
  J      the same share measured only along the concept's unit J-lens atom at that layer (linear)
  rank   the concept's rank in the J-lens over the dictionary (0 = top)
Bottleneck: the prefix "Keep this word in mind: X." holds an unrelated concept X (or the neutral e_base, nothing held)
before the carrier. The concept is graded on 21 strengths x 8 draws.

Predictions, fixed before the run:
  P1  steepness of the J share against strength jumps at the workspace onset (~0.6 depth) over earlier layers
  P2  at each layer's own threshold, the distribution over noise draws is bimodal in workspace layers (bimodality
      coefficient > 0.555, two-Gaussian BIC wins) and unimodal earlier
  P3  at the end of the sentence the concept survives all-or-none (steep), not in proportion to strength
  P4  holding X raises the threshold of the concept in workspace layers (not earlier), and the concept's entry
      lowers X's readout at the end of the sentence (eviction)

    python ws_ignition.py --model qwen3-4b --out runs/ws/qwen3-4b/ignition
"""
import argparse, math, os, time
import numpy as np
import torch

from ws_common import Lab, band, log, save_json

CONCEPTS = ['elephant', 'violin', 'volcano', 'pizza', 'castle', 'ocean', 'guitar', 'tiger', 'diamond', 'rocket',
            'spider', 'banana', 'snow', 'fire', 'coffee', 'piano', 'dragon', 'forest', 'bridge', 'horse', 'apple',
            'cloud', 'moon', 'train', 'camera', 'flower', 'shark', 'robot', 'candle', 'mountain', 'chocolate', 'wolf',
            'desert', 'lemon', 'clock', 'river']
CARRIERS = ['She whispered one word: {}. Then she closed the door and went to sleep.',
            'The picture on the wall showed the {}. We looked at it for a while and then left.',
            'In my dream last night I saw the {}. When I woke up, the room was quiet.',
            'He wrote a single word on the card: {}. Then he put the card in his pocket.',
            'Behind the old house we finally found the {}. After that we walked home for dinner.',
            'The answer to the riddle was: {}. Everyone laughed and the game went on.',
            'As a child I was obsessed with the {}. Now I am older and live in the city.',
            'On the screen appeared the word: {}. The meeting continued as planned.']
HOLD = 'Keep this word in mind: {}. '


def pieces(lab, text):
    """Token ids before and after the slot '{}' (the slot itself is one token, supplied as an embedding)."""
    pre, post = text.split('{}')
    ids = lambda s:lab.tok(s, add_special_tokens=False)['input_ids'] if s else []
    return ids(pre.rstrip(' ')), ids(post)


class Reader:
    """Forward batches with substituted slot embeddings; read lin/J shares and J ranks at chosen positions."""

    def __init__(self, lab, cids):
        self.lab = lab;self.cids = cids;self.L = [l for l in range(lab.n)]
        self.atoms = {l:lab.token_atoms(l, cids) for l in lab.lens_layers}             # (n_concepts, d) per layer
        W = lab.model.get_output_embeddings().weight
        self.cvec = ((W[torch.tensor(cids, device=lab.dev)].float()-lab.wbar)*lab.gamma)  # for the concept's own score

    @torch.no_grad()
    def run(self, ids, slot_embs, read_pos, which):
        """ids (T,) token ids with placeholders at slot positions; slot_embs {pos: (B, d)}; read_pos list of
        positions; which (B,) concept index per row. Returns proj J (B, P, n) float, ranks (B, P, n) int16,
        residual views at read positions for the lin share (n, B, P, d) float16."""
        lab = self.lab;B = next(iter(slot_embs.values())).shape[0]
        E = lab.model.get_input_embeddings()(torch.tensor(ids, device=lab.dev))[None].expand(B, -1, -1).clone()
        for p, e in slot_embs.items():E[:, p] = e.to(E.dtype)
        out = lab.model(inputs_embeds=E, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        hs = out.hidden_states;P = len(read_pos);n = lab.n
        projJ = torch.full((B, P, n), float('nan'), device=lab.dev);rank = torch.full((B, P, n), -1, dtype=torch.int16, device=lab.dev)
        H = torch.stack([hs[l+1][:, read_pos] for l in range(n)])                    # (n, B, P, d)
        for l in lab.lens_layers:
            h = H[l].float()
            projJ[:, :, l] = (h*self.atoms[l][which][:, None]).sum(-1)
            jh = (h.half()@lab.J[l].T).float()                                         # J h, (B, P, d)
            sd = jh.half()@lab.Wd.T                                                    # dictionary scores
            sc = (jh*self.cvec[which][:, None]).sum(-1)                                # the concept's own score
            rank[:, :, l] = (sd.float() > sc[:, :, None]).sum(-1).to(torch.int16)
        return projJ, rank, H.half()


def shares(x, x0, x1):
    """(x - x0)/(x1 - x0) elementwise, nan where the anchors coincide."""
    den = x1-x0
    return torch.where(den.abs() > 1e-6, (x-x0)/den, torch.full_like(x, float('nan')))


def lin_share(H, H0, H1):
    """Share along the anchor line of full residuals: H (n, B, P, d), anchors (n, 1, P, d)."""
    v = (H1-H0).float();num = ((H.float()-H0.float())*v).sum(-1);den = (v*v).sum(-1)
    return (num/den.clamp_min(1e-6)).permute(1, 2, 0)                                # (B, P, n)


def bimodality(x):
    """Sarle's bimodality coefficient and the BIC gain of two Gaussians over one (EM, 1-D)."""
    x = np.asarray(x, float);x = x[np.isfinite(x)];n = len(x)
    if n < 8 or x.std() < 1e-9:return float('nan'), float('nan')
    m = x-x.mean();s2 = (m**2).mean();g = (m**3).mean()/s2**1.5;k = (m**4).mean()/s2**2-3
    bc = (g**2+1)/(k+3*(n-1)**2/((n-2)*(n-3)))
    def ll1():
        return -.5*n*(np.log(2*np.pi*s2)+1)
    mu = np.percentile(x, [25, 75]).astype(float);sd = np.array([x.std(), x.std()])/2+1e-6;w = np.array([.5, .5])
    for _ in range(200):
        p = w*np.exp(-.5*((x[:, None]-mu)/sd)**2)/sd;r = p/p.sum(1, keepdims=True).clip(1e-300)
        nk = r.sum(0)+1e-9;w = nk/n;mu = (r*x[:, None]).sum(0)/nk;sd = np.sqrt((r*(x[:, None]-mu)**2).sum(0)/nk)+1e-6
    ll2 = np.log((w*np.exp(-.5*((x[:, None]-mu)/sd)**2)/(sd*np.sqrt(2*np.pi))).sum(1).clip(1e-300)).sum()
    dbic = (2*np.log(n)-2*ll1())-(5*np.log(n)-2*ll2)                                 # > 0: two components win
    return float(bc), float(dbic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True);ap.add_argument('--out', required=True);ap.add_argument('--vocab', default='english')
    ap.add_argument('--n_alpha', type=int, default=31);ap.add_argument('--n_noise', type=int, default=16)
    ap.add_argument('--noise', type=float, default=.3);ap.add_argument('--bn_alpha', type=int, default=21)
    ap.add_argument('--bn_noise', type=int, default=8);ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--concepts', type=int, default=24, help='use the first N single-token concepts')
    ap.add_argument('--carriers', type=int, default=0);ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args();os.makedirs(a.out, exist_ok=True);t0 = time.time()
    lab = Lab(a.model, vocab=a.vocab)
    words = [w for w, i in zip(CONCEPTS, lab.word_ids(CONCEPTS)) if i is not None]
    if a.concepts:words = words[:a.concepts]
    cids = [i for i in lab.word_ids(words)];carriers = CARRIERS[:a.carriers] if a.carriers else CARRIERS
    log(f'{len(words)} single-token concepts, {len(carriers)} carriers')
    R = Reader(lab, cids);emb = lab.model.get_input_embeddings().weight
    e_base = emb[lab.dict_ids].float().mean(0);E_c = emb[torch.tensor(cids, device=lab.dev)].float()
    g = torch.Generator(device=lab.dev);g.manual_seed(a.seed);n = lab.n;d = lab.d
    alphas = torch.linspace(0, 1, a.n_alpha, device=lab.dev);rows = []

    def stim(ci, al, k):
        """k noisy slot embeddings of concept ci at strength al (al tensor of len k)."""
        diff = E_c[ci]-e_base
        eps = torch.randn(k, d, generator=g, device=lab.dev);eps = eps/eps.norm(dim=-1, keepdim=True)*a.noise*diff.norm()
        return e_base+al[:, None]*diff+eps

    # ---------------- main design: graded strength, many noise draws ----------------
    shape = (len(carriers), len(words), a.n_alpha, a.n_noise)
    J_sh = np.full(shape+(2, n), np.nan, np.float16);L_sh = np.full(shape+(2, n), np.nan, np.float16)
    RK = np.full(shape+(2, n), -1, np.int16)
    per = max(1, a.batch//(a.n_alpha*a.n_noise+2))                                   # concepts per batch
    for ki, car in enumerate(carriers):
        pre, post = pieces(lab, car);ids = pre+[0]+post;slot = len(pre);read = [slot, len(ids)-1]
        for c0 in range(0, len(words), per):
            cs = list(range(c0, min(len(words), c0+per)));embs, which = [], []
            for ci in cs:
                al = alphas.repeat_interleave(a.n_noise)
                embs += [e_base[None], E_c[ci][None], stim(ci, al, len(al))];which += [ci]*(len(al)+2)
            pj, rk, H = R.run(ids, {slot:torch.cat(embs)}, read, torch.tensor(which, device=lab.dev))
            m = a.n_alpha*a.n_noise+2
            for j, ci in enumerate(cs):
                b = slice(j*m, (j+1)*m)
                Jb = shares(pj[b][2:], pj[b][0:1], pj[b][1:2]);Lb = lin_share(H[:, b][:, 2:], H[:, b][:, 0:1], H[:, b][:, 1:2])
                J_sh[ki, ci] = Jb.reshape(a.n_alpha, a.n_noise, 2, n).cpu().numpy();L_sh[ki, ci] = Lb.reshape(a.n_alpha, a.n_noise, 2, n).cpu().numpy()
                RK[ki, ci] = rk[b][2:].reshape(a.n_alpha, a.n_noise, 2, n).cpu().numpy()
        log(f'carrier {ki+1}/{len(carriers)} done ({time.time()-t0:.0f} s)')

    # ---------------- bottleneck: hold X, grade the concept ----------------
    bal = torch.linspace(0, 1, a.bn_alpha, device=lab.dev);perm = np.roll(np.arange(len(words)), len(words)//2)  # X != concept
    shapeb = (2, len(carriers), len(words), a.bn_alpha, a.bn_noise)
    BJ = np.full(shapeb+(2, n), np.nan, np.float16);BX = np.full(shapeb+(n,), np.nan, np.float16)
    per = max(1, a.batch//(a.bn_alpha*a.bn_noise+2))
    hpre, hpost = pieces(lab, HOLD)
    for held in (0, 1):
        for ki, car in enumerate(carriers):
            pre, post = pieces(lab, car)
            ids = hpre+[0]+hpost+pre+[0]+post;xs = len(hpre);slot = len(hpre)+1+len(hpost)+len(pre);read = [slot, len(ids)-1]
            for c0 in range(0, len(words), per):
                cs = list(range(c0, min(len(words), c0+per)));embs, xe, which, xw = [], [], [], []
                for ci in cs:
                    al = bal.repeat_interleave(a.bn_noise);k = len(al)+2;xi = int(perm[ci])
                    embs += [e_base[None], E_c[ci][None], stim(ci, al, len(al))];which += [ci]*k;xw += [xi]*k
                    xe.append((E_c[xi] if held else e_base)[None].expand(k, -1))
                pj, rk, H = R.run(ids, {slot:torch.cat(embs), xs:torch.cat(xe)}, read, torch.tensor(which, device=lab.dev))
                # X's readout at the end of the sentence: projection on X's own J-lens atom
                px = torch.full((H.shape[1], n), float('nan'), device=lab.dev)
                for l in lab.lens_layers:px[:, l] = (H[l][:, 1].float()*R.atoms[l][torch.tensor(xw, device=lab.dev)]).sum(-1)
                m = a.bn_alpha*a.bn_noise+2
                for j, ci in enumerate(cs):
                    b = slice(j*m, (j+1)*m)
                    BJ[held, ki, ci] = shares(pj[b][2:], pj[b][0:1], pj[b][1:2]).reshape(a.bn_alpha, a.bn_noise, 2, n).cpu().numpy()
                    BX[held, ki, ci] = px[b][2:].reshape(a.bn_alpha, a.bn_noise, n).cpu().numpy()
        log(f'bottleneck held={held} done ({time.time()-t0:.0f} s)')
    np.savez_compressed(os.path.join(a.out, 'ignition.npz'), J=J_sh, lin=L_sh, rank=RK, bn_J=BJ, bn_X=BX,
                        alphas=alphas.cpu().numpy(), bn_alphas=bal.cpu().numpy())

    # ---------------- analysis ----------------
    al = alphas.cpu().numpy();lens = lab.lens_layers;res = dict(model=a.model, n_layers=n, band=band(n), words=words,
                                                                carriers=carriers, noise=a.noise, layers={})

    def curve_stats(S, grid=None):
        """S (units, alpha, noise): per unit (nan where the readout does not respond) threshold (0.5 crossing of the
        mean curve), max-step fraction (share of the whole rise taken in one strength step) and 10-90 width."""
        grid = al if grid is None else grid;M = np.nanmean(S, 2);thr, step, width = [], [], []
        for u in M:
            if not np.isfinite(u).all() or u[-1]-u[0] < .2:thr.append(np.nan);step.append(np.nan);width.append(np.nan);continue
            v = (u-u[0])/(u[-1]-u[0]);step.append(float(np.max(np.diff(np.clip(np.maximum.accumulate(v), 0, 1)))))
            c = lambda q:np.interp(q, np.maximum.accumulate(v), grid)
            thr.append(float(c(.5)));width.append(float(c(.9)-c(.1)))
        return np.array(thr), np.array(step), np.array(width)

    med = lambda x:float(np.nanmedian(x)) if np.isfinite(x).any() else None

    for l in range(n):
        out = {}
        for name, A in (('J', J_sh), ('lin', L_sh)):
            if name == 'J' and l not in lens:continue
            for pi, pos in enumerate(('slot', 'end')):
                S = A[..., pi, l].astype(np.float32).reshape(-1, a.n_alpha, a.n_noise)
                thr, step, width = curve_stats(S)
                # bimodality at each unit's own threshold
                pooled = []
                for u, t in zip(S, [None]*len(S)):
                    mu = np.nanmean(u, 1)
                    if not np.isfinite(mu).all() or mu[-1]-mu[0] < .2:continue
                    v = (mu-mu[0])/(mu[-1]-mu[0]);ia = int(np.argmin(np.abs(v-.5)))
                    pooled += list((u[ia]-mu[0])/(mu[-1]-mu[0]))
                bc, dbic = bimodality(pooled)
                out[f'{name}_{pos}'] = dict(units=int(np.isfinite(thr).sum()), thr=med(thr), step=med(step), width=med(width),
                                            bc=bc, dbic=dbic, hist=np.histogram(np.clip(pooled, -.5, 1.5), 20, (-.5, 1.5))[0].tolist() if pooled else [])
        if l in lens:
            rk = RK[..., 0, l].reshape(-1, a.n_alpha, a.n_noise).astype(np.float32)
            out['top10_at_full'] = float((rk[:, -1] < 10).mean());out['top10_at_zero'] = float((rk[:, 0] < 10).mean())
            # bottleneck: threshold of the concept with X held vs nothing held; X's end-of-sentence readout vs strength
            bg = BJ.shape[3];bal_np = bal.cpu().numpy()
            t0_, s0_, _ = curve_stats(BJ[0, ..., 0, l].astype(np.float32).reshape(-1, bg, a.bn_noise), bal_np)
            t1_, s1_, _ = curve_stats(BJ[1, ..., 0, l].astype(np.float32).reshape(-1, bg, a.bn_noise), bal_np)
            ok = np.isfinite(t0_) & np.isfinite(t1_);dt = t1_[ok]-t0_[ok]
            Xu = np.nanmean(BX[1, ..., l].astype(np.float32).reshape(-1, bg, a.bn_noise), 2)       # (units, strength)
            dx = Xu[:, -3:].mean(1)-Xu[:, :3].mean(1)
            out['bottleneck'] = dict(thr_free=med(t0_), thr_held=med(t1_), units=int(ok.sum()),
                                     thr_shift=float(dt.mean()) if ok.any() else None,
                                     thr_shift_se=float(dt.std(ddof=1)/np.sqrt(ok.sum())) if ok.sum() > 1 else None,
                                     step_free=med(s0_), step_held=med(s1_),
                                     X_end_low=float(np.nanmean(Xu[:, :3])), X_end_high=float(np.nanmean(Xu[:, -3:])),
                                     X_drop=float(np.nanmean(dx)), X_drop_se=float(np.nanstd(dx, ddof=1)/np.sqrt(np.isfinite(dx).sum())))
        res['layers'][l] = out
    save_json(res, os.path.join(a.out, 'ignition.json'))
    log(f'saved {a.out} in {time.time()-t0:.0f} s')
    print(f"\n{'layer':>5} {'depth':>5} | J slot: thr step width  BC  dBIC | J end: thr step  BC | lin slot: step BC | bottleneck thr free->held, X end low->high")
    for l in range(n):
        o = res['layers'][l];js, je, ls = o.get('J_slot', {}), o.get('J_end', {}), o.get('lin_slot', {})
        f = lambda v, p='.2f':format(v, p) if isinstance(v, float) and np.isfinite(v) else '  -  '
        bn = o.get('bottleneck', {})
        print(f"{l:5d} {l/n:5.2f} | {f(js.get('thr'))} {f(js.get('step'))} {f(js.get('width'))} {f(js.get('bc'))} {f(js.get('dbic'), '.0f'):>5} |"
              f" {f(je.get('thr'))} {f(je.get('step'))} {f(je.get('bc'))} | {f(ls.get('step'))} {f(ls.get('bc'))} |"
              f" {f(bn.get('thr_free'))}->{f(bn.get('thr_held'))} (shift {f(bn.get('thr_shift'), '+.3f')}±{f(bn.get('thr_shift_se'), '.3f')}),"
              f" X {f(bn.get('X_end_low'))}->{f(bn.get('X_end_high'))}")


if __name__ == '__main__':
    main()
