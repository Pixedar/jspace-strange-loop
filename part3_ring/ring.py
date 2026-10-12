"""The ring: a latent loop that keeps a model of itself, run beside the controls that would expose it as plain feedback.

Part 1 and 2 fed back a summary of what the model had said. Its "self-model" was about content, and everything passed
through sampled words. The ring changes all three things the review found missing:

Substrate, no words in the loop
    After a short chat prompt that opens the assistant's thinking, the model runs one placeholder token at a time. At
    every step the J-space content it holds at the read layer (~70 % of depth) is added back at the injection layer
    (~50 % of depth) of the next placeholder, so the state passes through the network's own layers and attention and
    never through sampled text (the internality criterion of Lindsey 2025). Minds differ only by a small noise.

A self-model of the process, not of content
    Every mind has its own small online learner P (rank-r linear map with a skip connection) that predicts its next
    J-space content from the current one, trained at every step on the transition that just happened. It models the
    system's own transition function: how it moves, not what it holds.

Closure
    In the ring conditions the self-model's output is written back into the state at the next step, so the model of
    the system is part of the system, and its training target includes the effects of its own output.
      ring_pred   inject the prediction m: what it expects to think next (closure that can fulfil itself)
      ring_err    inject the surprise e = c - m_prev: where it failed to predict itself (the tension in "Inside the
                  Loop": modelling pushes toward stability, the surprise pushes away from what is already modelled)
      frozen_err  ring_err, but the self-model stops learning at --t_freeze
Controls, same norm and schedule, paired noise (rows with the same seed share every random draw):
      free        nothing injected (its self-model still learns: an observer of itself)
      content     inject the current content c: plain feedback, the "banana" control
      yoked_pred  another ring mind's prediction        yoked_err  another ring mind's surprise
      vel         the latest step of motion c - c_prev: momentum with no learning. The self-model starts as
                  "next = current", so ring_err begins as exactly this; a ring must outgrow it
      shuf_err    its own surprise from a random earlier step (self-made, wrong time)
      rand_err    isotropic noise

A ring that is more than feedback must differ from yoked, shuffled and random, and lose the difference when frozen.

    python ring.py --model Qwen/Qwen3-4B --jlens <lens.pt> --grid main --out runs/ring/qwen3-4b
"""
import argparse, json, math, os, time
import numpy as np
import torch

from strange_loop import JSpace, decoder_layers, log

PROMPT = ("<|im_start|>user\nLet your mind wander. You don't need to answer anything, just think.<|im_end|>\n"
          "<|im_start|>assistant\n<think>\n")
PLACE = '…'
GLIMPSE = "\n\nPutting it into words, what I was just thinking about was"
CONDS = ('free', 'content', 'vel', 'ring_pred', 'yoked_pred', 'ring_err', 'yoked_err', 'shuf_err', 'rand_err', 'frozen_err')


class Hooks:
    """Add a vector at the last position of block L_in's output; record the last position of blocks L_in and L_read."""

    def __init__(self, blocks, L_in, L_read):
        self.add = None;self.h_in = None;self.h = None
        blocks[L_in].register_forward_hook(self._in);blocks[L_read].register_forward_hook(self._read)

    def _in(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        self.h_in = h[:, -1].float()
        if self.add is None:return None
        h = torch.cat([h[:, :-1], h[:, -1:]+self.add[:, None].to(h.dtype)], 1)
        return (h,)+tuple(out[1:]) if isinstance(out, tuple) else h

    def _read(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        self.h = h[:, -1].float()


class SelfModels:
    """One online learner per mind: m = c + U V^T c + b (units of the typical content norm), cosine loss."""

    def __init__(self, B, d, r, lr, buf, dev, seed=0):
        g = torch.Generator(device=dev);g.manual_seed(seed)
        self.U = torch.zeros(B, d, r, device=dev, requires_grad=True)
        self.V = (torch.randn(B, d, r, generator=g, device=dev)/math.sqrt(d)).requires_grad_(True)
        self.b = torch.zeros(B, d, device=dev, requires_grad=True)
        self.opt = torch.optim.Adam([self.U, self.V, self.b], lr=lr)
        self.X = torch.zeros(B, buf, d, device=dev);self.Y = torch.zeros(B, buf, d, device=dev);self.n = 0;self.buf = buf
        self.frozen = torch.zeros(B, dtype=torch.bool, device=dev);self.saved = None

    def _f(self, X):                                                     # X (B, n, d)
        return X+torch.einsum('bdr,bnr->bnd', self.U, torch.einsum('bdr,bnd->bnr', self.V, X))+self.b[:, None]

    @torch.no_grad()
    def predict(self, c):
        return self._f(c[:, None])[:, 0]

    def learn(self, c_prev, c):
        """Store the transition c_prev -> c and take one Adam step on the replay of the last `buf` transitions."""
        i = self.n % self.buf;self.X[:, i] = c_prev;self.Y[:, i] = c;self.n += 1
        k = min(self.n, self.buf)
        P = self._f(self.X[:, :k])
        loss = (1-torch.nn.functional.cosine_similarity(P, self.Y[:, :k], dim=-1)).mean(1)    # (B,)
        self.opt.zero_grad();loss.sum().backward();self.opt.step()
        if self.saved is not None:                                       # frozen minds keep their weights exactly
            with torch.no_grad():
                for p, s in zip((self.U, self.V, self.b), self.saved):p[self.frozen] = s
        return loss.detach()

    def freeze(self, mask):
        self.frozen |= mask
        self.saved = [p.detach()[self.frozen].clone() for p in (self.U, self.V, self.b)]


def rows_for(a):
    """Seeds are labelled from --seed0, so runs that are too big for one card can be split into chunks of seeds."""
    rows = []
    if a.grid == 'main':
        for c in a.conds.split(','):
            for s in range(a.seeds):rows.append(dict(cond=c, seed=a.seed0+s, g=a.g, s=a.s if a.s >= 0 else a.g, sigma=a.sigma))
    elif a.grid == 'sweep':                                              # free substrate only: gain x noise
        for g in [float(x) for x in a.gs.split(',')]:
            for sg in [float(x) for x in a.sigmas.split(',')]:
                for s in range(a.seeds):rows.append(dict(cond='free', seed=a.seed0+s, g=g, s=0., sigma=sg))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True);p.add_argument('--jlens', required=True);p.add_argument('--out', required=True)
    p.add_argument('--grid', default='main');p.add_argument('--conds', default=','.join(CONDS));p.add_argument('--seeds', type=int, default=16)
    p.add_argument('--T', type=int, default=600);p.add_argument('--t_freeze', type=int, default=300);p.add_argument('--n_cal', type=int, default=24)
    p.add_argument('--read_frac', type=float, default=.7);p.add_argument('--in_frac', type=float, default=.5)
    p.add_argument('--layer', type=int, default=-1, help='identity lens only: read layer')
    p.add_argument('--g', type=float, default=.2, help='carried state, x residual norm at the injection layer')
    p.add_argument('--s', type=float, default=-1., help='self-model channel (default: = g)')
    p.add_argument('--sigma', type=float, default=.05, help='noise, x residual norm');p.add_argument('--alpha', type=float, default=.5)
    p.add_argument('--gs', default='.1,.2,.4,.8');p.add_argument('--sigmas', default='.05,.15')
    p.add_argument('--k', type=int, default=25);p.add_argument('--vocab', default='english');p.add_argument('--atoms', default='centred')
    p.add_argument('--rank', type=int, default=32);p.add_argument('--lr', type=float, default=3e-3);p.add_argument('--buf', type=int, default=32)
    p.add_argument('--glimpse', type=int, default=24);p.add_argument('--rng', type=int, default=1234)
    p.add_argument('--seed0', type=int, default=0, help='label of the first seed (chunks of one run use different --rng)')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if dev == 'cuda' else torch.float32,
                                                 device_map=dev if dev == 'cuda' else None).eval()
    if dev != 'cuda':model.to(dev)
    for q in model.parameters():q.requires_grad_(False)
    blocks = decoder_layers(model);n = len(blocks);d = model.config.hidden_size
    L_read = a.layer if a.layer >= 0 else int(round(a.read_frac*n));L_in = int(round(a.in_frac*n)) if a.layer < 0 else max(1, int(round(a.layer*a.in_frac/a.read_frac)))
    if a.jlens == 'identity':J = np.eye(d, dtype=np.float32)
    else:
        ck = torch.load(a.jlens, map_location='cpu', weights_only=True);J = ck['J'][L_read].float().numpy();del ck
    js = JSpace(model, tok, J, k=a.k, vocab=a.vocab, center=a.atoms == 'centred')
    hk = Hooks(blocks, L_in, L_read)
    rows = rows_for(a);B = len(rows);S = a.seeds
    log(f'{a.model}: d {d}, {n} layers, inject at {L_in}, read at {L_read} | {len(js.words)} atoms | {B} minds | {dev}')
    seed_of = torch.tensor([r['seed']-a.seed0 for r in rows], device=dev)
    g_ = torch.tensor([r['g'] for r in rows], device=dev);s_ = torch.tensor([r['s'] for r in rows], device=dev)
    sig_ = torch.tensor([r['sigma'] for r in rows], device=dev);cond = [r['cond'] for r in rows]
    idx = {(r['cond'], r['seed']):i for i, r in enumerate(rows)}
    partner = {'yoked_pred':'ring_pred', 'yoked_err':'ring_err'}
    src = torch.tensor([idx.get((partner[r['cond']], a.seed0+(r['seed']-a.seed0+1) % S), i) if r['cond'] in partner else i
                        for i, r in enumerate(rows)], device=dev)
    gen = torch.Generator(device=dev);gen.manual_seed(a.rng)
    unit = lambda v:v/v.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def noise():                                                         # one unit direction per seed, shared by its rows
        return unit(torch.randn(S, d, generator=gen, device=dev))[seed_of]

    place = tok(PLACE, add_special_tokens=False)['input_ids'];assert len(place) == 1, place
    place = torch.tensor([place], device=dev).expand(B, 1)
    ids = torch.tensor([tok(PROMPT, add_special_tokens=False)['input_ids']], device=dev).expand(B, -1)
    t0 = time.time()
    with torch.no_grad():
        hk.add = None;out = model(input_ids=ids, use_cache=True, logits_to_keep=1);cache = out.past_key_values
        # calibration: placeholders with noise only -> the typical read state, the residual norm, the typical workspace
        H, Nin = [], []
        for _ in range(a.n_cal):
            hk.add = None;out = model(input_ids=place, past_key_values=cache, use_cache=True, logits_to_keep=1);cache = out.past_key_values
            H.append(hk.h);Nin.append(hk.h_in.norm(dim=-1))
        Rn = float(torch.cat(Nin).median());H = torch.stack(H)                    # (n_cal, B, d)
        mu = H[a.n_cal//2:].reshape(-1, d).mean(0)
        parts = torch.cat([js.decompose(H[i]-mu)[2] for i in range(a.n_cal//2, a.n_cal)])
        jbar = parts.mean(0);rc = float((parts-jbar).norm(dim=-1).mean())
    log(f'calibrated in {time.time()-t0:.0f} s: residual norm at injection {Rn:.1f} | typical content {rc:.2f}')
    SM = SelfModels(B, d, a.rank, a.lr, a.buf, dev, seed=a.rng)
    T = a.T;K = a.k
    rec = dict(C=np.zeros((T, B, d), np.float16), idx=np.zeros((T, B, K), np.int32), coef=np.zeros((T, B, K), np.float16),
               pred_cos=np.zeros((T, B), np.float32), step_cos=np.zeros((T, B), np.float32), loss=np.zeros((T, B), np.float32),
               sig_cos=np.zeros((T, B), np.float32), err_rel=np.zeros((T, B), np.float32))
    x = torch.zeros(B, d, device=dev);c_prev = None;m_prev = None;sig = torch.zeros(B, d, device=dev)
    E = torch.zeros(T, B, d, device=dev, dtype=torch.float16)                     # surprise history (for shuf_err)
    ring_rows = torch.tensor([c != 'free' for c in cond], device=dev)
    frozen_rows = torch.tensor([c == 'frozen_err' for c in cond], device=dev)
    for t in range(T):
        if t == a.t_freeze and bool(frozen_rows.any()):SM.freeze(frozen_rows)
        nx = x.norm(dim=-1, keepdim=True)/rc
        with torch.no_grad():
            add = (g_*Rn)[:, None]*(x/rc)*torch.clamp(1.5/nx.clamp_min(1e-6), max=1.)
            add = add+(s_*Rn)[:, None]*unit(sig)*ring_rows[:, None]+(sig_*Rn)[:, None]*noise()
            hk.add = add;out = model(input_ids=place, past_key_values=cache, use_cache=True, logits_to_keep=1);cache = out.past_key_values
            jidx, jcoef, part = js.decompose(hk.h-mu)
            c = part-jbar;cn = c/rc
            if c_prev is not None:
                rec['step_cos'][t] = torch.nn.functional.cosine_similarity(c, c_prev, dim=-1).cpu().numpy()
                rec['sig_cos'][t] = torch.nn.functional.cosine_similarity(c, sig, dim=-1).cpu().numpy()
            if m_prev is not None:
                rec['pred_cos'][t] = torch.nn.functional.cosine_similarity(m_prev, cn, dim=-1).cpu().numpy()
        if c_prev is not None:rec['loss'][t] = SM.learn(c_prev/rc, cn).cpu().numpy()
        with torch.no_grad():
            m = SM.predict(cn)
            e = cn-m_prev if m_prev is not None else cn
            rec['err_rel'][t] = (e.norm(dim=-1)/cn.norm(dim=-1).clamp_min(1e-8)).cpu().numpy();E[t] = e.half()
            # what each condition writes back at the next step (normalised later; zero for free)
            new = torch.zeros_like(c)
            for i, cd in enumerate(cond):
                if cd == 'content':new[i] = cn[i]
                elif cd == 'vel':new[i] = cn[i]-(c_prev[i]/rc if c_prev is not None else 0.)
                elif cd == 'ring_pred':new[i] = m[i]
                elif cd == 'yoked_pred':new[i] = m[src[i]]
                elif cd in ('ring_err', 'frozen_err'):new[i] = e[i]
                elif cd == 'yoked_err':new[i] = e[src[i]]
            sh = [i for i, cd in enumerate(cond) if cd == 'shuf_err']
            if sh:
                back = torch.randint(0, max(1, t-9), (len(sh),), generator=gen, device=dev)
                new[sh] = E[back, torch.tensor(sh, device=dev)].float()
            rr = [i for i, cd in enumerate(cond) if cd == 'rand_err']
            if rr:new[rr] = torch.randn(S, d, generator=gen, device=dev)[seed_of[rr]]
            sig = new
            x = (1-a.alpha)*x+a.alpha*c;c_prev = c;m_prev = m
            rec['C'][t] = c.half().cpu().numpy();rec['idx'][t] = jidx.cpu().numpy();rec['coef'][t] = jcoef.half().cpu().numpy()
        if (t+1) % 50 == 0:
            el = time.time()-t0
            msg = ' | '.join(f"{cd} pred {rec['pred_cos'][t][[i for i, x_ in enumerate(cond) if x_ == cd]].mean():.2f} "
                             f"step {rec['step_cos'][t][[i for i, x_ in enumerate(cond) if x_ == cd]].mean():.2f}"
                             for cd in dict.fromkeys(cond))
            log(f'step {t+1}/{T} ({el/(t+1):.2f} s/step) | {msg[:600]}')
    # a glimpse in words: append a cue after the last placeholder and let it speak (nothing of this is fed back)
    # Ellipsis-only tokens are banned here: after hundreds of placeholders a model otherwise just continues the dots.
    glimpse = []
    if a.glimpse:
        with torch.no_grad():
            ban = torch.tensor([i for i, s in enumerate(tok.batch_decode([[i] for i in range(len(tok))]))
                                if s.strip() and set(s.strip()) <= set('….·')], device=dev)
            hk.add = None;g_ids = torch.tensor([tok(GLIMPSE, add_special_tokens=False)['input_ids']], device=dev).expand(B, -1)
            out = model(input_ids=g_ids, past_key_values=cache, use_cache=True, logits_to_keep=1);cache = out.past_key_values;toks = []
            for _ in range(a.glimpse):
                lg = out.logits[:, -1].float();lg[:, ban] = -math.inf
                nt = lg.argmax(-1, keepdim=True);toks.append(nt)
                out = model(input_ids=nt, past_key_values=cache, use_cache=True, logits_to_keep=1);cache = out.past_key_values
            glimpse = tok.batch_decode(torch.cat(toks, 1))
    peak = torch.cuda.max_memory_allocated()/2**30 if dev == 'cuda' else 0.
    np.savez(os.path.join(a.out, 'ring.npz'), **rec, mu=mu.cpu().numpy(), jbar=jbar.cpu().numpy())
    json.dump(dict(args=vars(a), rows=rows, src=src.tolist(), L_in=L_in, L_read=L_read, n_layers=n, d=d, Rn=Rn, rc=rc,
                   words=js.words, glimpse=glimpse, seconds=time.time()-t0, peak_gb=peak),
              open(os.path.join(a.out, 'meta.json'), 'w', encoding='utf-8'), ensure_ascii=False)
    log(f'saved {a.out} in {time.time()-t0:.0f} s')


if __name__ == '__main__':
    main()
