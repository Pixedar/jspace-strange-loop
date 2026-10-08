"""A strange loop in J-space: a latent-recurrent monologue that, every ten thoughts, models its own trajectory
and lives on with that self-model fed back into it.

The recurrent model
    Every thought is generated from the same fixed frame, with no text of earlier thoughts. The only thing that
    carries over from one thought to the next is a latent state z (layer-L residual space), added to the residual
    stream at layer L at every position. After a thought, the model's own layer-L states over its words (read before
    the injection is added, so they are what the words made of the model, not the injection itself) are averaged,
    centred on the unsteered mean, and decomposed into J-space: a sparse non-negative combination of J-lens vectors
    (Anthropic 2026, "Verbalizable Representations Form a Global Workspace in Language Models"). That workspace
    content j_t is the thought's point on the J-space path, and z leaks toward it: z <- (1-a) z + a j_t.

Self-reference
    Every K thoughts the model is asked what its mind has been doing over the last K moments. It gets no text: the K
    recurrent states z_{t-K+1..t} are added at layer L into K placeholder positions (Patchscope-style), and the model
    answers in words. The J-space decomposition of its answer (centred on the answer to an empty trace) is the
    self-model m. Depending on the condition, m (scaled to the size of a typical thought) is added to every following
    thought until the next reflection: thought trajectory -> compressed model of it -> model shapes the trajectory ->
    new compressed model ...

Conditions (one batch row each, several seeds; rows with the same seed share their sampling noise, so a row and its
paired control stay identical until the moment they are treated differently)
    free     self-models are formed but never fed back (observer only)
    loop     the row's own self-model is fed back
    yoked    a self-model of the same size and schedule is fed back, but it is another loop row's (about another mind)
    avg      the fed-back vector is a plain J-space average of the last K states, with no reflection by the model
    perturb  loop, but at --t_int the self-model is replaced once by another loop row's
    ablate   loop, but from --t_int on nothing is fed back

Run:  python strange_loop.py --out runs/main --grid main
"""
import argparse, json, math, os, re, time
import numpy as np
import torch

THOUGHT_FRAME = (
    "Think-aloud study of spontaneous thought. The participant lies with closed eyes and says out loud whatever passes "
    "through their mind, as it comes: words, images, memories, plans, worries, half-finished thoughts. Nothing was "
    "corrected. Excerpt from the middle of the session:\n\n\"...")
REFLECT_PRE = (
    "Think-aloud study of spontaneous thought. At intervals the recording was paused and the participant was played back "
    "a trace of what had been on their mind over the last ten moments, one mark per moment, in order:")
PLACE = ' ·'
REFLECT_FWD = (
    ".\n\nAsked where their mind would go next, the participant said: \"In the next few moments, my mind will probably "
    "turn to")
REFLECT_POST = (
    ".\n\nAsked what their mind had been doing over those ten moments, the participant said: \"Over these last moments, "
    "my mind has been")


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def decoder_layers(model):
    for path in ('model.layers', 'model.language_model.layers', 'transformer.h'):
        obj = model
        try:
            for part in path.split('.'):obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList):return obj
    raise ValueError('cannot find decoder layers')


def final_norm_weight(model):
    for path in ('model.norm', 'model.language_model.norm', 'transformer.ln_f'):
        obj = model
        try:
            for part in path.split('.'):obj = getattr(obj, part)
        except AttributeError:
            continue
        if getattr(obj, 'weight', None) is not None:return obj.weight.detach()
    raise ValueError('cannot find the final norm')


class JSpace:
    """J-lens atoms over whole English words (a leading space, >= 3 letters) and non-negative OMP on them."""

    def __init__(self, model, tok, J, k=16, vocab='words'):
        dev = next(model.parameters()).device;self.dev = dev;self.k = int(k)
        J = torch.as_tensor(J, dtype=torch.float32, device=dev)
        gamma = final_norm_weight(model).float().to(dev)
        W = model.get_output_embeddings().weight
        n = min(W.shape[0], len(tok))
        toks = tok.convert_ids_to_tokens(list(range(n)))
        keep = [i for i, s in enumerate(toks) if s and re.fullmatch(r'Ġ[A-Za-z]{3,}', s)]
        if vocab == 'english':
            # Common English words only, with an ordinary unembedding: the raw word set let under-trained tokens
            # (|w| 0.5-0.8 against a 1 % quantile of 0.97: ' rumpe', ' ForCanBeConvertedToForeach') and non-English
            # fragments (' modne', ' istedi') take a large share of every self-model (main run A, 8 Oct).
            from wordfreq import zipf_frequency
            nrm = W[keep].float().norm(dim=-1);lo = float(torch.quantile(nrm, .02))
            keep = [i for i, q in zip(keep, nrm.tolist()) if q >= lo and zipf_frequency(toks[i][1:].lower(), 'en') >= 3.
                    and re.fullmatch(r'[A-Z]?[a-z]+', toks[i][1:])]          # no code casing (' iNdEx', ' DEALINGS')
        self.sub = torch.tensor(keep, device=dev)
        self.words = [toks[i][1:] for i in keep]
        rows = []
        # The unembedding is centred first: its mean row raises every logit alike (no content), and left in, it gave
        # every atom a shared direction (mean pairwise cosine 0.64, one component = 64 % of the atoms' variance;
        # measured on Qwen3-4B-Base), so every workspace vector pointed mostly the same way.
        wbar = W[:n].float().mean(0)
        with torch.no_grad():
            for i in range(0, len(keep), 4096):
                a = ((W[self.sub[i:i+4096]].float()-wbar)*gamma[None])@J   # atom t = J^T (gamma * (w_t - mean w))
                rows.append((a/a.norm(dim=-1, keepdim=True).clamp_min(1e-6)).half())
        self.D = torch.cat(rows)                                         # (n_words, d) unit atoms

    @torch.no_grad()
    def decompose(self, x, k=None):
        """x (B, d) -> word indices (B, k), coefficients (B, k) >= 0, J-space part (B, d)."""
        k = self.k if k is None else int(k);x = x.float();B = x.shape[0]
        idx = torch.zeros((B, 0), dtype=torch.long, device=self.dev);r = x.clone()
        eye = torch.eye(k, device=self.dev)
        for j in range(k):
            c = (r.half()@self.D.T).float()
            if idx.shape[1]:c.scatter_(1, idx, -math.inf)
            idx = torch.cat([idx, c.argmax(-1, keepdim=True)], 1)
            A = self.D[idx].float()
            G = A@A.transpose(1, 2)+1e-4*eye[:j+1, :j+1]
            b = (A@x[:, :, None])[:, :, 0]
            coef = torch.linalg.solve(G, b)
            r = x-(coef[:, :, None]*A).sum(1)
        from scipy.optimize import nnls                                 # exact non-negativity on the k x k problem
        Gc, bc = G.double().cpu().numpy(), b.double().cpu().numpy();out = np.zeros_like(bc)
        for i in range(B):
            Lc = np.linalg.cholesky(Gc[i])
            out[i] = nnls(Lc.T, np.linalg.solve(Lc, bc[i]))[0]
        coef = torch.tensor(out, dtype=torch.float32, device=self.dev)
        return idx, coef, (coef[:, :, None]*A).sum(1)

    def top(self, idx, coef, n=8):
        o = np.argsort(-coef)[:n]
        return [(self.words[int(idx[i])], round(float(coef[i]), 2)) for i in o if coef[i] > 0]


class Hook:
    """On one decoder block: optionally record the last `read` positions of its output (before anything is
    added), then add a vector."""

    def __init__(self, module):
        self.add = None;self.read = 0;self.buf = []
        self.h = module.register_forward_hook(self.fn)

    def fn(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if self.read:self.buf.append(h[:, -self.read:].float())
        if self.add is None:return None
        h = h+self.add.to(h.dtype)
        return (h,)+tuple(out[1:]) if isinstance(out, tuple) else h


def make_rows(a):
    """The batch: one dict per row."""
    rows = []
    if a.grid == 'main':
        for c in a.conds.split(','):
            for s in range(a.seeds):rows.append(dict(cond=c, seed=s, g=a.g, beta=a.beta, g_r=a.g_r))
    elif a.grid == 'pilot':
        for g in [0., .03, .06, .12, .25]:
            for s in range(3):rows.append(dict(cond='free', seed=s, g=g, beta=0., g_r=a.g_r))
        for g in [.06, .12]:
            for b in [1., 2.]:
                for s in range(2):rows.append(dict(cond='loop', seed=s, g=g, beta=b, g_r=a.g_r))
    elif a.grid == 'smoke':
        for c in ['free', 'loop', 'yoked', 'avg', 'perturb', 'ablate']:
            for s in range(2):rows.append(dict(cond=c, seed=s, g=a.g, beta=a.beta, g_r=a.g_r))
    return rows


class Experiment:
    def __init__(self, a):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.a = a
        dev = 'cuda' if torch.cuda.is_available() else 'cpu';self.dev = dev
        dtype = torch.bfloat16 if dev == 'cuda' else torch.float32
        self.tok = AutoTokenizer.from_pretrained(a.model)
        self.model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(dev).eval()
        for p in self.model.parameters():p.requires_grad_(False)
        cfg = getattr(self.model.config, 'text_config', self.model.config);self.d = int(cfg.hidden_size)
        if a.jlens == 'identity':
            J, self.L = np.eye(self.d, dtype=np.float32), a.layer
        else:
            z = np.load(a.jlens);J, self.L = z['J'], int(z['layer'])
        self.js = JSpace(self.model, self.tok, J, k=a.k, vocab=a.vocab)
        blocks = decoder_layers(self.model)
        self.L_in = a.L_in if a.L_in >= 0 else self.L
        self.fid = [(int(c.split(':')[0]), float(c.split(':')[1])) for c in a.fid.split(',')] if a.fid else []
        self.hooks = {l:Hook(blocks[l]) for l in {self.L, self.L_in, *[l for l, _ in self.fid]}}
        self.hook = self.hooks[self.L]
        log(f'model {a.model} d={self.d} layer {self.L} | {len(self.js.words)} word atoms | {dev}')

        enc = lambda s:self.tok(s, add_special_tokens=False)['input_ids']
        self.frame = torch.tensor([enc(THOUGHT_FRAME)], device=dev)
        pl = enc(PLACE);assert len(pl) == 1, pl
        pre, post = enc(REFLECT_PRE), enc(REFLECT_POST)
        self.refl = torch.tensor([pre+pl*a.K+post], device=dev)
        self.place_pos = torch.arange(len(pre), len(pre)+a.K, device=dev)
        self.n_tail = len(post)-1                                        # the question's words after the trace
        fwd = enc(REFLECT_FWD);self.refl_f = torch.tensor([pre+pl*a.K+fwd], device=dev);self.n_tail_f = len(fwd)-1
        V = self.model.get_output_embeddings().weight.shape[0]
        nv = self.tok.vocab_size                                         # ordinary tokens; added/special ones follow
        ban = torch.zeros(V, dtype=torch.bool);ban[nv:] = True
        for i, t in enumerate(self.tok.batch_decode([[i] for i in range(nv)])):
            if '\n' in t or '"' in t or '“' in t or '”' in t:ban[i] = True   # the utterance stays one open quote
        for i in self.tok.all_special_ids:ban[i] = True
        self.ban = ban.to(dev)

        self.rows = make_rows(a);self.B = len(self.rows);self.S = max(r['seed'] for r in self.rows)+1
        R = self.rows
        self.seed_of = torch.tensor([r['seed'] for r in R], device=dev)
        self.g = torch.tensor([r['g'] for r in R], device=dev)
        self.beta = torch.tensor([r['beta'] for r in R], device=dev)
        self.g_r = torch.tensor([r['g_r'] for r in R], device=dev)
        self.cond = [r['cond'] for r in R]
        idx = {(r['cond'], r['seed']):i for i, r in enumerate(R)}
        S = self.S
        base = 'loopc' if any(r['cond'] == 'loopc' for r in R) else 'loop'
        self.partner = [idx.get((base, (r['seed']+1) % S), -1) for r in R]       # yoked source
        self.foreign = [idx.get((base, (r['seed']+S//2) % S), -1) for r in R]    # perturbation / clamp source
        self.free_rows = [b for b, r in enumerate(R) if r['cond'] == 'free']
        self.gen = torch.Generator(device=dev);self.gen.manual_seed(a.rng)

    # ---------------------------------------------------------------- generation
    def sample(self, logits, hist, temp, min_p):
        lg = logits.float()/temp
        lg[:, self.ban] = -math.inf
        # A soft penalty on repeating a 4-gram inside one utterance, applied before min-p so it can never leave a row
        # with nothing allowed. (A hard trigram ban did, once "don't" had been said: argmax over all -inf picked
        # token 0, "!", giving "I don!t", measured in the first pilot.)
        if hist and len(hist[0]) >= 3:
            for b in range(self.B):
                h = hist[b];key = (h[-3], h[-2], h[-1])
                bad = [h[i+3] for i in range(len(h)-3) if (h[i], h[i+1], h[i+2]) == key]
                if bad:lg[b, bad] -= 4.
        p = torch.softmax(lg, -1)
        lg[p < min_p*p.max(-1, keepdim=True).values] = -math.inf
        u = torch.rand((self.S, lg.shape[-1]), generator=self.gen, device=self.dev).clamp_(1e-10, 1-1e-7)
        gum = -torch.log(-torch.log(u))                                  # one noise draw per seed, shared by its rows
        return (lg+gum[self.seed_of]).argmax(-1)

    @torch.no_grad()
    def run(self, prompt, n_new, add_prompt, add_new, temp, min_p, tail=0, add_layer=None):
        """Prefill prompt for every row (adding add_prompt at add_layer, default L), then sample n_new tokens (adding
        add_new there). Returns token ids (B, n_new) and the layer-L states (read before any add at L) of the last
        `tail` prompt positions followed by the n_new sampled tokens: (B, tail+n_new, d)."""
        B = self.B;hk = self.hook;ha = self.hooks[self.L if add_layer is None else add_layer]
        hk.buf = []
        ha.add, hk.read = add_prompt, tail
        out = self.model(input_ids=prompt.expand(B, -1), use_cache=True)
        cache, logits = out.past_key_values, out.logits[:, -1]
        ids, hist = [], [[] for _ in range(B)]
        ha.add, hk.read = add_new, 1
        for i in range(n_new):
            t = self.sample(logits, hist, temp, min_p);ids.append(t)
            tl = t.tolist()
            for b in range(B):hist[b].append(tl[b])
            out = self.model(input_ids=t[:, None], past_key_values=cache, use_cache=True)
            cache, logits = out.past_key_values, out.logits[:, -1]
        ha.add, hk.read = None, 0
        st = torch.cat(hk.buf, 1);hk.buf = []
        return (torch.stack(ids, 1) if ids else torch.zeros((B, 0), dtype=torch.long, device=self.dev)), st

    def pool(self, st, skip):
        """Mean state over tokens (after the first `skip`), leaving out massive-activation tokens."""
        st = st[:, skip:];nrm = st.norm(dim=-1)
        ok = (nrm <= 2.5*nrm.median(1, keepdim=True).values).float()
        return (st*ok[..., None]).sum(1)/ok.sum(1, keepdim=True).clamp_min(1.)

    def thought(self, inj):
        a = self.a;add = inj[:, None, :]
        ids, st = self.run(self.frame, a.n_tok, add, add, a.temp, a.min_p)
        return ids, st

    def reflect(self, zwin, g_r, layer=None, n_desc=None, fwd=False):
        """zwin (B, K, d) recurrent states, added into the K placeholders at layer L_in. Returns the answer's ids
        (the verbal report, sampled) and the pooled layer-L state of the question's last words, where the model holds
        what it is about to say about its last K moments (deterministic: no sampling noise in the self-model)."""
        a = self.a;layer = self.L_in if layer is None else layer
        n_desc = a.n_desc if n_desc is None else n_desc
        prompt, nt = (self.refl_f, self.n_tail_f) if fwd else (self.refl, self.n_tail)
        add = torch.zeros((self.B, prompt.shape[1], self.d), device=self.dev)
        add[:, self.place_pos] = (g_r*self.Rn/self.rc)[:, None, None]*zwin
        ids, st = self.run(prompt, n_desc, add, None, a.temp_r, a.min_p, tail=nt, add_layer=layer)
        return ids, self.pool(st[:, :nt], 0)

    def content(self, x, base):
        """J-space decomposition of a centred state, relative to the typical workspace `base`."""
        idx, coef, part = self.js.decompose(x)
        return idx, coef, part-base

    # ---------------------------------------------------------------- calibration
    def calibrate(self):
        a = self.a;zero = torch.zeros((self.B, self.d), device=self.dev)
        sts = []
        for _ in range(a.n_cal):
            _, st = self.thought(zero);sts.append(st[:, a.skip:])
        st = torch.cat(sts, 1).reshape(-1, self.d);nrm = st.norm(dim=-1)
        ok = nrm <= 2.5*nrm.median()
        self.Rn = float(nrm[ok].median());self.mu = st[ok].mean(0)
        xs = torch.cat([self.pool(s, 0) for s in sts])-self.mu
        _, _, part = self.js.decompose(xs);self.rj = float(part.norm(dim=-1).mean())
        self.jbar = part.mean(0);self.rc = float((part-self.jbar).norm(dim=-1).mean())
        self.rj_x = float(xs.norm(dim=-1).mean())
        _, m = self.reflect(torch.zeros((self.B, a.K, self.d), device=self.dev), self.g_r, n_desc=0)
        self.mu_r = m.mean(0)                                            # deterministic: one pass is the mean
        _, _, mp = self.js.decompose(m-self.mu_r);self.mbar = mp.mean(0)
        _, m = self.reflect(torch.zeros((self.B, a.K, self.d), device=self.dev), self.g_r, n_desc=0, fwd=True)
        self.mu_rf = m.mean(0);_, _, mp = self.js.decompose(m-self.mu_rf);self.mbar_f = mp.mean(0)
        log(f'calibrated: residual norm {self.Rn:.1f} | thought-mean norm {self.rj_x:.2f} | J-space part {self.rj:.2f}'
            f' | its typical part {self.jbar.norm():.2f} | content (part - typical) {self.rc:.2f}')

    # ---------------------------------------------------------------- main loop
    def main(self):
        a = self.a;B, d, K, N = self.B, self.d, a.K, a.n_iter;NR = N//K;dev = self.dev
        ck = os.path.join(a.out, 'ckpt.pt')
        if os.path.exists(ck):
            S = torch.load(ck, map_location='cpu', weights_only=False)
            for k in ('mu', 'mu_r', 'jbar', 'mbar', 'mu_rf', 'mbar_f', 'clamp_vec'):setattr(self, k, S[k].to(dev))
            self.Rn, self.rj, self.rj_x, self.rc = S['Rn'], S['rj'], S['rj_x'], S['rc']
            z, inj_m, zhist, ablated = S['z'].to(dev), S['inj_m'].to(dev), S['zhist'].to(dev), S['ablated'].to(dev)
            self.gen.set_state(S['gen']);t0 = S['t'];D = S['data']
            log(f'resumed at thought {t0}')
        else:
            self.calibrate()
            z = torch.zeros((B, d), device=dev);inj_m = torch.zeros((B, d), device=dev)
            zhist = torch.zeros((K, B, d), device=dev);ablated = torch.zeros(B, dtype=torch.bool, device=dev)
            t0 = 0;self.clamp_vec = torch.zeros((B, d), device=dev)
            D = dict(mf=np.zeros((NR, B, d), np.float16), fdesc=[], generic=np.zeros((NR, d), np.float16),j=np.zeros((N, B, d), np.float16), j_idx=np.zeros((N, B, a.k), np.int32),
                     j_coef=np.zeros((N, B, a.k), np.float16), texts=[],
                     m=np.zeros((NR, B, d), np.float16), m_idx=np.zeros((NR, B, a.k), np.int32),
                     m_coef=np.zeros((NR, B, a.k), np.float16), desc=[],
                     inj=np.zeros((NR, B, d), np.float16), inj_src=np.zeros((NR, B), np.int32),
                     fid={f'{l}_{g}':np.zeros((NR, B, d), np.float16) for l, g in self.fid},
                     fid_desc={f'{l}_{g}':[] for l, g in self.fid})
        t_start = time.time()
        for t in range(t0, N):
            u = z+self.beta[:, None]*(~ablated)[:, None]*inj_m
            nu = u.norm(dim=-1, keepdim=True)/self.rc                    # bounded: never above 1.5 typical contents
            v = (self.g*self.Rn/self.rc)[:, None]*u*torch.clamp(1.5/nu.clamp_min(1e-6), max=1.)
            ids, st = self.thought(v)
            x = self.pool(st, a.skip)-self.mu
            jidx, jcoef, j = self.content(x, self.jbar)
            z = (1-a.alpha)*z+a.alpha*j
            zhist = torch.cat([zhist[1:], z[None]])
            D['j'][t] = j.cpu().half().numpy();D['j_idx'][t] = jidx.cpu().numpy();D['j_coef'][t] = jcoef.cpu().half().numpy()
            D['texts'].append(self.tok.batch_decode(ids))
            if (t+1) % K == 0:
                r = (t+1)//K-1;zwin = zhist.transpose(0, 1)
                dids, mstate = self.reflect(zwin, self.g_r)
                midx, mcoef, m = self.content(mstate-self.mu_r, self.mbar)
                D['m'][r] = m.cpu().half().numpy();D['m_idx'][r] = midx.cpu().numpy();D['m_coef'][r] = mcoef.cpu().half().numpy()
                D['desc'].append(self.tok.batch_decode(dids))
                fids, fst = self.reflect(zwin, self.g_r, n_desc=a.n_desc_f, fwd=True)   # self-prediction (logged only)
                D['mf'][r] = self.content(fst-self.mu_rf, self.mbar_f)[2].cpu().half().numpy()
                D['fdesc'].append(self.tok.batch_decode(fids))
                for l, g in self.fid:                                  # the same reflection at other settings
                    _, fst = self.reflect(zwin, torch.full_like(self.g_r, g), layer=l, n_desc=0)
                    D['fid'][f'{l}_{g}'][r] = self.content(fst-self.mu_r, self.mbar)[2].cpu().half().numpy()
                unit = lambda u:u/u.norm(dim=-1, keepdim=True).clamp_min(1e-6)*self.rc
                avg = self.content(zhist.mean(0), 0.)[2]
                # the population's generic self-model at this moment: the mean over the observer rows, which no
                # treatment touches. The individuated conditions (loopc, yokedc, clampc) feed back only what sets one
                # mind's self-model apart from it.
                generic = m[self.free_rows].mean(0) if self.free_rows else m.mean(0)
                D['generic'][r] = generic.cpu().half().numpy()
                mi = m-generic[None]
                new = torch.zeros_like(inj_m);src = []
                for b, c in enumerate(self.cond):
                    if c == 'free':src.append(-1);continue
                    if c in ('loop', 'perturb', 'ablate', 'loopc', 'clampc'):s = b
                    elif c in ('yoked', 'yokedc'):s = self.partner[b]
                    elif c == 'avg':s = -2
                    if c == 'perturb' and t+1 == a.t_int:s = self.foreign[b]
                    if c in ('loopc', 'yokedc', 'clampc'):
                        vec = unit(mi[s:s+1])[0]
                        if c == 'clampc' and t+1 == a.t_int:self.clamp_vec[b] = unit(mi[self.foreign[b]:self.foreign[b]+1])[0]
                        if c == 'clampc' and a.t_int <= t+1 < a.t_int+a.clamp_len:vec, s = self.clamp_vec[b], self.foreign[b]
                        new[b] = vec
                    else:new[b] = unit(avg[b:b+1])[0] if s == -2 else unit(m[s:s+1])[0]
                    src.append(s)
                    if c == 'ablate' and t+1 >= a.t_int:ablated[b] = True
                inj_m = new
                D['inj'][r] = inj_m.cpu().half().numpy();D['inj_src'][r] = np.array(src)
                el = time.time()-t_start;rate = el/(t+1-t0)
                w = [f"{self.cond[b]}/{self.rows[b]['seed']}: {self.js.top(midx[b].cpu().numpy(), mcoef[b].cpu().numpy(), 5)}"
                     for b in range(0, B, max(1, B//6))]
                log(f'thought {t+1}/{N} ({rate:.2f} s/thought, eta {(N-t-1)*rate/60:.1f} min) | self-models: ' + ' | '.join(w))
                log('   e.g. thought: ' + D['texts'][-1][0][:160].replace('\n', ' ') + ' || answer: ' + D['desc'][-1][0][:120])
            if (t+1) % a.ckpt_every == 0 or t+1 == N:
                S = dict(mu=self.mu.cpu(), mu_r=self.mu_r.cpu(), jbar=self.jbar.cpu(), mbar=self.mbar.cpu(), rc=self.rc,
                         mu_rf=self.mu_rf.cpu(), mbar_f=self.mbar_f.cpu(), clamp_vec=self.clamp_vec.cpu(),
                         Rn=self.Rn, rj=self.rj, rj_x=self.rj_x,
                         z=z.cpu(), inj_m=inj_m.cpu(), zhist=zhist.cpu(), ablated=ablated.cpu(), gen=self.gen.get_state(),
                         t=t+1, data=D)
                torch.save(S, ck+'.tmp');os.replace(ck+'.tmp', ck)
        self.save(D)

    def save(self, D):
        a = self.a
        np.savez(os.path.join(a.out, 'traj.npz'), j=D['j'], j_idx=D['j_idx'], j_coef=D['j_coef'], m=D['m'],
                 m_idx=D['m_idx'], m_coef=D['m_coef'], inj=D['inj'], inj_src=D['inj_src'], mu=self.mu.cpu().numpy(),
                 jbar=self.jbar.cpu().numpy(), mbar=self.mbar.cpu().numpy(), mf=D['mf'], generic=D['generic'],
                 **{f'fid_{k}':v for k, v in D['fid'].items()})
        meta = dict(args=vars(a), rows=self.rows, partner=self.partner, foreign=self.foreign, layer=self.L, L_in=self.L_in,
                    Rn=self.Rn, rj=self.rj, rj_x=self.rj_x, rc=self.rc, words=self.js.words, texts=D['texts'], desc=D['desc'],
                    fid_desc=D['fid_desc'], fdesc=D['fdesc'], thought_frame=THOUGHT_FRAME,
                    reflect_prompt=REFLECT_PRE+PLACE*a.K+REFLECT_POST, forward_prompt=REFLECT_PRE+PLACE*a.K+REFLECT_FWD)
        with open(os.path.join(a.out, 'meta.json'), 'w', encoding='utf-8') as f:json.dump(meta, f, ensure_ascii=False)
        log('saved', a.out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen3-4B-Base');p.add_argument('--jlens', default='qwen3-4b-base-jlens.npz')
    p.add_argument('--layer', type=int, default=16, help='only with --jlens identity')
    p.add_argument('--out', required=True);p.add_argument('--grid', default='main')
    p.add_argument('--conds', default='free,loop,yoked,avg,perturb,ablate');p.add_argument('--seeds', type=int, default=8)
    p.add_argument('--n_iter', type=int, default=300);p.add_argument('--K', type=int, default=10)
    p.add_argument('--t_int', type=int, default=150, help='thought at which perturb/ablate/clampc rows are treated')
    p.add_argument('--clamp_len', type=int, default=50, help='thoughts for which clampc holds a foreign self-model')
    p.add_argument('--n_desc_f', type=int, default=12, help='words said in answer to the forward question (display)')
    p.add_argument('--n_tok', type=int, default=40);p.add_argument('--n_desc', type=int, default=24)
    p.add_argument('--skip', type=int, default=3);p.add_argument('--k', type=int, default=16)
    p.add_argument('--vocab', default='english', help="J-lens dictionary: 'english' (common English words) or 'words'")
    p.add_argument('--g', type=float, default=.1, help='recurrent injection, in units of the residual norm')
    p.add_argument('--g_r', type=float, default=.5, help='placeholder injection in the reflection')
    p.add_argument('--beta', type=float, default=1., help='self-model weight relative to a typical thought')
    p.add_argument('--alpha', type=float, default=.4, help='leak of the recurrent state toward each new thought')
    p.add_argument('--temp', type=float, default=1.);p.add_argument('--temp_r', type=float, default=.3)
    p.add_argument('--min_p', type=float, default=.05);p.add_argument('--n_cal', type=int, default=3)
    p.add_argument('--L_in', type=int, default=12, help='layer where the reflection placeholders get the states (-1: L)')
    p.add_argument('--fid', default='', help='extra reflections to log for a fidelity check: layer:gain,layer:gain')
    p.add_argument('--rng', type=int, default=1234);p.add_argument('--ckpt_every', type=int, default=20)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.rng)
    Experiment(a).main()


if __name__ == '__main__':
    main()
