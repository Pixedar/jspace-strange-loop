"""Shared substrate for research v3: model + Jacobian-lens loading, batched J-space decomposition, residual hooks,
prompt building and answer scoring. Everything here is measurement machinery; experiments live in e1.py / e2.py."""
import os, re, json, time, math, hashlib, pathlib, random
import numpy as np
import torch

OUT = pathlib.Path(os.environ.get('V3_OUT', '/workspace/v3'))
LENS_REPO, LENS_REV = 'neuronpedia/jacobian-lens', '9fa267f841d62aa468dab5b9b38c3dc45a82d863'
MODELS = {
    # layers = calibration scan at ~45/60/70/82 % depth; read/write = loop layers (protocol section 4)
    'qwen3-8b': dict(name='Qwen/Qwen3-8B', rev='b968826d9c46dd6066d109eabc6255188de91218',
                     lens='qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt', layers=[16, 22, 25, 29],
                     read=25, write=16),
    'qwen3-14b': dict(name='Qwen/Qwen3-14B', rev='40c069824f4251a91eefaf281ebe4c544efd3e18',
                      lens='qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt', layers=[18, 24, 28, 33],
                      read=28, write=18),
    'qwen3-4b': dict(name='Qwen/Qwen3-4B', rev='1cfa9a7208912126459214e8b04321603b3df60c',
                     lens='qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt', layers=[16, 22, 25, 29],
                     read=25, write=16),
    # smoke-test substrate only (base checkpoint with the instruct lens: pipeline mechanics, not science)
    'qwen3-1.7b-smoke': dict(name='Qwen/Qwen3-1.7B-Base', rev=None,
                             lens='qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt',
                             layers=[12, 17, 20, 23], read=20, write=12),
}


def log(*a):
    print(time.strftime('%F %T'), *a, flush=True)


def atomic_npz(path, **arrs):
    path = pathlib.Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent/(path.name+'.tmp.npz')
    np.savez_compressed(tmp, **arrs);os.replace(tmp, path)


def atomic_json(path, obj):
    path = pathlib.Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent/(path.name+'.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o:o.tolist() if hasattr(o, 'tolist') else str(o)))
    os.replace(tmp, path)


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda:f.read(chunk), b''):h.update(b)
    return h.hexdigest()


def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)


def decoder_layers(model):
    for path in ('model.layers', 'model.model.layers', 'transformer.h'):
        m = model
        try:
            for p in path.split('.'):m = getattr(m, p)
            return m
        except AttributeError:continue
    raise ValueError('decoder layers not found')


# ----------------------------------------------------------------------------------------------------------------
# J-space dictionary and decomposition
# ----------------------------------------------------------------------------------------------------------------
def english_vocab(tok, W):
    """Common English whole-word tokens (leading space, >= 3 letters, zipf >= 3, ordinary unembedding norm).
    The paper's every-token dictionary halves self-reading on Qwen (repo Part 2), so this is the primary dictionary."""
    from wordfreq import zipf_frequency
    n = min(W.shape[0], len(tok))
    toks = tok.convert_ids_to_tokens(list(range(n)))
    keep = [i for i, s in enumerate(toks) if s and re.fullmatch(r'Ġ[A-Za-z]{3,}', s)]
    nrm = W[keep].float().norm(dim=-1);lo = float(torch.quantile(nrm, .02))
    keep = [i for i, q in zip(keep, nrm.tolist()) if q >= lo and zipf_frequency(toks[i][1:].lower(), 'en') >= 3.
            and re.fullmatch(r'[A-Z]?[a-z]+', toks[i][1:])]
    return keep, [toks[i][1:] for i in keep]


def full_vocab(tok, W):
    special = set(tok.all_special_ids);n = min(W.shape[0], tok.vocab_size)
    keep = [i for i in range(n) if i not in special]
    return keep, [tok.decode([i]).strip() for i in keep]


def nnls_fista(G, b, x0, iters=300):
    """Batched non-negative least squares min 0.5 c'Gc - b'c, c >= 0 (G: B x k x k). Projected FISTA with step 1/L."""
    L = torch.linalg.matrix_norm(G, ord=2).clamp_min(1e-8)[:, None]
    x = x0.clamp_min(0);y = x.clone();t = 1.
    for _ in range(iters):
        g = (G@y[:, :, None])[:, :, 0]-b
        xn = (y-g/L).clamp_min(0)
        tn = (1+math.sqrt(1+4*t*t))/2
        y = xn+((t-1)/tn)*(xn-x);x, t = xn, tn
    return x


class Lens:
    """Per-layer J-lens dictionaries. atom_l(w) = J_l^T (gamma * (u_w - mean u)), unit norm (repo convention, matches
    the paper's rows of W_U J_l with the final-norm gain and the unembedding mean removed)."""

    def __init__(self, model, tok, lens_path, layers, vocab='english', dev='cuda'):
        self.dev = dev;self.layers = list(layers);self.ddt = torch.float16 if str(dev).startswith('cuda') else torch.float32
        ck = torch.load(lens_path, map_location='cpu', weights_only=True)
        Js = ck['J'] if isinstance(ck, dict) and 'J' in ck else ck
        self.J = {l:torch.as_tensor(Js[l]).to(dev, torch.float32) for l in layers}
        W = model.get_output_embeddings().weight.detach()
        gamma = final_norm(model).weight.detach().float().to(dev)
        keep, self.words = english_vocab(tok, W) if vocab == 'english' else full_vocab(tok, W)
        self.ids = torch.tensor(keep, device=dev)
        n = min(W.shape[0], len(tok));wbar = W[:n].float().mean(0).to(dev)
        self.D = {}
        for l in layers:
            rows = []
            for i in range(0, len(keep), 4096):
                a = ((W[self.ids[i:i+4096]].float().to(dev)-wbar)*gamma)@self.J[l]
                rows.append((a/a.norm(dim=-1, keepdim=True).clamp_min(1e-6)).to(self.ddt))
            self.D[l] = torch.cat(rows)
        self.mu = {}                                   # (layer, position type) -> centring vector, set from calibration
        self.word_index = {w:i for i, w in enumerate(self.words)}

    @torch.no_grad()
    def decompose(self, l, x, k=25):
        """x (B, d) already centred -> idx (B, k) dictionary indices, coef (B, k) >= 0, recon (B, d)."""
        x = x.float();B = x.shape[0];D = self.D[l]
        idx = torch.zeros((B, 0), dtype=torch.long, device=self.dev);r = x.clone()
        eye = torch.eye(k, device=self.dev)
        for j in range(k):
            c = (r.to(self.ddt)@D.T).float()
            if idx.shape[1]:c.scatter_(1, idx, -math.inf)
            idx = torch.cat([idx, c.argmax(-1, keepdim=True)], 1)
            A = D[idx].float()
            G = A@A.transpose(1, 2)+1e-4*eye[:j+1, :j+1]
            b = (A@x[:, :, None])[:, :, 0]
            coef = torch.linalg.solve(G, b)
            r = x-(coef[:, :, None]*A).sum(1)
        coef = nnls_fista(G, b, coef)
        return idx, coef, (coef[:, :, None]*A).sum(1)

    def synth(self, l, idx, coef):
        """Re-synthesize a code (indices into the shared word list, coefficients) with layer-l atoms."""
        return (coef.float()[:, :, None]*self.D[l][idx].float()).sum(1)

    def words_of(self, idx, coef, n=8):
        o = np.argsort(-coef)[:n]
        return [(self.words[int(idx[i])], round(float(coef[i]), 2)) for i in o if coef[i] > 0]


def final_norm(model):
    for path in ('model.norm', 'model.model.norm', 'transformer.ln_f'):
        m = model
        try:
            for p in path.split('.'):m = getattr(m, p)
            return m
        except AttributeError:continue
    raise ValueError('final norm not found')


# ----------------------------------------------------------------------------------------------------------------
# Model wrapper with read / edit hooks on decoder-block outputs
# ----------------------------------------------------------------------------------------------------------------
class LM:
    def __init__(self, key, dtype=torch.bfloat16, vocab='english', dev='cuda', lens_path=None, cache_dir=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        cfg = MODELS[key];self.key = key;self.cfg = cfg;self.dev = dev
        kw = dict(revision=cfg['rev']) if cfg['rev'] else {}
        if cache_dir:kw['cache_dir'] = cache_dir
        self.tok = AutoTokenizer.from_pretrained(cfg['name'], **kw)
        self.tok.padding_side = 'left'
        if self.tok.pad_token is None:self.tok.pad_token = self.tok.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(cfg['name'], dtype=dtype, device_map=dev,
                                                              attn_implementation='sdpa', **kw).eval()
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(cfg['name'], torch_dtype=dtype, device_map=dev,
                                                              attn_implementation='sdpa', **kw).eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.blocks = decoder_layers(self.model);self.n_layers = len(self.blocks)
        self.d = self.model.config.hidden_size
        if lens_path is None:
            from huggingface_hub import hf_hub_download
            lens_path = hf_hub_download(LENS_REPO, cfg['lens'], revision=LENS_REV, cache_dir=cache_dir)
        self.lens_path = lens_path;self.lens_sha = sha256(lens_path)
        layers = sorted(set(cfg['layers'])|{cfg['read'], cfg['write']})
        self.lens = Lens(self.model, self.tok, lens_path, layers, vocab=vocab, dev=dev)
        self.edit = {};self.read = {};self.store = {}
        self.handles = [b.register_forward_hook(self._hook(i)) for i, b in enumerate(self.blocks)]
        self.provenance = dict(model=cfg['name'], revision=cfg['rev'], lens=cfg['lens'], lens_revision=LENS_REV,
                               lens_sha256=self.lens_sha, dtype=str(dtype), dictionary=vocab,
                               dictionary_size=len(self.lens.words), torch=torch.__version__,
                               transformers=__import__('transformers').__version__)

    def _hook(self, i):
        def fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if i in self.read:                          # natural state, read before any edit at this layer
                pos = self.read[i]
                self.store[i] = h[torch.arange(h.shape[0], device=h.device)[:, None], pos].float() if pos is not None \
                    else h.float()
            if i in self.edit:
                h = self.edit[i](h)
                return (h,)+tuple(out[1:]) if isinstance(out, tuple) else h
            return None
        return fn

    def clear(self):
        self.edit.clear();self.read.clear();self.store.clear()

    # -- prompts -------------------------------------------------------------------------------------------------
    def chat(self, user, thinking=False):
        return self.tok.apply_chat_template([{'role':'user', 'content':user}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=thinking)

    def encode(self, texts, marks=None):
        """Left-padded batch. marks: per text a list of character offsets; returns for each the index (in the padded
        batch) of the token whose span ends at or after that offset (the token that completes the marked text)."""
        enc = self.tok(texts, return_tensors='pt', padding=True, add_special_tokens=False,
                       return_offsets_mapping=marks is not None)
        ids = enc['input_ids'].to(self.dev);att = enc['attention_mask'].to(self.dev)
        pos = None
        if marks is not None:
            off = enc['offset_mapping'];T = ids.shape[1];pos = []
            for b, ms in enumerate(marks):
                ends = off[b, :, 1].tolist();valid = att[b].tolist()
                row = []
                for m in ms:
                    cand = [t for t in range(T) if valid[t] and ends[t] >= m]
                    row.append(cand[0] if cand else T-1)
                pos.append(row)
            pos = torch.tensor(pos, device=self.dev)
        return ids, att, pos

    @torch.no_grad()
    def forward(self, ids, att, keep_last=1):
        posid = (att.long().cumsum(-1)-1).clamp_min(0)
        out = self.model(input_ids=ids, attention_mask=att, position_ids=posid, use_cache=False,
                         logits_to_keep=keep_last)
        return out.logits.float()

    # -- answers ---------------------------------------------------------------------------------------------------
    def variant_ids(self, word):
        ids = set()
        for v in (word, ' '+word, word.capitalize(), ' '+word.capitalize(), word.lower(), ' '+word.lower()):
            t = self.tok.encode(v, add_special_tokens=False)
            if t:ids.add(t[0])
        return sorted(ids)

    def logodds(self, logits, cands):
        """logits (B, V); cands: list of (A, B) word pairs. Returns log P(A-variants) - log P(B-variants)."""
        lp = torch.log_softmax(logits.float(), -1);out = []
        for b, (a, c) in enumerate(cands):
            ia, ic = self.variant_ids(a), self.variant_ids(c)
            sa, sc = set(ia), set(ic)
            ia = [i for i in ia if i not in sc];ic = [i for i in ic if i not in sa]
            out.append((torch.logsumexp(lp[b, ia], 0)-torch.logsumexp(lp[b, ic], 0)).item())
        return np.array(out)

    def cand_mass(self, logits, cands):
        lp = torch.log_softmax(logits.float(), -1);out = []
        for b, (a, c) in enumerate(cands):
            ids = sorted(set(self.variant_ids(a))|set(self.variant_ids(c)))
            out.append(torch.logsumexp(lp[b, ids], 0).exp().item())
        return np.array(out)


def top_tokens(lm, logits, n=20):
    lp = torch.log_softmax(logits.float(), -1);v, i = lp.topk(n, -1)
    return i.cpu().numpy().astype(np.int32), v.cpu().numpy().astype(np.float16)


def batches(seq, n):
    for i in range(0, len(seq), n):yield seq[i:i+n]
