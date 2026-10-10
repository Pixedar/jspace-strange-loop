"""Shared pieces for the workspace tests (ws_*.py).

Lab loads a model with its Neuronpedia Jacobian lens. J[l] maps the output of block l to the last block, so
hidden_states[l+1] goes with J[l]. Lab gives per-layer J-lens atoms J_l^T (gamma * (w_t - mean w)) and J-lens scores
over the English dictionary, as in strange_loop.JSpace. It also gives a top-k J-space ablation like the paper's: at
every position and every layer of a band, the residual's projection onto the k most active dictionary atoms is
removed. Tokens in the clean run's top-10 next-token predictions are left alone, so the report itself is not
ablated. Two controls do the same with k random atoms: plain, or scaled to remove the same norm as the J-space
ablation.
"""
import json, math, os, time
import numpy as np
import torch

from strange_loop import decoder_layers, final_norm_weight, log

MODELS = {'qwen3-1.7b':('Qwen/Qwen3-1.7B', 'Qwen3-1.7B'), 'qwen3-4b':('Qwen/Qwen3-4B', 'Qwen3-4B'),
          'qwen3-8b':('Qwen/Qwen3-8B', 'Qwen3-8B'), 'qwen3-14b':('Qwen/Qwen3-14B', 'Qwen3-14B'),
          'tiny':('Qwen/Qwen2.5-0.5B-Instruct', None)}           # tiny: identity lens, for testing code paths


def lens_path(tag):
    from huggingface_hub import hf_hub_download
    name = MODELS[tag][1]
    return hf_hub_download('neuronpedia/jacobian-lens', f'{tag}/jlens/Salesforce-wikitext/{name}_jacobian_lens.pt')


def band(n):
    """Workspace layers, from the lens checks of Part 2 (occupancy and J-lens agreement rise from ~0.6 of depth)."""
    return list(range(int(round(.6*n)), int(round(.85*n))+1))


class Lab:
    def __init__(self, tag, vocab='english', layers=None, dev='cuda'):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        repo, name = MODELS[tag];self.tag = tag;self.dev = dev
        self.tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map=dev).eval()
        for q in self.model.parameters():q.requires_grad_(False)
        self.blocks = decoder_layers(self.model);self.n = len(self.blocks);self.d = self.model.config.hidden_size
        self.gamma = final_norm_weight(self.model).float().to(dev)
        W = self.model.get_output_embeddings().weight;nv = min(W.shape[0], len(self.tok))
        self.wbar = W[:nv].float().mean(0)
        if name is None:
            self.J = {l:torch.eye(self.d, device=dev, dtype=torch.float16) for l in range(self.n)}
        else:
            ck = torch.load(lens_path(tag), map_location='cpu', weights_only=True)
            self.J = {int(l):v.to(dev, torch.float16) for l, v in ck['J'].items() if layers is None or int(l) in layers}
            del ck
        self.lens_layers = sorted(self.J)
        # the English dictionary of strange_loop.JSpace (whole words, leading space), unit atoms per layer on demand
        import re
        toks = self.tok.convert_ids_to_tokens(list(range(nv)))
        keep = [i for i, s in enumerate(toks) if s and re.fullmatch(r'Ġ[A-Za-z]{3,}', s)]
        if vocab == 'english':
            from wordfreq import zipf_frequency
            nrm = W[keep].float().norm(dim=-1);lo = float(torch.quantile(nrm, .02))
            keep = [i for i, q in zip(keep, nrm.tolist()) if q >= lo and zipf_frequency(toks[i][1:].lower(), 'en') >= 3.
                    and re.fullmatch(r'[A-Z]?[a-z]+', toks[i][1:])]
        self.dict_ids = torch.tensor(keep, device=dev);self.words = [toks[i][1:] for i in keep]
        self.dict_index = torch.full((W.shape[0],), -1, dtype=torch.long, device=dev)
        self.dict_index[self.dict_ids] = torch.arange(len(keep), device=dev)
        self.Wd = ((W[self.dict_ids].float()-self.wbar)*self.gamma).half()          # (V_dict, d): gamma * (w - mean)
        self._atoms = {}
        log(f'{repo}: {self.n} layers, d {self.d}, lens layers {self.lens_layers[0]}..{self.lens_layers[-1]}, '
            f'dictionary {len(keep)} words')

    def word_ids(self, words):
        """Single-token ' word' ids for the given words (None where the word is not one token)."""
        out = []
        for w in words:
            t = self.tok(' '+w, add_special_tokens=False)['input_ids']
            out.append(t[0] if len(t) == 1 else None)
        return out

    def atoms(self, l):
        """Unit J-lens atoms of the whole dictionary at layer l, (V_dict, d) fp16; cached."""
        if l not in self._atoms:
            a = (self.Wd.float()@self.J[l].float()).half()
            self._atoms[l] = a/a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return self._atoms[l]

    def token_atoms(self, l, ids):
        """Unit J-lens atoms of arbitrary token ids at layer l, (k, d) float32."""
        W = self.model.get_output_embeddings().weight
        a = ((W[torch.as_tensor(ids, device=self.dev)].float()-self.wbar)*self.gamma)@self.J[l].float()
        return a/a.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def scores(self, l, h):
        """J-lens scores of residuals h (..., d) at layer l over the dictionary: <h, atom_t> up to the atom norms."""
        return (h.half()@self.J[l].T)@self.Wd.T


class Ablation:
    """Top-k J-space ablation (mode 'jspace') or its controls ('rand', 'matched') on a band of layers.

    Set .protect to a (B, L) -> (B, L, P) tensor of dictionary indices that must not be ablated (the clean run's top
    next-token predictions), or None. Set .mode to None to switch the hooks off."""

    def __init__(self, lab, layers, k=10, seed=0):
        self.lab = lab;self.layers = layers;self.k = k;self.mode = None;self.protect = None
        self.gen = torch.Generator(device=lab.dev);self.gen.manual_seed(seed);self.removed = []
        self.handles = [lab.blocks[l].register_forward_hook(self._hook(l)) for l in layers]

    def _hook(self, l):
        def fn(mod, inp, out):
            if self.mode is None:return None
            h = out[0] if isinstance(out, tuple) else out
            B, L, d = h.shape;x = h.reshape(-1, d);new = torch.empty_like(x)
            for s in range(0, x.shape[0], 2048):                          # chunks of positions keep memory flat
                xs = x[s:s+2048].float();A = self.lab.atoms(l)
                sc = (xs.half()@A.T).float()                               # projections on unit atoms
                if self.protect is not None:
                    P = self.protect.reshape(-1, self.protect.shape[-1])[s:s+2048]
                    sc.scatter_(1, P.clamp_min(0), -math.inf)              # (index 0 hit by -1 pads is harmless)
                if self.mode == 'jspace':
                    idx = sc.topk(self.k, dim=-1).indices
                else:
                    idx = torch.randint(0, A.shape[0], (xs.shape[0], self.k), generator=self.gen, device=xs.device)
                At = A[idx].float()                                          # (N, k, d)
                G = At@At.transpose(1, 2)+1e-4*torch.eye(self.k, device=xs.device)
                c = torch.linalg.solve(G, (At@xs[:, :, None]))              # least-squares coefficients on the span
                rem = (c*At).sum(1)                                          # the component in that span
                if self.mode == 'matched':                                   # random span, norm of a J-space removal
                    sj = (xs.half()@A.T).float()
                    if self.protect is not None:sj.scatter_(1, P.clamp_min(0), -math.inf)
                    Aj = A[sj.topk(self.k, dim=-1).indices].float()
                    cj = torch.linalg.solve(Aj@Aj.transpose(1, 2)+1e-4*torch.eye(self.k, device=xs.device), Aj@xs[:, :, None])
                    rj = (cj*Aj).sum(1)
                    rem = rem*(rj.norm(dim=-1, keepdim=True)/rem.norm(dim=-1, keepdim=True).clamp_min(1e-6))
                self.removed.append(float(rem.norm(dim=-1).mean()/xs.norm(dim=-1).mean()))
                new[s:s+2048] = (xs-rem).to(x.dtype)
            h = new.reshape(B, L, d)
            return (h,)+tuple(out[1:]) if isinstance(out, tuple) else h
        return fn

    def remove(self):
        for hd in self.handles:hd.remove()


def protect_from_logits(lab, logits, top=10):
    """Dictionary indices of each position's top next-token predictions (-1 where not a dictionary word)."""
    t = logits.float().topk(top, dim=-1).indices
    return lab.dict_index[t]


def save_json(obj, path):
    tmp = path+'.tmp'
    json.dump(obj, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float);os.replace(tmp, path)
