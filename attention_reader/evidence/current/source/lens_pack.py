"""A compact, exact form of lens_score.py's kind scores, so they can be computed without the model's output map: the
final norm weight, the mean and covariance of the output-map rows of the whole-word vocabulary, and each kind's mean
row. For a vector v and u = gam * v, a kind's score is (kind_mu . u - mu . u) / std, with std**2 = n/(n-1) u'Cov u
(the standard deviation over whole words, as torch.std).
Usage: python lens_pack.py MODEL OUT_NPZ  (make it, where the model is cached) | python lens_pack.py score PACK VEC_NPZ"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lexicons import token_sets  # noqa: E402


def load_maps(repo):
    import json
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    try:
        wmap = json.load(open(hf_hub_download(repo, "model.safetensors.index.json")))["weight_map"]
    except Exception:                                            # a single-file model
        wmap = None
    got = {}
    for k in ("lm_head.weight", "model.norm.weight"):
        with safe_open(hf_hub_download(repo, wmap[k] if (wmap and k in wmap) else "model.safetensors"), "pt") as fh:
            got[k] = fh.get_tensor(k if k in fh.keys() else "model.embed_tokens.weight").float()   # tied: the embedding
    return got["lm_head.weight"], got["model.norm.weight"]


def kind_scores(pack, V):
    """V (N, d) -> (N, kinds), as lens_score.py."""
    u = V * pack["gam"][None]
    n = float(pack["n_words"])
    std = np.sqrt(np.einsum("ni,ij,nj->n", u, pack["cov"], u) * n / (n - 1.0))
    return (u @ pack["kind_mu"].T - (u @ pack["mu"])[:, None]) / np.maximum(std, 1e-8)[:, None]


if __name__ == "__main__":
    if sys.argv[1] == "score":
        P = dict(np.load(sys.argv[2]))
        V = np.load(sys.argv[3])
        for k in V.files:
            print(k, np.round(kind_scores(P, V[k].astype(np.float64)), 3))
        sys.exit()
    from transformers import AutoTokenizer
    repo, out = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(repo)
    W_U, gam = load_maps(repo)
    toks = tok.convert_ids_to_tokens(list(range(min(len(tok), W_U.shape[0]))))
    words = [i for i, w in enumerate(toks) if w is not None and w.startswith("Ġ") and w[1:].isalpha() and w[1:].isascii()
             and len(w) >= 3]
    WK = W_U[torch.tensor(words)].double()
    if torch.cuda.is_available():
        WK = WK.cuda()
    mu = WK.mean(0)
    C = (WK - mu).T @ (WK - mu) / WK.shape[0]
    where = {i: j for j, i in enumerate(words)}
    kind_mu = torch.stack([WK[torch.tensor([where[i] for i in v if i in where], device=WK.device)].mean(0)
                           for v in token_sets(tok).values()])
    np.savez_compressed(out, gam=gam.double().numpy(), mu=mu.cpu().numpy(), cov=C.cpu().numpy().astype(np.float32),
                        kind_mu=kind_mu.cpu().numpy(), n_words=len(words), kinds=np.array(list(token_sets(tok))))
    # check against the direct computation on random vectors
    P = dict(np.load(out))
    g = torch.Generator().manual_seed(0)
    V = torch.randn(5, W_U.shape[1], generator=g).double()
    direct = []
    for v in V:
        lg = (WK @ (gam.double().to(WK.device) * v.to(WK.device)))
        z = (lg - lg.mean()) / lg.std()
        direct.append([float(z[torch.tensor([where[i] for i in s if i in where], device=WK.device)].mean())
                       for s in token_sets(tok).values()])
    print("pack vs direct, max difference:", float(np.abs(kind_scores(P, V.numpy()) - np.array(direct)).max()))
    print("saved", out, len(words), "whole words")
