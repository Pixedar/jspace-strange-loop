"""Check a Jacobian lens the way the paper does, before trusting it, and find the workspace band.

For every few layers, on held-out wikitext:
  agree_J / agree_logit  how often the lens's top token (J-lens, and the plain logit lens for comparison) matches the
                         model's own next-token prediction. Not a pass/fail test: J sums over all later target
                         positions, so it reads what a residual is disposed to say later, not next (on Qwen3-1.7B the
                         reference package gives Germany / EU / boots for "the country shaped like a boot", where the
                         logit lens gives "-shaped");
  probe                  the J-lens and logit-lens top tokens on a few concept prompts, at every tested layer;
  occupancy              the number of J-lens vectors worth using: grow a non-negative pursuit on the mean-subtracted
                         residual one atom at a time; occupancy is the K at which K J-lens vectors explain the most
                         variance in excess of K random directions (which explain K/d on average: the paper's "same-size
                         random directions"; plateau ~25 in the workspace there);
  excess_ve              variance explained by K = 25 J-lens atoms minus 25/d (paper: never above 10 %).

    python validate_lens.py --model Qwen/Qwen3-8B --jlens Qwen3-8B_jacobian_lens.pt --out val_qwen3-8b.json
"""
import argparse, json, math, time
import numpy as np
import torch

from strange_loop import decoder_layers, final_norm_weight, log


def wikitext(tok, n, length, seed=0):
    from datasets import load_dataset
    ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1', split='test')
    text = '\n'.join(t for t in ds['text'] if len(t) > 200)
    ids = tok(text, add_special_tokens=False)['input_ids']
    rng = np.random.default_rng(seed);starts = rng.choice(len(ids)-length, n, replace=False)
    return torch.tensor(np.stack([ids[s:s+length] for s in starts]))


@torch.no_grad()
def pursuit_gains(x, D, k):
    """Non-negative orthogonal matching pursuit of x (B, d) on unit atoms D (V, d): the share of energy explained
    after each of k atoms, with an exact non-negative least-squares refit at every step."""
    from scipy.optimize import nnls
    B = x.shape[0];r = x.clone();idx = torch.zeros((B, 0), dtype=torch.long, device=x.device);e0 = (x*x).sum(-1)
    out = []
    for j in range(k):
        c = (r.half()@D.T).float()
        if idx.shape[1]:c.scatter_(1, idx, -math.inf)
        idx = torch.cat([idx, c.argmax(-1, keepdim=True)], 1)
        A = D[idx].float();G = (A@A.transpose(1, 2)+1e-5*torch.eye(j+1, device=x.device)).double().cpu().numpy()
        bb = (A@x[:, :, None])[:, :, 0].double().cpu().numpy();coef = np.zeros_like(bb)
        for i in range(B):
            Lc = np.linalg.cholesky(G[i]);coef[i] = nnls(Lc.T, np.linalg.solve(Lc, bb[i]))[0]
        r = x-(torch.tensor(coef, dtype=torch.float32, device=x.device)[:, :, None]*A).sum(1)
        out.append((1-(r*r).sum(-1)/e0).mean().item())
    return np.array(out)


PROBES = ['Fact: The currency used in the country shaped like a boot is',
          'The capital of the country where the Eiffel Tower stands is',
          'She closed her eyes and thought about the ocean, the waves and the',
          'I keep thinking about my own thoughts, and about what it feels like to be']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True);p.add_argument('--jlens', required=True);p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=48);p.add_argument('--len', type=int, default=128)
    p.add_argument('--every', type=float, default=.05, help='layer spacing as a fraction of depth');p.add_argument('--K', type=int, default=40)
    a = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map='cuda').eval()
    blocks = decoder_layers(model);n_layers = len(blocks);dev = 'cuda'
    ck = torch.load(a.jlens, map_location='cpu', weights_only=True);Js = ck['J']
    gamma = final_norm_weight(model).float();W = model.get_output_embeddings().weight
    norm = model.model.norm
    special = set(tok.all_special_ids);keep = torch.tensor([i for i in range(tok.vocab_size) if i not in special], device=dev)
    ids = wikitext(tok, a.n, a.len).to(dev)
    probe_ids = [tok(t, return_tensors='pt')['input_ids'].to(dev) for t in PROBES]
    hs = {};store = {}
    hooks = [blocks[l].register_forward_hook(lambda m, i, o, l=l:store.__setitem__(l, (o[0] if isinstance(o, tuple) else o).detach()))
             for l in range(n_layers)]
    t0 = time.time();res = []
    layers = sorted({min(n_layers-1, int(round(f*n_layers))) for f in np.arange(a.every, 1., a.every)} & set(Js))
    g = torch.Generator(device=dev);g.manual_seed(0)
    bs = 8 if model.config.hidden_size*n_layers < 300_000 else 4                  # no_grad: the 32B OOM was an unused autograd graph
    for b0 in range(0, a.n, bs):
        with torch.no_grad():out = model(input_ids=ids[b0:b0+bs], use_cache=False, logits_to_keep=ids.shape[1]-16)   # positions 16..T-1
        final = out.logits[:, :-1].argmax(-1);del out
        for l in layers:hs.setdefault(l, []).append(store[l][:, 16:-1].float().cpu())   # CPU: a 32B model fills the GPU
        hs.setdefault('final', []).append(final);store.clear()
    probe_h = []
    for pi in probe_ids:
        with torch.no_grad():model(input_ids=pi, use_cache=False, logits_to_keep=1)
        probe_h.append({l:store[l][0, -2].float() for l in layers});store.clear()
    for h in hooks:h.remove()
    final = torch.cat(hs.pop('final')).reshape(-1);torch.cuda.empty_cache()
    for l in layers:
        H = torch.cat(hs[l]).reshape(-1, model.config.hidden_size).to(dev)             # (n positions, d)
        J = Js[l].float().to(dev)
        sel = torch.randperm(H.shape[0], generator=g, device=dev)[:2048]
        lj = model.lm_head(norm((H[sel]@J.T).to(torch.bfloat16))).argmax(-1)
        ll = model.lm_head(norm(H[sel].to(torch.bfloat16))).argmax(-1)
        agree_J = (lj == final[sel]).float().mean().item();agree_L = (ll == final[sel]).float().mean().item()
        # J-lens atoms (rows of W_U diag(gamma) J, unit length) against a same-size random dictionary
        D = torch.cat([(lambda a_:(a_/a_.norm(dim=-1, keepdim=True).clamp_min(1e-6)).half())((W[keep[i:i+8192]].float()*gamma)@J)
                       for i in range(0, len(keep), 8192)])
        X = H[sel[:256]]-H.mean(0);dd = X.shape[-1]
        eJ = pursuit_gains(X, D, a.K);eR = np.arange(1, a.K+1)/dd               # K random directions: K/d on average
        occ = int(np.argmax(eJ-eR))+1
        top = lambda v:[tok.decode([t]).strip() for t in model.lm_head(norm(v.to(torch.bfloat16)))[0].topk(8).indices]
        probe = {PROBES[i][:40]:dict(J=top((ph[l]@J.T)[None]), logit=top(ph[l][None])) for i, ph in enumerate(probe_h)}
        res.append(dict(layer=l, depth=l/n_layers, agree_J=agree_J, agree_logit=agree_L, occupancy=occ, probe=probe,
                        ve_J25=float(eJ[24]), ve_R25=float(eR[24]), excess_ve=float(eJ[24]-eR[24]),
                        curve_J=eJ.round(4).tolist(), curve_R=eR.round(4).tolist()))
        log(f'layer {l:3d} ({l/n_layers:.2f})  lens agree J {agree_J:.2f} logit {agree_L:.2f} | occupancy {occ:2d} | '
            f'VE@25 J {eJ[24]:.3f} random {eR[24]:.3f} excess {eJ[24]-eR[24]:+.3f} | boot: {probe[PROBES[0][:40]]["J"][:5]}')
        del D, J, H;torch.cuda.empty_cache()
    json.dump(dict(model=a.model, jlens=a.jlens, n_layers=n_layers, n_prompts=int(ck.get('n_prompts', -1)), layers=res),
              open(a.out, 'w'), indent=1)
    log(f'done in {time.time()-t0:.0f} s')


if __name__ == '__main__':
    main()
