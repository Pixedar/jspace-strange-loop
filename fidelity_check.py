"""Can the self-model read the trajectory, and does the answer depend on the J-space dictionary?

Takes a run made with --save_raw and measures reflection fidelity (does the self-model pick out its own last window
among all minds?) in four spaces built from the same residual states:
  raw            the centred residual states themselves (no J-space at all)
  paper          J-space with the paper's dictionary: every token, rows of W_U diag(gamma) J, K = 25
  paper16        the same with K = 16
  english        common English words, centred unembedding, K = 16 (the dictionary of runs B and C)

    python fidelity_check.py runs/ladder/qwen3-1.7b/diag --model Qwen/Qwen3-1.7B --jlens <lens.pt>
"""
import argparse, json
import numpy as np
import torch

from strange_loop import JSpace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run');p.add_argument('--model', required=True);p.add_argument('--jlens', required=True)
    a = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    Z = np.load(f'{a.run}/traj.npz');M = json.load(open(f'{a.run}/meta.json', encoding='utf-8'))
    K, L = M['args']['K'], M['layer']
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map='cuda').eval()
    J = torch.load(a.jlens, map_location='cpu', weights_only=True)['J'][L].float().numpy()
    spaces = {'paper':JSpace(model, tok, J, k=25, vocab='full', center=False), 'paper16':None,
              'english':JSpace(model, tok, J, k=16, vocab='english', center=True)}
    X = torch.tensor(Z['raw_x'].astype(np.float32), device='cuda')                   # (N, B, d)
    N, B, d = X.shape;NR = N//K
    selfs = {k[4:]:torch.tensor(Z[k].astype(np.float32), device='cuda') for k in Z.files if k.startswith('raw_') and k != 'raw_x'}
    unit = lambda v:v/v.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def windows(C):                                                                 # leaky state, mean per window
        zz = torch.zeros_like(C);s = torch.zeros_like(C[0])
        for t in range(N):s = .6*s+.4*C[t];zz[t] = s
        return unit(torch.stack([zz[r*K:(r+1)*K].mean(0) for r in range(NR)]))

    def retrieval(W, S):
        hit = [(unit(S[r])@W[r].T).argmax(1).eq(torch.arange(B, device='cuda')).float().mean().item() for r in range(1, NR)]
        return float(np.mean(hit))

    def jcontent(js, V, k=None):
        flat = V.reshape(-1, d);out = torch.cat([js.decompose(flat[i:i+256], k)[2] for i in range(0, len(flat), 256)])
        out = out.reshape(V.shape);return out-out.reshape(-1, d).mean(0)              # relative to the typical part

    res = {}
    Wr = windows(X)
    for name, S in selfs.items():res.setdefault(name, {})['raw'] = retrieval(Wr, S)
    for sp in ('paper', 'paper16', 'english'):
        js = spaces['paper'] if sp == 'paper16' else spaces[sp];k = 16 if sp == 'paper16' else None
        Wj = windows(jcontent(js, X, k))
        for name, S in selfs.items():res[name][sp] = retrieval(Wj, jcontent(js, S, k))
    print(f'chance {1/B:.3f}   (rows: reflection setting; columns: space)')
    for name, v in res.items():print(f'  {name:14s} ' + '  '.join(f'{k} {x:.2f}' for k, x in v.items()))
    json.dump(dict(chance=1/B, fidelity=res), open(f'{a.run}/fidelity_check.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
