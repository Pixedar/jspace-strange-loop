"""Why does the J loop converge where it does? The write->read word transport operator (GPU, frozen Qwen3-8B).

The loop maps a J code (words + coefficients) to a write at L22 and reads a new code at L29. Linearising around the
natural state, every word a written at L22 changes the read-layer projection onto every word b: M[b, a]. Iterating the
loop with a normalised write is power iteration on (a sparse, non-negative version of) M, so the loop's high-gain fixed
points should sit near M's leading eigenvectors. This script measures M over N common words in C contexts and compares
its leading eigenvectors and self-gains with the attractor words observed in v3 (GEOMETRY_RESULTS.md §4).
    python transport_v3.py --out /workspace/v5/transport [--n_words 3000] [--g 1]"""
import argparse, collections, json, pathlib, sys, time
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import spectrum_v3 as S                                     # noqa: E402
from common import log, atomic_npz, atomic_json              # noqa: E402


@torch.no_grad()
def main():
    p = argparse.ArgumentParser();p.add_argument('--out', required=True);p.add_argument('--cache', default='/workspace/data')
    p.add_argument('--n_words', type=int, default=3000);p.add_argument('--n_ctx', type=int, default=8);p.add_argument('--g', type=float, default=1.)
    a = p.parse_args();out = pathlib.Path(a.out);out.mkdir(parents=True, exist_ok=True);t0 = time.time()
    files = S.v3_files(a.cache);ctxs = json.loads(open(files['contexts.json']).read())
    sel = S.pick_contexts(ctxs)[::12][:a.n_ctx]
    lm = S.load(True);C = S.calib(lm, files);L = S.JLoop(lm, C);words = lm.lens.words     # fp32: clean differences
    from wordfreq import zipf_frequency
    zf = np.array([zipf_frequency(w.lower(), 'en') for w in words]);cand = np.argsort(-zf)[:a.n_words]
    Dw = L.Dw[torch.tensor(cand, device='cuda')].float();Dr = L.D[torch.tensor(cand, device='cuda')].float()   # unit atoms
    M = torch.zeros(a.n_words, a.n_words, device='cuda');selfg = torch.zeros(a.n_words, device='cuda')
    for ci in sel:
        P = L.prefix([ctxs[ci]['text']]);h0, _ = L.read(P, None);del P
        for s in range(0, a.n_words, 128):
            n = min(128, a.n_words-s);Pb = L.prefix([ctxs[ci]['text']]*n)
            W = a.g*C.n_ref*Dw[s:s+n];h, _ = L.read(Pb, W);del Pb
            M[:, s:s+n] += ((h-h0)@Dr.T).T/len(sel)                   # read-atom projections of the change, per written atom
        torch.cuda.empty_cache();log('context', ci, f'{time.time()-t0:.0f}s')
    M = M/(a.g*C.n_ref)                                               # response per unit write
    diag = torch.diagonal(M).cpu().numpy();Mc = M.cpu().numpy().astype(np.float64)
    ev, vec = np.linalg.eig(Mc);o = np.argsort(-ev.real);ev, vec = ev[o], vec[:, o]
    top_eigs = []
    for j in range(6):
        v = vec[:, j].real;v = v if v.sum() >= 0 else -v;ix = np.argsort(-v)[:15]
        top_eigs.append(dict(eig=[float(ev[j].real), float(ev[j].imag)], words=[words[cand[i]] for i in ix]))
    # power iteration with the loop's non-negativity (coefficients clipped at 0, normalised), from 20 random starts
    rng = np.random.default_rng(0);finals = collections.Counter()
    Mt = M.cpu().numpy()
    for r in range(20):
        x = np.abs(rng.standard_normal(a.n_words))
        for _ in range(200):x = np.clip(Mt@x, 0, None);x = x/max(np.linalg.norm(x), 1e-12)
        for i in np.argsort(-x)[:10]:finals[words[cand[i]]] += 1
    res = dict(n_words=a.n_words, n_ctx=len(sel), g=a.g, self_gain_top=[(words[cand[i]], float(diag[i])) for i in np.argsort(-diag)[:40]],
               self_gain_bottom=[(words[cand[i]], float(diag[i])) for i in np.argsort(diag)[:20]], leading_eigvecs=top_eigs,
               nonneg_power_iteration_words=finals.most_common(30), spectral_radius=float(np.abs(ev).max()))
    atomic_npz(out/'transport.npz', M=Mt.astype(np.float32), cand=cand);atomic_json(out/'transport.json', res)
    log('self-gain top', res['self_gain_top'][:20]);log('eig', [(e['eig'][0], e['words'][:8]) for e in top_eigs[:3]])
    log('power iteration', res['nonneg_power_iteration_words'][:20]);log('done', f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
