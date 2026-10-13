"""Self-deflation: the J loop without its own dominant self-reproduction modes (declared Regime-C-style mechanism).

The v3 J loop is power iteration on the model's write->read word-transport operator M (transport_v3.py): it converges on
M's leading modes (answering frame, Q&A structure, stance, casual register) and drops what reproduces itself least
(entities — the evidence). Self-deflation removes from every write its projection on the top-m modes of the system's
OWN operator (orthonormalised in the write space), then renormalises to the same broadcast norm. Nothing external is
measured or rewarded: the loop is denied only the self-description it already reproduces best ("move toward the rarer
path", original comments). Compared with the plain loop (m = 0) and with deflation of m random directions of the same
write-space span (control), at g = 1 and 4 on 96 contexts (v3 calibration):
  evidence kept in the workspace (grouped decoding of A vs B from the J code z over steps), relational structure (RSA of z
  with the natural code), late dynamics (bf16 step size, period-2 ratio), code overlap between contexts, attractor words.
    python deflate_v3.py --out /workspace/v5/deflate [--modes 0,1,3,6]"""
import argparse, collections, json, pathlib, sys, time
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import spectrum_v3 as S                                        # noqa: E402
import qual as Q                                                # noqa: E402
from common import log, atomic_npz, atomic_json                 # noqa: E402


def decode_acc(Z, y, groups):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    acc = []
    for tr, te in GroupKFold(5).split(Z, y, groups):
        if len(set(y[tr])) < 2:continue
        m = LogisticRegression(C=1., max_iter=3000).fit(Z[tr], y[tr]);acc.append(float((m.predict(Z[te]) == y[te]).mean()))
    return float(np.mean(acc)) if acc else float('nan')


@torch.no_grad()
def main():
    p = argparse.ArgumentParser();p.add_argument('--out', required=True);p.add_argument('--cache', default='/workspace/data')
    p.add_argument('--modes', default='0,1,3,6');p.add_argument('--T', type=int, default=32)
    a = p.parse_args();out = pathlib.Path(a.out);out.mkdir(parents=True, exist_ok=True);t0 = time.time()
    files = S.v3_files(a.cache);ctxs = json.loads(open(files['contexts.json']).read())
    sel = [i for i, c in enumerate(ctxs) if c['fam'] in ('F2', 'F3', 'F6') and c['tag'] in ('A', 'B')][:240]
    lm = S.load();C = S.calib(lm, files);L = S.JLoop(lm, C);words = lm.lens.words;eng = Q.Eng(lm)
    tz = np.load('/workspace/v5/transport/transport.npz');M = tz['M'].astype(np.float64);cand = tz['cand']
    ev, vec = np.linalg.eig(M);o = np.argsort(-ev.real)
    Dw = L.Dw[torch.tensor(cand, device='cuda')].float()                        # write-layer atoms of the 3 000 words
    U = [];gen = torch.Generator(device='cuda');gen.manual_seed(11)
    for k in range(6):
        v = torch.tensor(vec[:, o[k]].real, device='cuda', dtype=torch.float32);u = v@Dw;U.append(u/u.norm())
    U = torch.stack(U);Qm, _ = torch.linalg.qr(U.T)                            # orthonormal basis of the mode span (d, 6)
    R = torch.randn(6, Dw.shape[0], device='cuda', generator=gen)@Dw;Qr, _ = torch.linalg.qr(R.T)   # random directions in the same atom span
    y = np.array([0 if ctxs[i]['tag'] == 'A' else 1 for i in sel]);groups = np.array([ctxs[i]['sid'] for i in sel])
    fam = np.array([ctxs[i]['fam'] for i in sel]);res = {}
    for g in (1., 4.):
        for kind in ['plain']+[f'deflate{m}' for m in [int(x) for x in a.modes.split(',') if int(x) > 0]]+['random6']:
            if kind == 'plain':B = None
            elif kind == 'random6':B = Qr[:, :6]
            else:B = Qm[:, :int(kind[7:])]
            Zs, Is, Cs = [], [], []
            for s in range(0, len(sel), 48):
                ii = sel[s:s+48];P = L.prefix([ctxs[i]['text'] for i in ii]);h, _ = L.read(P, None);idx, coef, rec = L.code(h)
                Z, I, Cf = [L.z(rec).cpu().numpy()], [idx.cpu().numpy()], [coef.cpu().numpy()]
                for t in range(a.T):
                    W = lm.lens.synth(S.LW, idx, coef)
                    if B is not None:W = W-(W@B)@B.T                               # remove the loop's own dominant modes
                    W = g*C.n_ref*W/W.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    h, _ = L.read(P, W);idx, coef, rec = L.code(h)
                    Z.append(L.z(rec).cpu().numpy());I.append(idx.cpu().numpy());Cf.append(coef.cpu().numpy())
                Zs.append(np.stack(Z));Is.append(np.stack(I));Cs.append(np.stack(Cf));del P;torch.cuda.empty_cache()
            Z, I, Cf = np.concatenate(Zs, 1), np.concatenate(Is, 1), np.concatenate(Cs, 1)
            dec = {t:decode_acc(Z[t], y, groups) for t in (0, 1, 2, 4, 8, 16, a.T)}
            from scipy.spatial.distance import pdist
            from scipy.stats import spearmanr
            rsa = {t:float(spearmanr(pdist(Z[0]), pdist(Z[t])).correlation) for t in (1, 4, 16, a.T)}
            late = Z[-9:].astype(np.float64);step = float(np.median(np.abs(np.diff(late, axis=0)).max(axis=(0, 2))))
            d1 = np.linalg.norm(np.diff(late, axis=0), axis=-1);d2 = np.linalg.norm(late[2:]-late[:-2], axis=-1)
            r2 = float(np.median(np.median(d2/np.maximum(d1[1:], 1e-12), 0)))
            wc = collections.Counter()
            for i in range(I.shape[1]):
                for j in np.argsort(-Cf[-1, i])[:5]:wc[words[int(I[-1, i, j])]] += 1
            ov = []                                                             # code overlap between contexts at the end
            for i in range(0, I.shape[1], 3):
                for j in range(i+1, min(I.shape[1], i+30), 3):ov.append(len(set(I[-1, i])&set(I[-1, j]))/25)
            res[f'{kind}@{g}'] = dict(decode_by_step=dec, rsa_with_natural=rsa, late_step_median=step, period2_ratio=r2,
                                      code_overlap_end=float(np.mean(ov)), top_words_end=wc.most_common(20))
            log(kind, g, {k:v for k, v in res[f'{kind}@{g}'].items() if k != 'top_words_end'}, '\n   words', wc.most_common(10))
            atomic_json(out/'deflate.json', res)
    log('done', f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
