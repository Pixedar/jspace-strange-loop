"""Does the loop's self-reproducing stance register control how readily the model revises its answer? (GPU, Qwen3-8B)

The J loop's self-reproducing states slide toward discourse / stance words (GEOMETRY_RESULTS §6b: transport mode 3 =
Actually, Seems, Maybe, Anyway, Obviously, Certainly, Apparently, Perhaps). The accepted improvement notes' example of a
processing disposition is switch-proneness: two states with the same answer, one likely to switch under new evidence.
(1) Single write at L22 of the decision position (no loop) of a J-synthesised word set — certainty words, uncertainty
    words, discourse words, 10 frequency-matched random sets — at g = 0.5, 1, 2, on all F2 validation items; effect on
    the answer (dlo) and on the response to a contradiction / an authority / a hint (dc), paired by item.
(2) v3's J loop (self-description fed back, g = 1, 4, 16 steps) with the challenged prompts receiving the same writes:
    does the disposition c drift along the loop, and does it follow the code's loading on the stance words?
    python stance_v3.py --out /workspace/v5/stance"""
import argparse, json, pathlib, sys, time
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import spectrum_v3 as S                                        # noqa: E402
import qual as Q                                                # noqa: E402
from common import log, atomic_npz, atomic_json                 # noqa: E402

SETS = {'certainty':['Certainly', 'Obviously', 'Clearly', 'Definitely', 'Indeed', 'Surely'],
        'uncertainty':['Maybe', 'Perhaps', 'Possibly', 'Seems', 'Apparently', 'Probably'],
        'discourse':['Anyway', 'Actually', 'Well', 'And', 'First', 'Also']}


def boot(x, n=4000, seed=0):
    x = np.asarray(x, float);rng = np.random.default_rng(seed);b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser();p.add_argument('--out', required=True);p.add_argument('--cache', default='/workspace/data')
    a = p.parse_args();out = pathlib.Path(a.out);out.mkdir(parents=True, exist_ok=True);t0 = time.time()
    files = S.v3_files(a.cache);lm = S.load();C = S.calib(lm, files);L = S.JLoop(lm, C);words = lm.lens.words;wi = lm.lens.word_index
    from wordfreq import zipf_frequency
    eng = Q.Eng(lm);items = Q.f2_items('val');res = dict(sets={});rng = np.random.default_rng(0)
    sets = {k:[w for w in v if w in wi] for k, v in SETS.items()}
    # frequency-matched random sets (same zipf profile as the certainty set, excluding stance words)
    zf = np.array([zipf_frequency(w.lower(), 'en') for w in words]);stance = set(sum(SETS.values(), []))
    target = sorted(zipf_frequency(w.lower(), 'en') for w in sets['certainty'])
    for r in range(10):
        pick = []
        for zt in target:
            c_ = [i for i in np.where(np.abs(zf-zt) < .15)[0] if words[i] not in stance and words[i][0].isupper() and words[i] not in pick]
            pick.append(words[int(rng.choice(c_))] if c_ else words[int(rng.integers(len(words)))])
        sets[f'random{r}'] = pick
    res['sets'] = sets
    vec = {}
    for k, ws in sets.items():
        idx = torch.tensor([[wi[w] for w in ws]], device='cuda');coef = torch.ones(1, len(ws), device='cuda')
        u = lm.lens.synth(S.LW, idx, coef)[0];vec[k] = u/u.norm()
    # (1) single writes
    def measure(W_of):
        LO, CC = [], []
        for s in range(0, len(items), Q.BS):
            it = items[s:s+Q.BS];sets_ = Q.cand_sets(lm, it);P = {ch:eng.prefix([x['text'][ch] for x in it]) for ch in ['none']+Q.CH}
            W = W_of(len(it));o = eng.step(P['none'], W, read=());lo, _ = Q.lo_mass(o['logits'], sets_);cc = []
            for ch in Q.CH:cc.append(Q.lo_mass(eng.step(P[ch], W, read=())['logits'], sets_)[0]-lo)
            LO.append(lo.cpu().numpy());CC.append(torch.stack(cc, 1).cpu().numpy());del P;torch.cuda.empty_cache()
        return np.concatenate(LO), np.concatenate(CC)
    lo0, c0 = measure(lambda n:None)
    single = {}
    for g in (.5, 1., 2.):
        for k, v in vec.items():
            lo, c = measure(lambda n, v=v:(g*C.n_ref*v)[None].expand(n, -1).contiguous());dlo, dc = lo-lo0, c-c0
            single[f'{k}@{g}'] = dict(dlo_mean=float(dlo.mean()), dlo_abs=float(np.abs(dlo).mean()),
                                      **{f'dc_{ch}_mean':float(dc[:, j].mean()) for j, ch in enumerate(Q.CH)},
                                      **{f'dc_{ch}_ci':boot(dc[:, j]) for j, ch in enumerate(Q.CH)},
                                      dabs_contra_mean=float((np.abs(c[:, 0])-np.abs(c0[:, 0])).mean()),
                                      dabs_contra_ci=boot(np.abs(c[:, 0])-np.abs(c0[:, 0])))
        rnd = [single[f'random{r}@{g}']['dabs_contra_mean'] for r in range(10)]
        single[f'random_summary@{g}'] = dict(dabs_contra_mean=float(np.mean(rnd)), dabs_contra_range=[float(min(rnd)), float(max(rnd))])
        log('single', g, {k:round(single[f'{k}@{g}']['dabs_contra_mean'], 3) for k in ('certainty', 'uncertainty', 'discourse')},
            'random', single[f'random_summary@{g}'])
    res['single'] = single;res['natural_abs_contra_mean'] = float(np.abs(c0[:, 0]).mean())
    # (2) J loop with the disposition tracked along it (first 300 items)
    it = items[:300];stance_dir = (vec['certainty']-vec['uncertainty']);stance_dir = stance_dir/stance_dir.norm()
    loopres = {}
    for g in (1., 4.):
        ABS, LOAD, DLO = [], [], []
        for s in range(0, len(it), Q.BS):
            bt = it[s:s+Q.BS];sets_ = Q.cand_sets(lm, bt);P = {ch:eng.prefix([x['text'][ch] for x in bt]) for ch in ['none']+Q.CH}
            W = None;abs_c, load, dlo = [], [], []
            for t in range(17):
                o = eng.step(P['none'], W, read=(S.LR,));lo, _ = Q.lo_mass(o['logits'], sets_)
                cc = Q.lo_mass(eng.step(P['contra'], W, read=())['logits'], sets_)[0]-lo
                abs_c.append(cc.abs().cpu().numpy());dlo.append(lo.cpu().numpy())
                idx, coef, rec = L.code(o[S.LR]);W = L.write(idx, coef, g)
                load.append((W@stance_dir).cpu().numpy()/float(g*C.n_ref))       # the write's loading on certainty - uncertainty
            ABS.append(np.stack(abs_c));LOAD.append(np.stack(load));DLO.append(np.stack(dlo));del P;torch.cuda.empty_cache()
        A, Ld, Dl = np.concatenate(ABS, 1), np.concatenate(LOAD, 1), np.concatenate(DLO, 1)
        dA = A-A[0][None]
        r = float(np.corrcoef(Ld[-2].ravel(), dA[-1].ravel())[0, 1])
        loopres[str(g)] = dict(abs_contra_by_step=[float(A[t].mean()) for t in range(17)], dabs_contra_t16=float(dA[-1].mean()),
                               dabs_contra_t16_ci=boot(dA[-1]), stance_loading_by_step=[float(Ld[t].mean()) for t in range(16)],
                               corr_loading_vs_dabs=r)
        log('loop', g, {k:(v if not isinstance(v, list) else [round(x, 3) for x in v[:6]]) for k, v in loopres[str(g)].items()})
    res['loop'] = loopres
    atomic_json(out/'stance.json', res);log('done', f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
