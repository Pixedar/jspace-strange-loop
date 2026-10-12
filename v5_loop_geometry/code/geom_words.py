"""What keeps moving in the 'fixed' states of the v3 loops, in words (GEOMETRY_PLAN.md M6/M9 follow-up). CPU only.

For each late window (last 32 steps) of J-loop and self-model trajectories:
* period-2 index  r2 = median_t ||z_t - z_{t-2}|| / ||z_t - z_{t-1}||   (r2 << 1: an alternation between two states)
* per-atom coefficient series (0 when the atom is absent); oscillating atoms = largest share of their variance at the
  period-2 frequency; substitution pairs = atoms present on alternate steps.
Aggregated over trajectories: the most frequent oscillating words and substitution pairs, and whether the oscillating
atoms are the strongest atoms of the code or marginal ones.
    python geom_words.py --root <.../v3/e2/qwen3-8b/main> --model Qwen/Qwen3-8B --out <dir> [--bc BC]"""
import argparse, collections, json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from geom_v3 import words_of_model   # noqa: E402


def analyse(z, idx, coef, words, late=32):
    T = z.shape[0]-1;Z = z[T-late+1:].astype(np.float64);I = idx[T-late+1:];Cf = coef[T-late+1:].astype(np.float64)
    n = Z.shape[1];r2s, osc_w, pair_w, rank_osc, amp_osc = [], collections.Counter(), collections.Counter(), [], []
    p2share = []
    for i in range(n):
        d1 = np.linalg.norm(Z[1:, i]-Z[:-1, i], axis=-1);d2 = np.linalg.norm(Z[2:, i]-Z[:-2, i], axis=-1)
        ok = d1[1:] > 1e-9
        if ok.sum() < 4:continue
        r2s.append(float(np.median(d2[ok]/d1[1:][ok])))
        atoms = sorted(set(I[:, i].ravel().tolist()));pos = {a:j for j, a in enumerate(atoms)}
        S = np.zeros((late, len(atoms)))
        for t in range(late):
            for a, c in zip(I[t, i].tolist(), Cf[t, i].tolist()):S[t, pos[a]] = c
        D = S-S.mean(0);var = D.var(0)
        if var.sum() <= 0:continue
        alt = np.where(np.arange(late) % 2 == 0, 1., -1.)
        p2 = (D*alt[:, None]).mean(0)**2                       # power at the period-2 frequency per atom
        p2share.append(float(p2.sum()/var.sum()))
        top = np.argsort(-var)[:3];mean_rank = np.argsort(np.argsort(-S.mean(0)))
        for j in top:
            osc_w[words[atoms[j]]] += 1;rank_osc.append(int(mean_rank[j]));amp_osc.append(float(np.sqrt(var[j])/max(S.mean(0).max(), 1e-12)))
        pres = (S > 0).astype(int)
        for j in range(len(atoms)):
            pj = pres[:, j]
            if 0 < pj.sum() < late and np.all(pj[1:] != pj[:-1]):      # present on alternate steps exactly
                for k in range(j+1, len(atoms)):
                    if np.all(pres[:, k] == 1-pj):pair_w[(words[atoms[j]], words[atoms[k]])] += 1
    return dict(n=int(n), r2_median=float(np.median(r2s)) if r2s else None, frac_r2_lt_0_3=float(np.mean(np.array(r2s) < .3)) if r2s else None,
                period2_power_share_median=float(np.median(p2share)) if p2share else None,
                oscillating_words=osc_w.most_common(25), substitution_pairs=[(list(k), v) for k, v in pair_w.most_common(15)],
                oscillating_atom_rank_median=float(np.median(rank_osc)) if rank_osc else None,
                oscillating_amp_rel_median=float(np.median(amp_osc)) if amp_osc else None)


def load(root, stage, pat, conds):
    G = collections.defaultdict(lambda:collections.defaultdict(list))
    for f in sorted((root/stage).glob(pat)):
        meta = json.loads(f.with_suffix('.json').read_text());a = np.load(f);z, idx, coef = a['z'], a['idx'], a['coef']
        for i, r in enumerate(meta['rows']):
            if r['cond'] not in conds:continue
            if r.get('seed', 0) != 0:continue
            d = G[(r['cond'], r['g'])];d['z'].append(z[:, i]);d['idx'].append(idx[:, i]);d['coef'].append(coef[:, i])
    return {k:{f:np.stack(v, 1) for f, v in d.items()} for k, d in G.items()}


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True);p.add_argument('--out', required=True)
    p.add_argument('--model', default=None);p.add_argument('--bc', default='BC')
    a = p.parse_args();root = pathlib.Path(a.root);out = pathlib.Path(a.out);out.mkdir(parents=True, exist_ok=True)
    prov = json.loads((root/'provenance.json').read_text());words = words_of_model(a.model or prov['model'], prov['revision'])
    R = {}
    for k, d in sorted(load(root, 'A1', 'A1_std_*.npz', ('J',)).items(), key=str):
        R[f'A1|J@{k[1]}'] = analyse(d['z'], d['idx'], d['coef'], words);print('A1', k, R[f'A1|J@{k[1]}'], flush=True)
    if (root/a.bc).exists():
        for k, d in sorted(load(root, a.bc, 'BC*_std_*.npz', ('B', 'C1', 'R', 'Y')).items(), key=str):
            R[f'{a.bc}|{k[0]}@{k[1]}'] = analyse(d['z'], d['idx'], d['coef'], words);print(a.bc, k, R[f'{a.bc}|{k[0]}@{k[1]}'], flush=True)
    (out/'late_words.json').write_text(json.dumps(R, indent=1, default=float));print('written', out/'late_words.json', flush=True)


if __name__ == '__main__':
    main()
