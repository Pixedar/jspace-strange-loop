"""WHERE_ITS_OWN_STATE_GOES.md section 96 (diagnostic for a new reader; laptop, read only): would a picture that simply
keeps what a copy reads carry what is individual in it? On the milestone's twelve lives with what each copy read logged
(section 89's learned-only lives, identical to the milestone), tokens 300-2999:
- the read itself (256 numbers per copy, what its attention takes from the window's values);
- c8's picture (its trained picture map);
- a transparent picture: the leading 32 directions of what the copies read (fitted once on all twelve lives' reads,
  copies and tokens pooled), the read projected onto them;
for each: how far apart the copies are (share of the spread over time that lies between them), how much of it two
shared prototypes explain, and how often its two states follow the commonest genre.
Usage: python reader96.py ROOT STATES_S2_NPZ"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attractor86 import sphere_means  # noqa: E402

root = sys.argv[1]
G = np.load(sys.argv[2])["genre"][0]
FROM, SEEDS, INDIV = 300, range(10, 22), (10, 12, 17, 20)


def apart(X):
    return float(((X - X.mean(1, keepdims=True)) ** 2).sum() / ((X - X.mean((0, 1), keepdims=True)) ** 2).sum())


def two_states(X):
    """Share of the spread the two nearest shared prototypes explain, and the states' agreement with the genre."""
    U = X / np.linalg.norm(X, axis=-1, keepdims=True)
    C = sphere_means(U.reshape(-1, U.shape[-1]), 2)
    lab = (U @ C.T).argmax(-1)
    flat, near = U.reshape(-1, U.shape[-1]), C[lab.reshape(-1)]
    expl = 1.0 - float(((flat - near) ** 2).sum() / ((flat - flat.mean(0)) ** 2).sum())
    g = G[FROM:3000][:, None] == np.bincount(G[FROM:3000]).argmax()
    a = float((g == (lab == 0)).mean())
    return expl, max(a, 1.0 - a)


reads, pics = {}, {}
for sd in SEEDS:
    z = np.load(os.path.join(root, f"pj_c8_SIx_seed{sd}", "life.npz"))
    reads[sd] = z["rd"][FROM:3000].astype(np.float64)
    pics[sd] = z["pic"][FROM:3000].astype(np.float64)
pool = np.concatenate([r.reshape(-1, r.shape[-1]) for r in reads.values()])
mu = pool.mean(0)
_, S, Vt = np.linalg.svd(pool - mu, full_matrices=False)
W = Vt[:32]
kept = float((S[:32] ** 2).sum() / (S ** 2).sum())
print(f"the transparent picture keeps {kept:.0%} of the spread of what the copies read (32 of 256 directions)")
rows = []
for sd in SEEDS:
    tp = (reads[sd] - mu) @ W.T
    r = dict(seed=sd, read=apart(reads[sd]), c8=apart(pics[sd]), tp=apart(tp))
    r["c8_two"], r["c8_genre"] = two_states(pics[sd])
    r["tp_two"], r["tp_genre"] = two_states(tp)
    rows.append(r)
    tag = "individual" if sd in INDIV else "shared"
    print(f"   seed {sd} ({tag}): copies apart - read {r['read']:.1%}, c8 picture {r['c8']:.1%}, transparent picture "
          f"{r['tp']:.1%}; two shared states explain c8 {r['c8_two']:.0%} / transparent {r['tp_two']:.0%}; follow the "
          f"genre c8 {r['c8_genre']:.0%} / transparent {r['tp_genre']:.0%}")
for label, sds in (("individual lives", INDIV), ("shared lives", [s for s in SEEDS if s not in INDIV])):
    v = [r for r in rows if r["seed"] in sds]
    print(f"   {label}: copies apart - read {np.mean([r['read'] for r in v]):.1%}, c8 picture {np.mean([r['c8'] for r in v]):.1%}, "
          f"transparent picture {np.mean([r['tp'] for r in v]):.1%}; two states explain c8 {np.mean([r['c8_two'] for r in v]):.0%} "
          f"/ transparent {np.mean([r['tp_two'] for r in v]):.0%}")
