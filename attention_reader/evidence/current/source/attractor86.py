"""WHERE_ITS_OWN_STATE_GOES.md section 86: the milestone's lives with the picture reduced to prototypes.
  python attractor86.py ref ROOT    configs for S and Sn, rerun with the picture logged (section 46's lives)
  python attractor86.py snap ROOT   prototypes from those lives (one: the copies' mean picture; two: spherical two-means
                                    over tokens 300-2999 of all four copies), and configs for S1, S2, Sn1, Sn2"""
import json
import os
import sys

import numpy as np

SRC = "D:/jspace_v11_runs/core32_schema45"
SEEDS = range(10, 16)
FROM = 300


def sphere_means(X, k, restarts=20, iters=100, seed=86):
    """Spherical k-means on the rows of X (n, p): centres of unit length; the best of `restarts` by total cosine."""
    U = X / np.linalg.norm(X, axis=-1, keepdims=True)
    if k == 1:
        c = U.mean(0)
        return (c / np.linalg.norm(c))[None]
    rng = np.random.default_rng(seed)
    best, best_fit = None, -np.inf
    for _ in range(restarts):
        C = U[rng.choice(len(U), k, replace=False)]
        for _ in range(iters):
            lab = (U @ C.T).argmax(-1)
            Cn = np.stack([U[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            Cn = Cn / np.linalg.norm(Cn, axis=-1, keepdims=True)
            if np.allclose(Cn, C):
                break
            C = Cn
        fit = float((U @ C.T).max(-1).sum())
        if fit > best_fit:
            best, best_fit = C, fit
    return best


def main():
    mode, root = sys.argv[1], sys.argv[2]
    names = []
    for sd in SEEDS:
        for arm in ("S", "Sn"):
            base = json.load(open(f"{SRC}/ph_c8_{arm}_seed{sd}/run.json"))
            if mode == "ref":
                name = f"pa_c8_{arm}_seed{sd}"
                cfg = dict(base, out=f"{root}/{name}")
                os.makedirs(cfg["out"], exist_ok=True)
                json.dump(cfg, open(f"{cfg['out']}/run.json", "w"), indent=1)
                names.append(name)
                continue
            z = np.load(f"{root}/pa_c8_{arm}_seed{sd}/life.npz")
            X = z["pic"][FROM:].astype(np.float64).reshape(-1, z["pic"].shape[-1])
            for k in (1, 2):
                C = sphere_means(X, k)
                name = f"pa_c8_{arm}{k}_seed{sd}"
                out = f"{root}/{name}"
                os.makedirs(out, exist_ok=True)
                np.save(f"{out}/protos.npy", C.astype(np.float32))
                ps = dict(base["process_self"], snap={"protos": f"{out}/protos.npy", "from": FROM})
                cfg = dict(base, out=out, process_self=ps)
                json.dump(cfg, open(f"{out}/run.json", "w"), indent=1)
                names.append(name)
    fn = f"{root}/names_{mode}.txt"
    open(fn, "w").write("\n".join(names) + "\n")
    print(f"{len(names)} lives -> {fn}")


if __name__ == "__main__":
    main()
