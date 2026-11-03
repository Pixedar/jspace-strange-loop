"""WHERE_ITS_OWN_STATE_GOES.md section 97: the milestone with a transparent picture (DESIGN_A_PICTURE_OF_WHAT_IT_READS.md,
Direction 1).
  python reader97.py fit ROOT       the 32 leading directions of what the copies read (the milestone's twelve lives with
                                    their reads logged, all copies, tokens 16-2999 pooled), their mean and each one's
                                    spread; saved once in ROOT and never changed
  python reader97.py configs ROOT   the milestone's configuration with that picture: its own schema (S) and a
                                    neighbour's (Sn), seeds 10-21"""
import json
import os
import sys

import numpy as np

READS = "D:/jspace_v11_runs/core32_interp88/pj_c8_SIx_seed{}/life.npz"
SRC = "D:/jspace_v11_runs/core32_schema45/ph_c8_{}_seed{}/run.json"
SEEDS = range(10, 22)


def main():
    mode, root = sys.argv[1], sys.argv[2]
    os.makedirs(root, exist_ok=True)
    if mode == "fit":
        X = np.concatenate([np.load(READS.format(sd))["rd"][16:3000].astype(np.float64).reshape(-1, 256) for sd in SEEDS])
        mu = X.mean(0)
        _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
        W = Vt[:32]
        sd = ((X - mu) @ W.T).std(0)
        for nm, a in (("tp_W", W), ("tp_mu", mu), ("tp_sd", sd)):
            np.save(os.path.join(root, f"{nm}.npy"), a.astype(np.float32))
        print(f"fitted on {len(X)} reads; the 32 directions keep {float((S[:32] ** 2).sum() / (S ** 2).sum()):.1%}; "
              f"spread of the first 8: {np.round(sd[:8], 2)}")
        return
    names = []
    for sd in SEEDS:
        for arm in ("S", "Sn"):
            base = json.load(open(SRC.format(arm, sd)))
            name = f"pt_c8_{arm}_seed{sd}"
            pic = {"kind": "transparent", "W": f"{root}/tp_W.npy", "mu": f"{root}/tp_mu.npy", "sd": f"{root}/tp_sd.npy",
                   "k": 8}
            cfg = dict(base, out=f"{root}/{name}", process_self=dict(base["process_self"], picture=pic))
            os.makedirs(cfg["out"], exist_ok=True)
            json.dump(cfg, open(f"{cfg['out']}/run.json", "w"), indent=1)
            names.append(name)
    open(f"{root}/names.txt", "w").write("\n".join(names) + "\n")
    print(f"{len(names)} lives -> {root}/names.txt")


if __name__ == "__main__":
    main()
