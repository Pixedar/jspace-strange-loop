"""WHERE_ITS_OWN_STATE_GOES.md section 97: the milestone with a transparent picture (its own schema, S; a neighbour's,
Sn; seeds 10-21) against the milestone's lives. Written before the run; readings as fixed in section 97.
Usage: python crit97.py ROOT LENS_PACK_NPZ STATES_S2_NPZ"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attractor86 import sphere_means  # noqa: E402
from lens_pack import kind_scores  # noqa: E402

root, pack = sys.argv[1], dict(np.load(sys.argv[2]))
G = np.load(sys.argv[3])["genre"][0]
P95, SEG, FROM, SEEDS = 0.728, 100, 300, range(10, 22)
MILE = "D:/jspace_v11_runs/core32_schema45/ph_c8_{}_seed{}/life.npz"
corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731


def own_other(L, k_lo, k_hi):
    own, oth = [], []
    for k1 in range(k_lo, k_hi + 1):
        for k2 in range(k1 + 3, k_hi + 1):
            for b in range(L.shape[1]):
                own.append(corr(L[k1, b], L[k2, b]))
                oth.append(np.mean([corr(L[k1, b], L[k2, c]) for c in range(L.shape[1]) if c != b]))
    return float(np.mean(own) - np.mean(oth))


def organization(z):
    """crit45.organization, unchanged; plus persistence within each half of the shared text."""
    sg = np.array([np.bincount(G[i:i + SEG]).argmax() for i in range(0, 3000, SEG)])
    L = z["lens_seg"].astype(np.float64)
    own_m, oth_m, same, diff = [], [], [], []
    for k1 in range(3, L.shape[0]):
        for k2 in range(k1 + 3, L.shape[0]):
            for b in range(L.shape[1]):
                o = corr(L[k1, b], L[k2, b])
                own_m.append(o)
                oth_m.append(np.mean([corr(L[k1, b], L[k2, c]) for c in range(L.shape[1]) if c != b]))
                (same if sg[k1] == sg[k2] else diff).append(o)
    sc = kind_scores(pack, z["lens_late"].astype(np.float64))
    s = z["s"].astype(np.float64)
    u = s[255] / np.linalg.norm(s[255], axis=-1, keepdims=True)
    same_pairs = sum(float(u[a] @ u[b]) > 0.999 for a in range(4) for b in range(a + 1, 4))
    return dict(persist=float(np.mean(own_m) - np.mean(oth_m)), respond=float(np.mean(same) - np.mean(diff)),
                kinds=int((np.abs(sc).max(-1) > P95).sum()), apart=float(np.nanmean(z["att_apart"][1500:])),
                early=own_other(L, 3, 14), late=own_other(L, 15, 29), pasts=same_pairs == 0,
                turn=float(np.nanmean(z["turn"][FROM:])))


def picture_extra(z):
    X = z["pic"][FROM:3000].astype(np.float64)
    U = X / np.linalg.norm(X, axis=-1, keepdims=True)
    lab = (U @ sphere_means(U.reshape(-1, U.shape[-1]), 2).T).argmax(-1)
    g = G[FROM:3000][:, None] == np.bincount(G[FROM:3000]).argmax()
    a = float((g == (lab == 0)).mean())
    return dict(genre=max(a, 1.0 - a),
                pic_apart=float(((X - X.mean(1, keepdims=True)) ** 2).sum() / ((X - X.mean((0, 1), keepdims=True)) ** 2).sum()))


res = {}
for sd in SEEDS:
    for arm in ("S", "Sn"):
        if os.path.exists(MILE.format(arm, sd)):
            res["M" + arm, sd] = organization(dict(np.load(MILE.format(arm, sd))))
        p_ = os.path.join(root, f"pt_c8_{arm}_seed{sd}", "life.npz")
        if os.path.exists(p_):
            z = dict(np.load(p_))
            res["T" + arm, sd] = organization(z) | picture_extra(z)
print(f"lives found: {len(res)} (MS, MSn = the milestone with its own / a neighbour's schema; TS, TSn = the transparent picture)")

print("\nPer arm (means over the lives found): persistence, responsiveness, kinds, attention apart; persistence in the first "
      "and second half; lives where the pasts register; individual lives (attention apart > 0.1); stance turn per token")
for arm in ("MS", "MSn", "TS", "TSn"):
    v = [res[arm, sd] for sd in SEEDS if (arm, sd) in res]
    if v:
        mn = lambda k: float(np.mean([x[k] for x in v]))  # noqa: E731
        extra = (f"; pictures apart {mn('pic_apart'):.3f}; genre {min(x['genre'] for x in v):.0%} lowest"
                 if "genre" in v[0] else "")
        print(f"   {arm:3s}: {mn('persist'):+.3f} {mn('respond'):+.3f} {sum(x['kinds'] for x in v)}/{4 * len(v)} "
              f"{mn('apart'):.3f}; halves {mn('early'):+.3f} / {mn('late'):+.3f}; pasts register {sum(x['pasts'] for x in v)} "
              f"of {len(v)}; individual {sum(x['apart'] > 0.1 for x in v)} of {len(v)}; turn {mn('turn'):.4f}{extra}")
for sd in SEEDS:
    print(f"   seed {sd}: " + "  ".join(f"{arm} {res[arm, sd]['persist']:+.3f}/{res[arm, sd]['respond']:+.3f}/{res[arm, sd]['apart']:.3f}"
                                       for arm in ("MS", "MSn", "TS", "TSn") if (arm, sd) in res)
          + "   (persistence/responsiveness/attention apart)")

print("\nReadings (fixed in section 97)")
sds = [sd for sd in SEEDS if all((a, sd) in res for a in ("MS", "TS", "TSn"))]
if sds:
    ind_T = sum(res["TS", sd]["apart"] > 0.1 for sd in sds)
    ind_M = sum(res["MS", sd]["apart"] > 0.1 for sd in sds)
    dp = float(np.mean([res["TS", sd]["persist"] - res["MS", sd]["persist"] for sd in sds]))
    dr = float(np.mean([res["TS", sd]["respond"] - res["MS", sd]["respond"] for sd in sds]))
    own = sum(res["TS", sd]["persist"] > res["TSn", sd]["persist"] for sd in sds)
    ratio = float(np.mean([res["TS", sd]["apart"] for sd in sds])) / max(float(np.mean([res["MS", sd]["apart"] for sd in sds])), 1e-9)
    lower = sum(res["TS", sd]["persist"] < res["MS", sd]["persist"] - 0.02 for sd in sds)
    shaped = ind_T >= max(8, 2 * ind_M) and dp >= -0.02 and own >= 9
    print(f"   individual lives {ind_T} of {len(sds)} (milestone {ind_M}); persistence against the milestone {dp:+.3f}; own "
          f"schema above a neighbour's in {own} of {len(sds)} -> the histories shape the copies: {shaped}")
    print(f"   variety (attention apart x{ratio:.2f} with persistence {dp:+.3f}): {ratio > 1.5 and dp < -0.02}")
    print(f"   broken loop (persistence lower by > 0.02 in {lower} of {len(sds)}, attention apart x{ratio:.2f}): "
          f"{lower >= 7 and ratio <= 1.0}")
    ind = [sd for sd in sds if res["TS", sd]["apart"] > 0.1]
    fade_T = sum(res["TS", sd]["late"] < res["TS", sd]["early"] for sd in ind)
    fade_M = sum(res["MS", sd]["late"] < res["MS", sd]["early"] for sd in sds if res["MS", sd]["apart"] > 0.1)
    print(f"   fading imprint: second half below the first in {fade_T} of {len(ind)} individual lives (milestone: {fade_M} of "
          f"{ind_M}) -> {len(ind) > 0 and fade_T > len(ind) / 2 and not fade_M > ind_M / 2}")
    print(f"   responsiveness against the milestone {dr:+.3f} (a fall beyond 0.02 means following the text less)")
