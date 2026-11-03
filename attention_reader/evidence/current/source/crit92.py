"""WHERE_ITS_OWN_STATE_GOES.md section 92: the lean of section 90 at a quarter and at half strength, own model against a
neighbour's, in the four individual lives; with section 91's full-strength lives for the dose table. Written before
the run. Readings as fixed in section 92.
Usage: python crit92.py ROOT LENS_PACK_NPZ STATES_S2_NPZ"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attractor86 import sphere_means  # noqa: E402
from lens_pack import kind_scores  # noqa: E402

root, pack = sys.argv[1], dict(np.load(sys.argv[2]))
G = np.load(sys.argv[3])["genre"][0]
P95, SEG, FROM = 0.728, 100, 300
INDIV = (10, 12, 17, 20)
NAME = {"M": "pj_c8_SIx", "L25": "pk_c8_L25", "Ln25": "pk_c8_Ln25", "L50": "pk_c8_L50", "Ln50": "pk_c8_Ln50",
        "L100": "pk_c8_L", "Ln100": "pk_c8_Ln"}
corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731


def organization(z):
    """crit45.organization, unchanged."""
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
    return dict(persist=float(np.mean(own_m) - np.mean(oth_m)), respond=float(np.mean(same) - np.mean(diff)),
                kinds=int((np.abs(sc).max(-1) > P95).sum()), apart=float(np.nanmean(z["att_apart"][1500:])))


def apart(X):
    X = X[FROM:3000].astype(np.float64)
    return float(((X - X.mean(1, keepdims=True)) ** 2).sum() / ((X - X.mean((0, 1), keepdims=True)) ** 2).sum())


def genre_agree(pic):
    X = pic[FROM:3000].astype(np.float64)
    U = X / np.linalg.norm(X, axis=-1, keepdims=True)
    lab = (U @ sphere_means(U.reshape(-1, U.shape[-1]), 2).T).argmax(-1)
    g = G[FROM:3000][:, None] == np.bincount(G[FROM:3000]).argmax()
    a = float((g == (lab == 0)).mean())
    return max(a, 1.0 - a)


res = {}
for sd in INDIV:
    for arm, nm in NAME.items():
        p_ = os.path.join(root, f"{nm}_seed{sd}", "life.npz")
        if os.path.exists(p_):
            z = dict(np.load(p_))
            m = organization(z)
            m["pic_apart"], m["genre"] = apart(z["pic"]), genre_agree(z["pic"])
            m["moved"] = float(np.nanmean(z["i_moved"][FROM:])) if "i_moved" in z and np.isfinite(z["i_moved"]).any() else np.nan
            res[arm, sd] = m
print(f"lives found: {len(res)}")
print("\nThe individual lives, by strength (means over seeds 10, 12, 17, 20): persistence, responsiveness, attention apart, "
      "pictures apart, lean moves attention")
for arm in NAME:
    v = [res[arm, sd] for sd in INDIV if (arm, sd) in res]
    if v:
        mn = lambda k: float(np.nanmean([x[k] for x in v]))  # noqa: E731
        print(f"   {arm:5s}: {mn('persist'):+.3f} {mn('respond'):+.3f} {mn('apart'):.3f} {mn('pic_apart'):.3f} "
              f"{mn('moved'):.3f}; genre min {min(x['genre'] for x in v):.0%} ({len(v)} lives)")
for sd in INDIV:
    print(f"   seed {sd}: " + "  ".join(f"{arm} {res[arm, sd]['persist']:+.3f}/{res[arm, sd]['apart']:.3f}"
                                       for arm in NAME if (arm, sd) in res) + "   (persistence/attention apart)")

print("\nReadings (fixed in section 92)")
for lv in ("25", "50", "100"):
    L, Ln = "L" + lv, "Ln" + lv
    sds = [sd for sd in INDIV if all((a, sd) in res for a in ("M", L, Ln))]
    if not sds:
        continue
    up = sum(res[L, sd]["persist"] > res[Ln, sd]["persist"] for sd in sds)
    ap = sum(res[L, sd]["apart"] > res[Ln, sd]["apart"] for sd in sds)
    dp = float(np.mean([res[L, sd]["persist"] - res["M", sd]["persist"] for sd in sds]))
    dr = float(np.mean([res[L, sd]["respond"] - res["M", sd]["respond"] for sd in sds]))
    dpa = float(np.mean([res[L, sd]["pic_apart"] - res["M", sd]["pic_apart"] for sd in sds]))
    g_ok = all(res[L, sd]["genre"] >= 0.85 for sd in sds)
    r_ok = all(res[L, sd]["respond"] >= res["M", sd]["respond"] - 0.02 for sd in sds)
    A = up >= 3 and ap >= 3 and dp >= -0.02 and dr >= -0.02 and g_ok and r_ok
    chase = dpa > 0 and dp < -0.02
    print(f"   strength {int(lv) / 100:.2f}: own above neighbour's in persistence {up} of {len(sds)}, attention apart {ap} of "
          f"{len(sds)}; against the milestone persistence {dp:+.3f}, responsiveness {dr:+.3f}, pictures apart {dpa:+.3f}; "
          f"genre in every life {g_ok}; responsiveness in every life {r_ok} -> A' {A}; the chase {chase}")
