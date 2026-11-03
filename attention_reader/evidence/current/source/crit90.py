"""WHERE_ITS_OWN_STATE_GOES.md section 90: knowledge of its interpreting, acting on its looking. Written before the run.
Arms: L (its own model leans its looking), Ln (a model learned from its neighbour), S+ (the attention schema at double
strength, no model of interpreting), Lswap / Lshift (copies 0 and 1 exchange their models, or have theirs changed at
random, tokens 1000-1999; individual lives only). Reference: the milestone's life (section 89's learned-only lives).
Readings A-D as fixed in section 90; the individual lives (seeds 10, 12, 17, 20) are read separately.
Usage: python crit90.py ROOT LENS_PACK_NPZ STATES_S2_NPZ"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attractor86 import sphere_means  # noqa: E402
from lens_pack import kind_scores  # noqa: E402

root, pack = sys.argv[1], dict(np.load(sys.argv[2]))
G = np.load(sys.argv[3])["genre"][0]
P95, SEG, FROM, LO = 0.728, 100, 300, 500
SEEDS, INDIV = range(10, 22), (10, 12, 17, 20)
NAME = {"M": "pj_c8_SIx", "L": "pk_c8_L", "Ln": "pk_c8_Ln", "S+": "pk_c8_Splus", "Lswap": "pk_c8_Lswap",
        "Lshift": "pk_c8_Lshift"}
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


def seg_persistence(X):
    T = X.shape[0] // SEG * SEG
    L = X[:T].reshape(-1, SEG, X.shape[1], X.shape[2]).mean(1)
    L = L - L.mean(1, keepdims=True)
    own, oth = [], []
    for k1 in range(3, L.shape[0]):
        for k2 in range(k1 + 3, L.shape[0]):
            for b in range(L.shape[1]):
                own.append(corr(L[k1, b], L[k2, b]))
                oth.append(np.mean([corr(L[k1, b], L[k2, c]) for c in range(L.shape[1]) if c != b]))
    return float(np.mean(own) - np.mean(oth))


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


def measures(z):
    m = organization(z)
    m["pic_apart"], m["pic_persist"] = apart(z["pic"]), seg_persistence(z["pic"][:3000].astype(np.float64))
    m["read_apart"] = apart(z["rd"]) if "rd" in z else np.nan
    m["genre"] = genre_agree(z["pic"])
    Q = z["Q"].reshape(z["Q"].shape[0], -1).astype(np.float64)
    m["habit_apart"] = float(np.mean([np.linalg.norm(Q[a] - Q[b]) for a in range(len(Q)) for b in range(a + 1, len(Q))])
                             / np.linalg.norm(Q, axis=-1).mean())
    if "i_err" in z:
        m["r2_own"] = 1.0 - float(np.nansum(z["i_err"][LO:]) / np.nansum(z["i_tot"][LO:]))
        m["r2_other"] = 1.0 - float(np.nansum(z["i_err_other"][LO:]) / np.nansum(z["i_tot"][LO:]))
    else:
        m["r2_own"] = m["r2_other"] = np.nan
    m["moved"] = float(np.nanmean(z["i_moved"][FROM:])) if "i_moved" in z and np.isfinite(z["i_moved"]).any() else np.nan
    L = z["lens_seg"].astype(np.float64)
    m["like01"] = {"during": float(np.mean([corr(L[k, 0], L[k, 1]) for k in range(10, 20)])),
                   "after": float(np.mean([corr(L[k, 0], L[k, 1]) for k in range(20, 30)]))}
    return m


res, st = {}, {}
for sd in SEEDS:
    for arm, nm in NAME.items():
        p_ = os.path.join(root, f"{nm}_seed{sd}", "life.npz")
        if os.path.exists(p_):
            z = dict(np.load(p_))
            res[arm, sd], st[arm, sd] = measures(z), z["s"]
have = lambda arm, sds: [sd for sd in sds if (arm, sd) in res]  # noqa: E731
print(f"lives found: {len(res)}")

print("\n0. Checks (first token at which the stance differs)")
for sd in SEEDS:
    line = []
    for arm, base in (("L", "M"), ("Ln", "M"), ("S+", "M"), ("Lswap", "L"), ("Lshift", "L")):
        if (arm, sd) in res and (base, sd) in res:
            d = np.abs(st[arm, sd] - st[base, sd]).reshape(st[arm, sd].shape[0], -1).max(-1)
            line.append(f"{arm} from {base}: {int(np.argmax(d > 0)) if (d > 0).any() else 'never'}")
    print(f"   seed {sd}: " + "; ".join(line))

ARMS = ("M", "L", "Ln", "S+")
for label, sds in (("the individual lives (seeds 10, 12, 17, 20)", INDIV), ("the shared lives", [s for s in SEEDS if s not in INDIV])):
    print(f"\n== {label}   (M = the milestone)")
    print("   arm: persistence responsiveness kinds attention-apart | pictures-apart picture-persistence reads-apart "
          "habits-apart | genre min | model R2 own/another's | lean moves attention")
    for arm in ARMS:
        v = [res[arm, sd] for sd in have(arm, sds)]
        if v:
            mn = lambda k: float(np.nanmean([x[k] for x in v]))  # noqa: E731
            print(f"   {arm:3s}: {mn('persist'):+.3f} {mn('respond'):+.3f} {sum(x['kinds'] for x in v):2d}/{4 * len(v)} "
                  f"{mn('apart'):.3f} | {mn('pic_apart'):.3f} {mn('pic_persist'):+.3f} {mn('read_apart'):.4f} "
                  f"{mn('habit_apart'):.3f} | {min(x['genre'] for x in v):.0%} | {mn('r2_own'):+.2f}/{mn('r2_other'):+.2f} | "
                  f"{mn('moved'):.3f}  ({len(v)} lives)")
    for sd in sds:
        print(f"   seed {sd}: " + "  ".join(f"{arm} {res[arm, sd]['persist']:+.3f}/{res[arm, sd]['respond']:+.3f}/"
                                           f"{res[arm, sd]['apart']:.3f}/{res[arm, sd]['pic_apart']:.3f}/{res[arm, sd]['pic_persist']:+.3f}"
                                           for arm in ARMS if (arm, sd) in res)
              + "   (persistence/responsiveness/attention apart/pictures apart/picture persistence)")

print("\n== The exchange (individual lives; how alike copies 0 and 1's take-up profiles are, against L)")
moved = back = n_ex = 0
for sd in INDIV:
    if all((a, sd) in res for a in ("L", "Lswap", "Lshift")):
        n_ex += 1
        dl = {a: {w: res[a, sd]["like01"][w] - res["L", sd]["like01"][w] for w in ("during", "after")} for a in ("Lswap", "Lshift")}
        mv = dl["Lswap"]["during"] > max(dl["Lshift"]["during"], 0.0)
        bk = dl["Lswap"]["after"] < dl["Lswap"]["during"]
        moved += mv
        back += bk
        print(f"   seed {sd}: likeness in L {res['L', sd]['like01']['during']:+.3f}; during: exchanged {dl['Lswap']['during']:+.3f}, "
              f"random change {dl['Lshift']['during']:+.3f}; after: exchanged {dl['Lswap']['after']:+.3f}")
print(f"   came closer than under the random change in {moved} of {n_ex}; fell back after in {back} of {n_ex}")

print("\n== Readings (fixed in section 90)")
ind = [sd for sd in INDIV if all((a, sd) in res for a in ("M", "L", "Ln"))]
p_up = sum(res["L", sd]["persist"] > res["Ln", sd]["persist"] for sd in ind)
a_up = sum(res["L", sd]["apart"] > res["Ln", sd]["apart"] for sd in ind)
d_p = float(np.mean([res["L", sd]["persist"] - res["M", sd]["persist"] for sd in ind])) if ind else np.nan
d_r = float(np.mean([res["L", sd]["respond"] - res["M", sd]["respond"] for sd in ind])) if ind else np.nan
allL = have("L", SEEDS)
genre_ok = all(res["L", sd]["genre"] >= 0.85 for sd in allL)
resp_ok = all(res["L", sd]["respond"] >= res["M", sd]["respond"] - 0.02 for sd in allL if ("M", sd) in res)
A = p_up >= 3 and a_up >= 3 and d_p >= -0.02 and d_r >= -0.02 and moved >= 3 and back >= 3 and genre_ok and resp_ok
print(f"   A: own above neighbour's in persistence {p_up} of {len(ind)}, attention apart {a_up} of {len(ind)}; against the "
      f"milestone: persistence {d_p:+.3f}, responsiveness {d_r:+.3f}; exchange {moved} and {back} of {n_ex}; genre in every "
      f"life {genre_ok}; responsiveness in every life {resp_ok} -> {A}")
ind_s = [sd for sd in ind if ("S+", sd) in res]
pa = sum(res["L", sd]["pic_apart"] > res["S+", sd]["pic_apart"] for sd in ind_s)
pp = sum(res["L", sd]["pic_persist"] > res["S+", sd]["pic_persist"] for sd in ind_s)
print(f"   B: pictures further apart than under S+ in {pa} of {len(ind_s)}, picture persistence higher in {pp} of "
      f"{len(ind_s)} -> {A and pa >= 3 and pp >= 3}")
for sd in [s for s in SEEDS if s not in INDIV]:
    if ("L", sd) in res and ("M", sd) in res and res["L", sd]["apart"] > 1.5 * res["M", sd]["apart"]:
        coh = res["L", sd]["read_apart"] > res["M", sd]["read_apart"] and ("Ln", sd) in res and res["L", sd]["persist"] > res["Ln", sd]["persist"]
        print(f"   C: shared seed {sd}: attention apart {res['M', sd]['apart']:.3f} -> {res['L', sd]['apart']:.3f}; coherent "
              f"(reads apart up, own above neighbour's) {coh}")
D = all(abs(res["L", sd]["persist"] - res["M", sd]["persist"]) <= 0.01 and abs(res["L", sd]["apart"] - res["M", sd]["apart"]) <= 0.01
        for sd in ind)
print(f"   D: nothing changes (individual lives within 0.01 of the milestone in persistence and attention apart): {D}")
