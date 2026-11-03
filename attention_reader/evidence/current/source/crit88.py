"""DESIGN_INTERPRETATION_SCHEMA.md / WHERE_ITS_OWN_STATE_GOES.md section 88: a model of its own interpreting, used
while it interprets. Written before the run. The individual lives (seeds 10, 12, 17, 20) and the eight shared ones are
read separately. Measures (the note's numbering):
1. knowledge (SIx): R2 of its own model from token 500, against another copy's;
2. its own interpretation: the copies' pictures and stances apart (share of the spread over tokens 300-2999), each
   copy's picture persistence (section 46's own-against-other measure on 100-token segments);
3. specific and repairable: the neighbour's model (SIn) against its own (SI); the exchange (copies 0 and 1, tokens
   1000-1999): how much closer their pictures come than in SI, against the random change, and whether that undoes
   itself in tokens 2000-2999;
4. development: the organization (crit45's measures) and how far apart the copies' habits end;
5. still reads the text: agreement of the picture's two states with the commonest genre against the rest;
   responsiveness against S's (SIx equals S);
6. coherent, not just different (made precise before the run): where SI or SI1 moves the copies' pictures apart by
   more than 10% against SIx, what they take (their reads) must move apart too.
Usage: python crit88.py ROOT LENS_PACK_NPZ STATES_S2_NPZ"""
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
ARMS = ("SIx", "SI", "SI1", "SIn", "SIswap", "SIshift")
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
    """Section 46's persistence on 100-token segment means of X (T, B, n), centred across copies (segments 3..)."""
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
    """The share of X's spread (tokens 300-2999) that lies between the copies."""
    X = X[FROM:3000].astype(np.float64)
    return float(((X - X.mean(1, keepdims=True)) ** 2).sum() / ((X - X.mean((0, 1), keepdims=True)) ** 2).sum())


def genre_agree(pic):
    X = pic[FROM:3000].astype(np.float64)
    U = X / np.linalg.norm(X, axis=-1, keepdims=True)
    C = sphere_means(U.reshape(-1, U.shape[-1]), 2)
    lab = (U @ C.T).argmax(-1)                                   # (T, B)
    g = G[FROM:3000][:, None] == np.bincount(G[FROM:3000]).argmax()
    a = float((g == (lab == 0)).mean())
    return max(a, 1.0 - a)


def measures(z):
    m = organization(z)
    m["r2_own"] = 1.0 - float(np.nansum(z["i_err"][LO:]) / np.nansum(z["i_tot"][LO:]))
    m["r2_other"] = 1.0 - float(np.nansum(z["i_err_other"][LO:]) / np.nansum(z["i_tot"][LO:]))
    m["pic_apart"], m["st_apart"] = apart(z["pic"]), apart(z["s"])
    m["pic_persist"] = seg_persistence(z["pic"][:3000].astype(np.float64))
    m["genre"] = genre_agree(z["pic"])
    Q = z["Q"].reshape(z["Q"].shape[0], -1).astype(np.float64)
    m["habit_apart"] = float(np.mean([np.linalg.norm(Q[a] - Q[b]) for a in range(len(Q)) for b in range(a + 1, len(Q))])
                             / np.linalg.norm(Q, axis=-1).mean())
    m["read_apart"] = apart(z["rd"]) if "rd" in z else np.nan
    m["turn"] = float(np.nanmean(z["turn"][FROM:]))              # (added after the individual lives were seen; beside
    m["pic_move"] = float(np.linalg.norm(np.diff(z["pic"][FROM:3000].astype(np.float64), axis=0), axis=-1).mean())   # the criteria)
    m["used"] = float(np.nanmean(z["i_used"][FROM:]) / np.sqrt(np.nanmean(z["i_tot"][FROM:]))) if np.isfinite(z["i_used"]).any() else np.nan
    P = z["pic"][:3000].astype(np.float64)
    m["d01"] = {w: float(np.linalg.norm(P[a:b, 0] - P[a:b, 1], axis=-1).mean()) for w, (a, b) in
                (("during", (1000, 2000)), ("after", (2000, 3000)))}
    return m


res, part = {}, {}
for sd in SEEDS:
    for arm in ARMS:
        p_ = os.path.join(root, f"pj_c8_{arm}_seed{sd}", "life.npz")
        if os.path.exists(p_):
            z = dict(np.load(p_))
            res[arm, sd] = measures(z)
            part[arm, sd] = z["s"]
have = lambda arm, sds: [sd for sd in sds if (arm, sd) in res]  # noqa: E731
print(f"lives found: {len(res)}")

print("\n0. Checks")
for sd in SEEDS:
    line = []
    ref = os.path.join("D:/jspace_v11_runs/core32_schema45", f"ph_c8_S_seed{sd}", "life.npz")
    if ("SIx", sd) in res and os.path.exists(ref):
        b_ = np.load(ref)
        line.append(f"SIx against S: max difference {max(float(np.abs(part['SIx', sd] - b_['s']).max()), 0.0):.1e}")
    for arm, base in (("SI", "SIx"), ("SI1", "SIx"), ("SIn", "SIx"), ("SIswap", "SI"), ("SIshift", "SI")):
        if (arm, sd) in res and (base, sd) in res:
            d = np.abs(part[arm, sd] - part[base, sd]).reshape(part[arm, sd].shape[0], -1).max(-1)
            line.append(f"{arm} parts from {base} at {int(np.argmax(d > 0)) if (d > 0).any() else 'never'}")
    print(f"   seed {sd}: " + "; ".join(line))

for label, sds in (("the individual lives (seeds 10, 12, 17, 20)", INDIV), ("the shared lives", [s for s in SEEDS if s not in INDIV])):
    print(f"\n== {label}")
    print("   arm    : persistence responsiveness kinds attention-apart | pictures-apart stances-apart picture-persistence "
          "reads-apart habits-apart | genre-agreement | model R2 own/another's | expectation size")
    for arm in ARMS:
        v = [res[arm, sd] for sd in have(arm, sds)]
        if not v:
            continue
        mn = lambda k: np.nanmean([x[k] for x in v])  # noqa: E731
        print(f"   {arm:7s}: {mn('persist'):+.3f} {mn('respond'):+.3f} {sum(x['kinds'] for x in v):2d}/{4 * len(v)} "
              f"{mn('apart'):.3f} | {mn('pic_apart'):.3f} {mn('st_apart'):.3f} {mn('pic_persist'):+.3f} {mn('read_apart'):.4f} "
              f"{mn('habit_apart'):.3f} | {min(x['genre'] for x in v):.0%} min | {mn('r2_own'):+.2f}/{mn('r2_other'):+.2f} | "
              f"{mn('used'):.2f}  ({len(v)} lives)")
    for sd in sds:
        print(f"   seed {sd}: " + "  ".join(f"{arm} {res[arm, sd]['persist']:+.3f}/{res[arm, sd]['respond']:+.3f}/"
                                           f"{res[arm, sd]['pic_apart']:.3f}/{res[arm, sd]['pic_persist']:+.3f}"
                                           for arm in ("SIx", "SI", "SI1", "SIn") if (arm, sd) in res)
              + "   (persistence/responsiveness/pictures apart/picture persistence)")
        print(f"            beside the criteria (added after the individual lives were seen): stance turn per token "
              + "  ".join(f"{arm} {res[arm, sd]['turn']:.4f}" for arm in ("SIx", "SI", "SI1", "SIn") if (arm, sd) in res)
              + "; picture change per token " + "  ".join(f"{arm} {res[arm, sd]['pic_move']:.3f}" for arm in ("SIx", "SI", "SI1", "SIn") if (arm, sd) in res))

print("\n== The exchange (copies 0 and 1; distance between their pictures, tokens 1000-1999 and 2000-2999)")
moved = back = 0
n_ex = 0
for sd in SEEDS:
    if all((a, sd) in res for a in ("SI", "SIswap", "SIshift")):
        d = {a: res[a, sd]["d01"] for a in ("SI", "SIswap", "SIshift")}
        like = lambda a, w: (d["SI"][w] - d[a][w]) / max(d["SI"][w], 1e-9)  # noqa: E731
        ls, lr, la = like("SIswap", "during"), like("SIshift", "during"), like("SIswap", "after")
        tag = "individual" if sd in INDIV else "shared"
        print(f"   seed {sd} ({tag}): distance in SI {d['SI']['during']:.3f}; closer during: exchanged {ls:+.0%}, random "
              f"change {lr:+.0%}; after: exchanged {la:+.0%}")
        if sd in INDIV:
            n_ex += 1
            moved += ls > max(lr, 0.0)
            back += la < ls
print(f"   individual lives: moved toward each other more than under the random change in {moved} of {n_ex}; "
      f"undid it after in {back} of {n_ex}")

print("\n== Readings (fixed in DESIGN_INTERPRETATION_SCHEMA.md section 5)")
ind = lambda arm: have(arm, INDIV)  # noqa: E731
know_all = all(res["SIx", sd]["r2_own"] >= 0.10 for sd in have("SIx", SEEDS))
know_own = sum(res["SIx", sd]["r2_own"] > res["SIx", sd]["r2_other"] for sd in ind("SIx"))
pp = sum(res["SI", sd]["pic_persist"] > res["SIn", sd]["pic_persist"] for sd in ind("SIn") if ("SI", sd) in res)
genre_ok = all(res[a, sd]["genre"] >= 0.85 for (a, sd) in res if a in ("SI", "SI1"))
resp = {a: np.mean([res[a, sd]["respond"] - res["SIx", sd]["respond"] for sd in have(a, SEEDS) if ("SIx", sd) in res])
        for a in ("SI", "SI1")}
pers = {a: np.mean([res[a, sd]["persist"] - res["SIx", sd]["persist"] for sd in have(a, SEEDS) if ("SIx", sd) in res])
        for a in ("SI", "SI1")}
print(f"   1. knowledge: R2 >= 0.10 in every life {know_all}; own above another's in {know_own} of {len(ind('SIx'))} individual lives")
print(f"   3. picture persistence own (SI) above neighbour's (SIn): {pp} of {len(ind('SIn'))} individual lives; "
      f"exchange moved {moved} of {n_ex}, undid {back} of {n_ex}")
print(f"   5. the picture's two states still follow the genre (>= 85% in every SI, SI1 life): {genre_ok}; "
      f"responsiveness against S: SI {resp['SI']:+.3f}, SI1 {resp['SI1']:+.3f}")
A = know_all and know_own >= 3 and pp >= 3 and moved >= 3 and back >= 3 and genre_ok and resp["SI"] >= -0.02
print(f"   -> a model of its own interpreting takes part in what a copy makes of the text: {A}")
po = sum(res["SI", sd]["persist"] > res["SIn", sd]["persist"] for sd in ind("SIn") if ("SI", sd) in res)
B = A and pers["SI"] >= -0.02 and resp["SI"] >= -0.02 and po >= 3
print(f"   4. organization against S: SI persistence {pers['SI']:+.3f}, responsiveness {resp['SI']:+.3f}; SI1 {pers['SI1']:+.3f}, "
      f"{resp['SI1']:+.3f}; own above neighbour's in persistence {po} of {len(ind('SIn'))} individual lives")
print(f"   -> it takes part in development: {B}")
none_change = all(abs(res["SI1", sd]["pic_apart"] - res["SIx", sd]["pic_apart"]) <= 0.1 * res["SIx", sd]["pic_apart"] and
                  abs(res["SI1", sd]["pic_persist"] - res["SIx", sd]["pic_persist"]) <= 0.1 * abs(res["SIx", sd]["pic_persist"])
                  for sd in ind("SI1"))
print(f"   nothing changes even at full strength (individual lives, pictures apart and picture persistence within 10%): {none_change}")
print("   6. coherence (pictures apart up by more than 10% against SIx -> reads apart up too):")
for a in ("SI", "SI1"):
    for sd in have(a, SEEDS):
        x, y = res[a, sd], res["SIx", sd]
        if x["pic_apart"] > 1.1 * y["pic_apart"]:
            ok = x["read_apart"] > y["read_apart"]
            print(f"      {a} seed {sd}: pictures apart {y['pic_apart']:.3f} -> {x['pic_apart']:.3f}; reads apart "
                  f"{y['read_apart']:.4f} -> {x['read_apart']:.4f} -> coherent {ok}")
