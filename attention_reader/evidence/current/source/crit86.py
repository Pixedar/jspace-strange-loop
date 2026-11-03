"""WHERE_ITS_OWN_STATE_GOES.md section 86: the milestone with its picture reduced to one or two shared prototypes.
The organization as in section 46 (crit45.py's measures), the schema's fit, how well two prototypes describe the
unreduced pictures, how the copies use the prototypes, and the readings fixed in section 86.
Usage: python crit86.py ROOT LENS_PACK_NPZ STATES_S2_NPZ"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lens_pack import kind_scores  # noqa: E402

root, pack = sys.argv[1], dict(np.load(sys.argv[2]))
G = np.load(sys.argv[3])["genre"][0]
P95, SEG, SEEDS, FROM = 0.728, 100, range(10, 16), 300
ARMS = ("S", "S2", "S1", "Sn", "Sn2", "Sn1")
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
                kinds=int((np.abs(sc).max(-1) > P95).sum()), apart=float(np.nanmean(z["att_apart"][1500:])),
                r2=1.0 - float(np.nansum(z["sch_err"][FROM:]) / max(np.nansum(z["sch_tot"][FROM:]), 1e-12)),
                turn=float(np.nanmean(z["turn"][FROM:])))


def use_of(lab):
    """lab (T, B) prototype per token and copy: share in the first prototype per copy, changes per 1000 tokens, and
    how often two copies sit in different prototypes."""
    share = (lab == 0).mean(0)
    sw = float((lab[1:] != lab[:-1]).sum(0).mean() * 1000.0 / max(len(lab) - 1, 1))
    B = lab.shape[1]
    apart = float(np.mean([(lab[:, a] != lab[:, b]).mean() for a in range(B) for b in range(a + 1, B)]))
    return share, sw, apart


res, pics = {}, {}
for sd in SEEDS:
    for arm in ARMS:
        p_ = os.path.join(root, f"pa_c8_{arm}_seed{sd}", "life.npz")
        if os.path.exists(p_):
            z = dict(np.load(p_))
            res[arm, sd] = organization(z)
            if arm in ("S2", "Sn2"):
                res[arm, sd]["use"] = use_of(z["proto"][FROM:])
            if arm in ("S", "Sn"):
                pics[arm, sd] = z["pic"][FROM:].astype(np.float64)
done = sorted({a for a, _ in res})
print(f"lives found: {len(res)}; arms: {done}")

print("\n0. Checks")
for sd in SEEDS:
    line = []
    for arm in ("S", "Sn"):
        ref = os.path.join("D:/jspace_v11_runs/core32_schema45", f"ph_c8_{arm}_seed{sd}", "life.npz")
        if (arm, sd) in res and os.path.exists(ref):
            a_, b_ = np.load(os.path.join(root, f"pa_c8_{arm}_seed{sd}", "life.npz")), np.load(ref)
            line.append(f"{arm} rerun against section 46: max difference {max(float(np.abs(a_[k] - b_[k]).max()) for k in ('s', 'g', 'lens_seg')):.1e}")
    for arm in ("S2", "Sn2", "S1", "Sn1"):
        if (arm, sd) in res:
            a_ = np.load(os.path.join(root, f"pa_c8_{arm}_seed{sd}", "life.npz"))
            b_ = np.load(os.path.join(root, f"pa_c8_{arm[:-1]}_seed{sd}", "life.npz"))
            first = int(np.argmax(np.abs(a_["s"] - b_["s"]).reshape(3000, -1).max(-1) > 0))
            line.append(f"{arm} parts from {arm[:-1]} at token {first}")
    print(f"   seed {sd}: " + "; ".join(line))

print("\n1. How well two shared prototypes describe the unreduced pictures (tokens 300-2999)")
for arm in ("S", "Sn"):
    for sd in SEEDS:
        if (arm, sd) not in pics or not os.path.exists(os.path.join(root, f"pa_c8_{arm}2_seed{sd}", "protos.npy")):
            continue
        X = pics[arm, sd]
        U = X / np.linalg.norm(X, axis=-1, keepdims=True)
        C = np.load(os.path.join(root, f"pa_c8_{arm}2_seed{sd}", "protos.npy"))
        lab = (U @ C.T).argmax(-1)
        flat = U.reshape(-1, U.shape[-1])
        near = C[lab.reshape(-1)]
        expl = 1.0 - float(((flat - near) ** 2).sum() / ((flat - flat.mean(0)) ** 2).sum())
        share, sw, apart = use_of(lab)
        print(f"   {arm:2s} seed {sd}: the two explain {expl:.0%} of the spread; prototypes cos {float(C[0] @ C[1]):+.2f}; "
              f"share in the first per copy {np.round(share, 2)}; changes per 1000 tokens {sw:.0f}; two copies in "
              f"different prototypes {apart:.0%} of the time")

print("\n1b. What the two states are: the text's genre; and whether the copies' stances differ")
g_txt = G[FROM:3000]
gsw = (np.flatnonzero(G[257:3000] != G[256:2999]) + 257).tolist()
print(f"   the text's genre changes at tokens {gsw}")
for arm in ("S", "Sn"):
    for sd in SEEDS:
        if (arm, sd) not in pics or not os.path.exists(os.path.join(root, f"pa_c8_{arm}2_seed{sd}", "protos.npy")):
            continue
        U = pics[arm, sd] / np.linalg.norm(pics[arm, sd], axis=-1, keepdims=True)
        lab = (U @ np.load(os.path.join(root, f"pa_c8_{arm}2_seed{sd}", "protos.npy")).T).argmax(-1)[:, 0]
        agree = max((max(float(((g_txt == gv) == (lab == 0)).mean()), float(((g_txt == gv) == (lab == 1)).mean())), int(gv))
                    for gv in np.unique(g_txt))
        swp = (np.flatnonzero(lab[1:] != lab[:-1]) + FROM + 1).tolist()
        s_ = np.load(os.path.join(root, f"pa_c8_{arm}_seed{sd}", "life.npz"))["s"].astype(np.float64)
        u_ = s_ / np.linalg.norm(s_, axis=-1, keepdims=True)
        cs = lambda t: [float(u_[t, a] @ u_[t, b]) for a in range(4) for b in range(a + 1, 4)]  # noqa: E731
        print(f"   {arm:2s} seed {sd}: state agrees with genre {agree[1]} against the rest on {agree[0]:.0%} of tokens; "
              f"changes at {swp}; stances between copies (cos, the 6 pairs) at 300 {np.round(cs(FROM), 2)}, "
              f"at 2999 {np.round(cs(2999), 2)}")

print("\n2. The organization (persistence, responsiveness, kinds above chance, attention apart; schema R2; stance turn)")
for arm in ARMS:
    v = [res[arm, sd] for sd in SEEDS if (arm, sd) in res]
    if v:
        print(f"   {arm:3s}: persistence {np.mean([x['persist'] for x in v]):+.3f}, responsiveness "
              f"{np.mean([x['respond'] for x in v]):+.3f}, kinds {sum(x['kinds'] for x in v)}/{4 * len(v)}, apart "
              f"{np.mean([x['apart'] for x in v]):.3f}; schema R2 {np.mean([x['r2'] for x in v]):+.3f}; stance turn "
              f"{np.mean([x['turn'] for x in v]):.4f} ({len(v)} lives)")
for sd in SEEDS:
    print(f"   seed {sd}: " + "  ".join(f"{arm} {res[arm, sd]['persist']:+.3f} {res[arm, sd]['respond']:+.3f} "
                                       f"{res[arm, sd]['kinds']} {res[arm, sd]['apart']:.3f}" for arm in ARMS if (arm, sd) in res)
          + "   (persistence, responsiveness, kinds, apart)")
    print(f"            stance turn per token: " + "  ".join(f"{arm} {res[arm, sd]['turn']:.4f}" for arm in ARMS if (arm, sd) in res))
for arm in ("S2", "Sn2"):
    for sd in SEEDS:
        if (arm, sd) in res:
            share, sw, apart = res[arm, sd]["use"]
            print(f"   {arm} seed {sd}: share in the first prototype per copy {np.round(share, 2)}; changes per 1000 "
                  f"tokens {sw:.0f}; two copies in different prototypes {apart:.0%} of the time")

print("\n3. Readings (fixed in section 86)")
both = lambda a, b: [sd for sd in SEEDS if (a, sd) in res and (b, sd) in res]  # noqa: E731
mean_d = lambda a, b, k: float(np.mean([res[a, sd][k] - res[b, sd][k] for sd in both(a, b)]))  # noqa: E731
for red in ("S2", "S1"):
    if both(red, "S"):
        dp, dr = mean_d(red, "S", "persist"), mean_d(red, "S", "respond")
        keeps = abs(dp) <= 0.02 and abs(dr) <= 0.02
        print(f"   {red} against S: persistence {dp:+.3f}, responsiveness {dr:+.3f}, kinds "
              f"{sum(res[red, sd]['kinds'] for sd in both(red, 'S'))} against {sum(res['S', sd]['kinds'] for sd in both(red, 'S'))}"
              f" -> keeps S's organization: {keeps}")
for own, nb in (("S", "Sn"), ("S2", "Sn2"), ("S1", "Sn1")):
    sds = both(own, nb)
    if sds:
        up = sum(res[own, sd]["persist"] > res[nb, sd]["persist"] for sd in sds)
        ap = sum(res[own, sd]["apart"] > res[nb, sd]["apart"] for sd in sds)
        print(f"   own against neighbour's schema, {own} against {nb}: persistence {mean_d(own, nb, 'persist'):+.3f} "
              f"(above in {up} of {len(sds)}), responsiveness {mean_d(own, nb, 'respond'):+.3f}, attention apart "
              f"{mean_d(own, nb, 'apart'):+.3f} (higher in {ap} of {len(sds)}) -> holds: {up >= 5 and mean_d(own, nb, 'apart') > 0}")
