"""WHERE_ITS_OWN_STATE_GOES.md section 94: the opposite lean (toward the content that would move its picture against what
it expects to make of the text), own model against a neighbour's, at a quarter, half and full strength, in the four
individual lives; against the milestone and the confirming lean of sections 91 and 93. Written before the run.
Readings R1-R3, the chase and variety as fixed in section 94.
Usage: python crit94.py ROOT LENS_PACK_NPZ STATES_S2_NPZ"""
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
LEVELS = ("25", "50", "100")
NAME = {"M": "pj_c8_SIx", "L25": "pk_c8_L25", "L50": "pk_c8_L50", "L100": "pk_c8_L"}
for lv in LEVELS:
    NAME["O" + lv], NAME["On" + lv] = f"po_c8_O{lv}", f"po_c8_On{lv}"
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


res, st = {}, {}
for sd in INDIV:
    for arm, nm in NAME.items():
        p_ = os.path.join(root, f"{nm}_seed{sd}", "life.npz")
        if os.path.exists(p_):
            z = dict(np.load(p_))
            m = organization(z)
            m["pic_apart"], m["genre"] = apart(z["pic"]), genre_agree(z["pic"])
            m["moved"] = float(np.nanmean(z["i_moved"][FROM:])) if "i_moved" in z and np.isfinite(z["i_moved"]).any() else np.nan
            res[arm, sd], st[arm, sd] = m, z["s"]
print(f"lives found: {len(res)}")
print("\n0. Checks (first token at which the stance differs from the milestone's)")
for sd in INDIV:
    print(f"   seed {sd}: " + "; ".join(
        f"{arm} {int(np.argmax(np.abs(st[arm, sd] - st['M', sd]).reshape(3000, -1).max(-1) > 0))}"
        for arm in NAME if arm.startswith("O") and (arm, sd) in res and ("M", sd) in res))

print("\nThe individual lives (means): persistence, responsiveness, attention apart, pictures apart, how far the lean moves "
      "attention; genre agreement, lowest life")
for arm in NAME:
    v = [res[arm, sd] for sd in INDIV if (arm, sd) in res]
    if v:
        mn = lambda k: float(np.nanmean([x[k] for x in v]))  # noqa: E731
        print(f"   {arm:5s}: {mn('persist'):+.3f} {mn('respond'):+.3f} {mn('apart'):.3f} {mn('pic_apart'):.3f} "
              f"{mn('moved'):.3f}; genre {min(x['genre'] for x in v):.0%} ({len(v)} lives)")
for sd in INDIV:
    print(f"   seed {sd}: " + "  ".join(f"{arm} {res[arm, sd]['persist']:+.3f}/{res[arm, sd]['respond']:+.3f}/"
                                       f"{res[arm, sd]['apart']:.3f}" for arm in NAME if (arm, sd) in res)
          + "   (persistence/responsiveness/attention apart)")


def keeps_text(arm):
    sds = [sd for sd in INDIV if (arm, sd) in res and ("M", sd) in res]
    dr = float(np.mean([res[arm, sd]["respond"] - res["M", sd]["respond"] for sd in sds]))
    return dr >= -0.02 and all(res[arm, sd]["genre"] >= 0.85 for sd in sds), dr


print("\nReadings (fixed in section 94)")
r1_count, r3 = 0, False
for lv in LEVELS:
    O, On, L = "O" + lv, "On" + lv, "L" + lv
    sds = [sd for sd in INDIV if all((a, sd) in res for a in ("M", O, On, L))]
    if not sds:
        continue
    ko, dro = keeps_text(O)
    kl, drl = keeps_text(L)
    r1 = ko and not kl
    r1_count += r1
    d = [res[O, sd]["persist"] - res[On, sd]["persist"] for sd in sds]
    pos, neg = sum(x > 0.02 for x in d), sum(x < -0.02 for x in d)
    r2 = max(pos, neg) >= 3
    dp = float(np.mean([res[O, sd]["persist"] - res["M", sd]["persist"] for sd in sds]))
    dpa = float(np.mean([res[O, sd]["pic_apart"] - res["M", sd]["pic_apart"] for sd in sds]))
    da = float(np.mean([res[O, sd]["apart"] for sd in sds])) / max(float(np.mean([res["M", sd]["apart"] for sd in sds])), 1e-9)
    r3 = r3 or (r1 and r2 and dp >= -0.02)
    print(f"   strength {int(lv) / 100:.2f}: opposite lean keeps the text {ko} (responsiveness {dro:+.3f}); confirming lean "
          f"kept it {kl} ({drl:+.3f}) -> R1 {r1}; own against neighbour's persistence differs > 0.02: own above in {pos}, "
          f"below in {neg} of {len(sds)} -> R2 {r2}; persistence against the milestone {dp:+.3f}; chase "
          f"{dpa > 0 and dp < -0.02}; variety {da > 1.5 and dp < -0.02}")
print(f"   R1 overall (at 2 or more strengths): {r1_count >= 2} ({r1_count} of 3); R3: {r3}")
