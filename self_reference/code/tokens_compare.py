"""Do lives that move in their own ways also change the model's predictions in their own ways? (PLAN section 18,
prediction 6.) From spotlight_check.py's per-token records (the same four texts and initial stances for every life):
for each life and checkpoint, the change in next-token loss that its own spotlight causes, per token and copy.
Correlation of these change sequences between two lives (same tokens, same copies), and within a life between two
checkpoints (2,500 against 3,000). Usage: python -I tokens_compare.py OUT_JSON LIFE_DIR [LIFE_DIR ...]"""
import glob
import json
import os
import re
import sys

import numpy as np


def corr(x, y):
    x, y = x.ravel() - x.mean(), y.ravel() - y.mean()
    return float((x * y).sum() / max(np.sqrt((x ** 2).sum() * (y ** 2).sum()), 1e-12))


recs = {}
for d in sys.argv[2:]:
    n = os.path.basename(d.rstrip("/\\"))
    for f in glob.glob(os.path.join(d, "spotlight_check_*_tokens.npz")):
        it = int(re.findall(r"spotlight_check_(\d+)_", os.path.basename(f))[0])
        recs[(n, it)] = np.load(f)["dloss"]
keys = sorted(recs)
rows = []
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        x, y = recs[a], recs[b]
        k = min(len(x), len(y))
        rows.append(dict(a=f"{a[0]}@{a[1]}", b=f"{b[0]}@{b[1]}", same_life=a[0] == b[0], corr=corr(x[:k], y[:k])))
within = [r["corr"] for r in rows if r["same_life"]]
between = [r["corr"] for r in rows if not r["same_life"] and r["a"].endswith("@3000") and r["b"].endswith("@3000")]
for r in rows:
    if r["same_life"] or (r["a"].endswith("@3000") and r["b"].endswith("@3000")):
        print(f"{r['a']:>28s} vs {r['b']:<28s} {'(same life)' if r['same_life'] else '':12s} {r['corr']:+.3f}")
out = dict(pairs=rows, within_median=float(np.median(within)) if within else None,
           between_median=float(np.median(between)) if between else None)
print(f"within a life (2,500 vs 3,000): median {out['within_median']}; between lives at 3,000: median {out['between_median']}")
json.dump(out, open(sys.argv[1], "w"), indent=1)
