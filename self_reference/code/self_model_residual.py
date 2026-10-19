"""Does the self-picture represent how where the system goes depends on where it is? (step B, prediction 6; the tested criterion
"a system develops representations of its own operations, including aspects of how those representations are produced
and used")
Frozen pass of a life's final interface on four new texts. Every 5 tokens (last two thirds), at 8 random probe stances
and the 4 current ones: the actual field D(S) (where the stance would go from S; probe passes as in life) and the field
the copy's picture predicts, predict_field(P, S). The destinations D(S) + S are split into their mean over probes (the
"one destination") and the remainder, the part of where the system goes that depends on where it is. Reported: the
remainder's share of the field's variance; how well the picture's predicted remainder matches the actual one
(correlation, and the share of the actual remainder's variance explained); the same for the whole field.
Writes self_model_residual.json into the run directory. Usage: python self_model_residual.py RUN_DIR IFACE_PT [T_EVAL] [OUT_NAME]"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_loop import Interface, make_streams  # noqa: E402
from loop12 import Runner12, channel, set_picture_geometry, set_picture_map  # noqa: E402

RUN, IFACE = sys.argv[1], sys.argv[2]
TE = int(sys.argv[3]) if len(sys.argv) > 3 else 900
OUT_NAME = sys.argv[4] if len(sys.argv) > 4 else "self_model_residual.json"   # parallel runs on one life need their own
cfg = json.load(open(os.path.join(RUN, "run.json")))
B, m, K, EV = 4, cfg["m"], 8, 5
unit = lambda z: z / z.norm(dim=-1, keepdim=True)  # noqa: E731
R = Runner12(cfg["model"], band=tuple(cfg["band"]), read_layer=cfg["read_layer"], window=cfg["window"], revision=cfg.get("model_revision"))
f = Interface(R, m=m, p=cfg["p"], R=cfg["R"], bound=cfg["bound"], steer_reader=cfg.get("steer_reader", True)).to(R.dev)
set_picture_map(f, cfg.get("picture_map"))
set_picture_geometry(f, cfg.get("picture", "cube"), cfg.get("reader_route", "operation"))
sd_ = torch.load(IFACE, map_location=R.dev)
sd_.pop("lam_raw", None)
f.load_state_dict(sd_)
f.eval()
offs_fn, spot_live, spot_probe = channel(f, cfg)          # (v24) the shared spotlight, if the life has one
CONTENT = cfg.get("reader_route") == "content"
corpus = json.load(open(os.environ.get("JSPACE_CORPUS", "data/corpus.json")))
ids, gen, genres = make_streams(corpus, R.tok, B, TE + 5, 99)
ids = ids.to(R.dev)
pg = torch.Generator().manual_seed(999)


def rd(s_, h_, item=None):
    return f.read(s_, h_, item=item) if CONTENT else f.read(s_, h_)


acts, preds, probes = [], [], []
t0 = time.time()
with torch.no_grad():
    st = R.new_state(B)
    s = unit(torch.randn(B, m, generator=torch.Generator().manual_seed(0))).to(R.dev)
    P_prev = None
    for t in range(TE):
        h_in = R.lower(st, ids[:, t])
        probe = t % EV == 0 and t >= TE // 3
        if probe:
            S = torch.cat([unit(torch.randn(K, m, generator=pg)).to(R.dev), s], 0)       # (Kp, m) shared probes
            b_idx = torch.arange(B, device=R.dev).repeat_interleave(S.shape[0])
            SS = S.repeat(B, 1)
            hr = R.upper_probe(st, h_in, offs_fn, SS, b_idx, sb=spot_probe(SS, st["hist"][b_idx], s, st["hist"], b_idx)).float()
            hp = torch.cat([st["hist"][b_idx], hr], 1)[:, -f.R:]
            nxt = f.next_stance(rd(SS, hp, None if P_prev is None else P_prev[b_idx])).view(B, -1, m)
            D = nxt - S[None]
        h_read, _ = R.upper_live(st, h_in, offs_fn, s, sb=spot_live(s, st["hist"])[0])
        st["hist"] = torch.cat([st["hist"], h_read.float()], 1)[:, -f.R:]
        P = rd(s, st["hist"], P_prev)
        if probe:
            pred = torch.stack([f.predict_field(P[b:b + 1], S) for b in range(B)])        # (B, Kp, m)
            acts.append(D.cpu().numpy())
            preds.append(pred.cpu().numpy())
            probes.append(S.cpu().numpy())
        P_prev = P
        s = f.next_stance(P)
A, Pd, Sp = np.array(acts), np.array(preds), np.array(probes)    # (n, B, Kp, m), (n, B, Kp, m), (n, Kp, m)
dest_a, dest_p = A + Sp[:, None], Pd + Sp[:, None]
res_a = dest_a - dest_a.mean(2, keepdims=True)                     # where the destination depends on the probe
res_p = dest_p - dest_p.mean(2, keepdims=True)
field_var = ((A - A.mean((0, 1, 2), keepdims=True)) ** 2).sum()


def corr(x, y):
    x, y = x.ravel() - x.mean(), y.ravel() - y.mean()
    return float((x * y).sum() / max(np.sqrt((x ** 2).sum() * (y ** 2).sum()), 1e-12))


out = {"life": os.path.basename(RUN.rstrip("/\\")), "describe": cfg.get("describe", "itself"),
       "route": cfg.get("reader_route", "operation"), "picture": cfg.get("picture", "cube"),
       "remainder_share_of_field": float((res_a ** 2).sum() / max(field_var, 1e-12)),
       "predicted_remainder_correlation": corr(res_p, res_a),
       "predicted_remainder_explained": float(1 - ((res_a - res_p) ** 2).sum() / max((res_a ** 2).sum(), 1e-12)),
       "predicted_field_correlation": corr(Pd, A),
       "predicted_field_explained": float(1 - ((A - Pd) ** 2).sum() / max(field_var, 1e-12))}
print(json.dumps(out), flush=True)
out["interface"] = os.path.basename(IFACE)
json.dump(out, open(os.path.join(RUN, OUT_NAME), "w"), indent=1)
print(f"done {time.time() - t0:.0f}s", flush=True)
