"""Where is the spotlight's effect lost on its way back into the loop? (v24: the model path carries <= 0.6% of the
state dependence although the spotlight changes the model's processing.) Frozen pass of a life's interface on the
four texts of dynamics_anatomy.py; every 10 tokens of the last two thirds, at 12 fixed probe stances and the copies'
own: the probe stance sets only the spotlight (the reader keeps the live query), or only the reader's query (the
spotlight stays live). For each, relative to the live step: the change of the current token's read-layer state, of
the picture, and of the next stance; and the share of the reader's attention on the newest window position (the one
state the spotlight can change at this token). Usage: python spotlight_chain.py OUT_JSON RUN_DIR [IFACE]"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_loop import Interface, make_streams  # noqa: E402
from loop12 import Runner12, set_picture_geometry, set_picture_map, spotlight  # noqa: E402

OUT, RUN = sys.argv[1], sys.argv[2]
IFACE = sys.argv[3] if len(sys.argv) > 3 else os.path.join(RUN, "iface_3000.pt")
cfg = json.load(open(os.path.join(RUN, "run.json")))
kappa = cfg.get("spotlight") or 1.5
B, m, NP, TE = 4, cfg["m"], 12, 600
unit = lambda z: z / z.norm(dim=-1, keepdim=True)  # noqa: E731
NO = lambda l, x: None  # noqa: E731
R = Runner12(cfg["model"], band=tuple(cfg["band"]), read_layer=cfg["read_layer"], window=cfg["window"], revision=cfg.get("model_revision"))
f = Interface(R, m=m, p=cfg["p"], R=cfg["R"], bound=cfg["bound"], steer_reader=cfg.get("steer_reader", True)).to(R.dev)
set_picture_map(f, cfg.get("picture_map"))
set_picture_geometry(f, cfg.get("picture", "cube"), cfg.get("reader_route", "operation"))
sd_ = torch.load(IFACE, map_location=R.dev)
sd_.pop("lam_raw", None)
f.load_state_dict(sd_)
f.eval()
corpus = json.load(open(os.environ.get("JSPACE_CORPUS", "data/corpus.json")))
ids, _, _ = make_streams(corpus, R.tok, B, TE + 5, 99)
ids = ids.to(R.dev)
S0 = unit(torch.randn(NP, m, generator=torch.Generator().manual_seed(4242))).to(R.dev)
rows = {"spotlight_only": {"read_layer": [], "picture": [], "stance": []}, "reading_only": {"picture": [], "stance": []}}
newest = []
with torch.no_grad():
    st = R.new_state(B)
    s = unit(torch.randn(B, m, generator=torch.Generator().manual_seed(0))).to(R.dev)
    for t in range(TE):
        h_in = R.lower(st, ids[:, t])
        if t >= TE // 3 and t % 10 == 0:
            S = torch.cat([S0, s], 0)
            b_idx = torch.arange(B, device=R.dev).repeat_interleave(S.shape[0])
            SS, hist_b, s_b = S.repeat(B, 1), st["hist"][b_idx], s[b_idx]
            sb_live = spotlight(f, s, st["hist"], kappa)[0][b_idx]
            sb_probe = spotlight(f, SS, hist_b, kappa)[0]
            h_live = R.upper_probe(st, h_in, NO, SS, b_idx, sb=sb_live).float()           # live spotlight
            h_spot = R.upper_probe(st, h_in, NO, SS, b_idx, sb=sb_probe).float()          # the probe's spotlight
            win_live = torch.cat([hist_b, h_live], 1)[:, -f.R:]
            win_spot = torch.cat([hist_b, h_spot], 1)[:, -f.R:]
            P_live, P_spot, P_read = f.read(s_b, win_live), f.read(s_b, win_spot), f.read(SS, win_live)
            n_live, n_spot, n_read = f.next_stance(P_live), f.next_stance(P_spot), f.next_stance(P_read)
            rel = lambda a, b: ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-9)).cpu().numpy()  # noqa: E731
            rows["spotlight_only"]["read_layer"].append(rel(h_spot[:, 0], h_live[:, 0]))
            rows["spotlight_only"]["picture"].append(rel(P_spot, P_live))
            rows["spotlight_only"]["stance"].append((n_spot - n_live).norm(dim=-1).cpu().numpy())
            rows["reading_only"]["picture"].append(rel(P_read, P_live))
            rows["reading_only"]["stance"].append((n_read - n_live).norm(dim=-1).cpu().numpy())
            a = f.rk(f.rln(win_live)).view(len(SS), win_live.shape[1], f.rh, -1)
            q = (f.rq_bias.expand(len(SS), -1) + f.rq(s_b)).view(len(SS), f.rh, -1)
            att = torch.softmax(torch.einsum("nhd,nlhd->nhl", q, a) / np.sqrt(q.shape[-1]) + f.rpos[-win_live.shape[1]:][None, None], -1)
            newest.append(att[:, :, -1].mean(1).cpu().numpy())
        sb = spotlight(f, s, st["hist"], kappa)[0] if cfg.get("spotlight") else None
        hr, _ = R.upper_live(st, h_in, NO if cfg.get("spotlight") else f.offs, s, sb=sb)
        st["hist"] = torch.cat([st["hist"], hr.float()], 1)[:, -f.R:]
        s = f.next_stance(f.read(s, st["hist"]))
out = {"life": os.path.basename(RUN.rstrip("/\\")), "interface": os.path.basename(IFACE),
       "spotlight_only": {k: float(np.median(np.concatenate(v))) for k, v in rows["spotlight_only"].items()},
       "reading_only": {k: float(np.median(np.concatenate(v))) for k, v in rows["reading_only"].items()},
       "reader_attention_on_newest": float(np.mean(np.concatenate(newest)))}
print(json.dumps(out), flush=True)
json.dump(out, open(OUT, "w"), indent=1)
