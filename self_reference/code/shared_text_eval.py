"""Do copies keep their own ways of moving when they read the same text? (developmental individuality; your answer to
my second question, 18 October: different texts for training, then a shared-text evaluation). No training.
A life's interface is run frozen: until the switch (token 850 by default) each copy reads its own text (the same four texts as
dynamics_anatomy.py); from the switch all four copies read one and the same new text. The model's context window is 256 tokens, so from 256 tokens after the
switch the copies' inputs and contexts are identical, and any difference that remains is carried only by the loop's
own state (stance, picture). Recorded after the switch: the copies' mutual distance in stance and picture at 0, 50,
100, 256, 300, 400, 500, 600 tokens; whether two copies started in exactly the same state stay identical (a symmetry
check, copy 3 is made a duplicate of copy 0 at the switch); the field's state-dependent part on the shared text.
Writes shared_text_<iface>.json into the run directory. Usage: python shared_text_eval.py RUN_DIR IFACE_PT [SWITCH_TOKEN, default 850]"""
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
cfg = json.load(open(os.path.join(RUN, "run.json")))
B, m, NP, EV, T1, T2 = 4, cfg["m"], 12, 10, int(sys.argv[3]) if len(sys.argv) > 3 else 850, 600
LAGS = (0, 50, 100, 256, 300, 400, 500, 599)
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
corpus = json.load(open(os.environ.get("JSPACE_CORPUS", "data/corpus.json")))
ids, gen, genres = make_streams(corpus, R.tok, B, 905, 99)             # the same four texts as dynamics_anatomy.py
ids = ids.to(R.dev)
ids2, _, _ = make_streams(corpus, R.tok, 1, T2 + 5, 98)                # the shared text: a new one, read by every copy
shared = ids2[0, :T2 + 1].to(R.dev)
S = unit(torch.randn(NP, m, generator=torch.Generator().manual_seed(4242))).to(R.dev)
t0 = time.time()
with torch.no_grad():
    st = R.new_state(B)
    s = unit(torch.randn(B, m, generator=torch.Generator().manual_seed(0))).to(R.dev)
    for t in range(T1):
        h_in = R.lower(st, ids[:, t])
        h_read, _ = R.upper_live(st, h_in, offs_fn, s, sb=spot_live(s, st["hist"])[0])
        st["hist"] = torch.cat([st["hist"], h_read.float()], 1)[:, -f.R:]
        s = f.next_stance(f.read(s, st["hist"]))
    # symmetry check: copy 3 becomes an exact duplicate of copy 0 at the switch
    for l in range(len(st["K"])):
        st["K"][l][3] = st["K"][l][0]
        st["V"][l][3] = st["V"][l][0]
    st["hist"][3] = st["hist"][0]
    s[3] = s[0]
    rows, deps = {}, []
    for k in range(T2):
        tok = shared[k].expand(B)
        h_in = R.lower(st, tok)
        if k % EV == 0 and k >= 256:
            b_idx = torch.arange(B, device=R.dev).repeat_interleave(NP)
            SS = S.repeat(B, 1)
            hr = R.upper_probe(st, h_in, offs_fn, SS, b_idx, sb=spot_probe(SS, st["hist"][b_idx], s, st["hist"], b_idx)).float()
            hp = torch.cat([st["hist"][b_idx], hr], 1)[:, -f.R:]
            dest = f.next_stance(f.read(SS, hp)).view(B, NP, m)
            res = dest - dest.mean(1, keepdim=True)
            Dp = dest - S[None]
            deps.append(float((res ** 2).sum() / ((Dp - Dp.mean((0, 1), keepdim=True)) ** 2).sum().clamp_min(1e-12)))
        h_read, _ = R.upper_live(st, h_in, offs_fn, s, sb=spot_live(s, st["hist"])[0])
        st["hist"] = torch.cat([st["hist"], h_read.float()], 1)[:, -f.R:]
        P = f.read(s, st["hist"])
        if k in LAGS:
            pairs = [(a, b) for a in range(3) for b in range(a + 1, 3)]          # the three distinct copies
            rows[str(k)] = {"stance_apart": float(np.mean([(s[a] - s[b]).norm().item() for a, b in pairs])),
                            "picture_apart": float(np.mean([((P[a] - P[b]).norm() / P[a].norm()).item() for a, b in pairs])),
                            "duplicate_vs_original": float((s[3] - s[0]).norm().item())}
        s = f.next_stance(P)
out = {"life": os.path.basename(RUN.rstrip("/\\")), "interface": os.path.basename(IFACE), "after_switch": rows,
       "state_dependent_part_on_shared_text": float(np.mean(deps)) if deps else None}
print(json.dumps(out), flush=True)
json.dump(out, open(os.path.join(RUN, f"shared_text_{os.path.basename(IFACE).split('.')[0]}.json"), "w"), indent=1)
print(f"done {time.time() - t0:.0f}s", flush=True)
