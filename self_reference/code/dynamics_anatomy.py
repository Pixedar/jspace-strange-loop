"""What is the loop's dependence on its own state made of? (v22; your cautions: an arbitrary movement signature, a
flip). Frozen pass of a life's interface on four new texts; every 5 tokens (last two thirds) the field at 12 FIXED
probe stances (the same at every time, so fields can be compared over time) and the stance path.
Reported:
- the stance path: share of big steps (> 0.5), cosine of consecutive stances and of stances two tokens apart (a
  two-state flip has −1-like and +1 values), rests;
- the state-dependent part of the field (destinations minus their mean over the fixed probes) and how it is made:
  the share of its variance that is each copy's own constant pattern over time (a fixed signature), the share that
  changes with time, and how much of the time-changing part follows the genre being read (between-genre variance
  of each copy's pattern, against the same with genre labels shifted);
- the picture's description of the time-changing part (correlation of predicted and actual), separately from the
  signature;
- whether the dependence differs between copies at all: the share of its variance that is copy-specific (deviation
  from the four copies' mean at each time) and the copies' mutual stance distance in this frozen pass.
Writes dynamics_anatomy_<iface>.json into the run directory. Usage: python dynamics_anatomy.py RUN_DIR IFACE_PT [T_EVAL]"""
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
cfg = json.load(open(os.path.join(RUN, "run.json")))
B, m, NP, EV = 4, cfg["m"], 12, 5
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
ids, gen, genres = make_streams(corpus, R.tok, B, TE + 5, 99)
ids = ids.to(R.dev)
S = unit(torch.randn(NP, m, generator=torch.Generator().manual_seed(4242))).to(R.dev)      # fixed probes
t0 = time.time()
res_a, res_p, gtimes, path = [], [], [], []
with torch.no_grad():
    st = R.new_state(B)
    s = unit(torch.randn(B, m, generator=torch.Generator().manual_seed(0))).to(R.dev)
    for t in range(TE):
        h_in = R.lower(st, ids[:, t])
        probe = t % EV == 0 and t >= TE // 3
        if probe:
            b_idx = torch.arange(B, device=R.dev).repeat_interleave(NP)
            SS = S.repeat(B, 1)
            hr = R.upper_probe(st, h_in, offs_fn, SS, b_idx, sb=spot_probe(SS, st["hist"][b_idx], s, st["hist"], b_idx)).float()
            hp = torch.cat([st["hist"][b_idx], hr], 1)[:, -f.R:]
            dest = f.next_stance(f.read(SS, hp)).view(B, NP, m)
        h_read, _ = R.upper_live(st, h_in, offs_fn, s, sb=spot_live(s, st["hist"])[0])
        st["hist"] = torch.cat([st["hist"], h_read.float()], 1)[:, -f.R:]
        P = f.read(s, st["hist"])
        if probe:
            pdest = torch.stack([f.predict_field(P[b:b + 1], S) for b in range(B)]) + S[None]
            res_a.append((dest - dest.mean(1, keepdim=True)).cpu().numpy())
            res_p.append((pdest - pdest.mean(1, keepdim=True)).cpu().numpy())
            gtimes.append(gen[:, t])
        path.append(s.cpu().numpy())
        s = f.next_stance(P)
A, Pr, G = np.array(res_a), np.array(res_p), np.array(gtimes)       # (n, B, NP, m), (n, B, NP, m), (n, B)
path = np.array(path)[TE // 3:]
stp = np.linalg.norm(path[1:] - path[:-1], axis=-1)
c1 = (path[1:] * path[:-1]).sum(-1)
c2 = (path[2:] * path[:-2]).sum(-1)
tot = float((A ** 2).sum())
sig = A.mean(0, keepdims=True)                                     # each copy's constant pattern over time
var_sig = float(((sig - 0) ** 2).sum() * A.shape[0])
var_time = float(((A - sig) ** 2).sum())


def corr(x, y):
    x, y = x.ravel() - x.mean(), y.ravel() - y.mean()
    return float((x * y).sum() / max(np.sqrt((x ** 2).sum() * (y ** 2).sum()), 1e-12))


def genre_share(X, labels):
    """Share of the time-varying variance explained by genre means, per copy."""
    out = []
    for b in range(B):
        xb, lb = X[:, b].reshape(len(X), -1), labels[:, b]
        mu = xb.mean(0)
        between = sum(((xb[lb == g].mean(0) - mu) ** 2).sum() * (lb == g).sum() for g in np.unique(lb))
        out.append(between / max(((xb - mu) ** 2).sum(), 1e-12))
    return float(np.mean(out))


Atv, Ptv = A - sig, Pr - Pr.mean(0, keepdims=True)
shift = np.roll(G, len(G) // 2, axis=0)
out = {"life": os.path.basename(RUN.rstrip("/\\")), "interface": os.path.basename(IFACE),
       "big_steps": float((stp > 0.5).mean()), "rests": float((stp < 0.01).mean()),
       "cos_next_median": float(np.median(c1)), "cos_two_apart_median": float(np.median(c2)),
       "state_dependent_size": float(np.sqrt(tot / A.size)),
       "signature_share": var_sig / max(tot, 1e-12), "time_varying_share": var_time / max(tot, 1e-12),
       "time_varying_genre_share": genre_share(Atv, G), "same_genre_shifted": genre_share(Atv, shift),
       "copies_apart_frozen": float(np.mean([np.linalg.norm(path[:, a] - path[:, b], axis=-1).mean() for a in range(B) for b in range(a + 1, B)])),
       "copy_specific_share": float(((A - A.mean(1, keepdims=True)) ** 2).sum() / max(tot, 1e-12)),
       "description_of_time_varying_part": corr(Ptv, Atv), "description_of_signature": corr(Pr.mean(0), sig[0]),
       "description_overall": corr(Pr, A)}
print(json.dumps(out), flush=True)
json.dump(out, open(os.path.join(RUN, f"dynamics_anatomy_{os.path.basename(IFACE).split('.')[0]}.json"), "w"), indent=1)
print(f"done {time.time() - t0:.0f}s", flush=True)
