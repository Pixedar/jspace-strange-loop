"""Does the self-picture shape how its own future pictures are formed, and does that depend on the picture being a
representation of the system's own processes? (the research question, 17 October night). No training: a frozen pass of an
existing life (its final interface), four copies reading four new texts, the loop as in life (no probes, no body).

1. Reference run: at every token the stance, the picture, the reader's attention over its 16-token window (entropy
   and mean age of what it attends to) and each copy's next-token loss.
2. Interventions at checkpoints (every 120 tokens from token 300). The state is copied and each copy's picture is
   replaced by
     "foreign":  another copy's picture at the same moment (a natural picture of a different state);
     "own past": the copy's own picture from 40 tokens earlier (a state from its own history).
   "foreign" is applied through three routes for the next token: "full" (the stance it implies sets both the model's
   attention and the reader's query), "processing" (only the model's attention; the reader keeps the reference
   stance), "reading" (only the reader's query; the model keeps the reference stance). Then the loop runs on
   unchanged for 40 tokens. Recorded: how much of the injected difference in the picture remains after 1, 2, 5, 10,
   20, 40 tokens (distance to the reference picture divided by the injected distance).
3. Route cuts over the whole run: the reader's query, or the model's attention, set from a constant stance (each
   copy's mean reference stance) instead of the live one. Counted in each run: steps above 1 and above 0.5 (jumps),
   steps below 0.01 (rests), the spread of resting times (std of their log), as in v11.
Writes self_formation.json and self_formation_ref.npz into the run directory.
Usage: python self_formation.py RUN_DIR IFACE_PT [T_EVAL]"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_loop import Interface, make_streams  # noqa: E402
from loop12 import Runner12, channel, set_picture_geometry, set_picture_map  # noqa: E402

RUN, IFACE = sys.argv[1], sys.argv[2]
TE = int(sys.argv[3]) if len(sys.argv) > 3 else 1100
cfg = json.load(open(os.path.join(RUN, "run.json")))
B, m = 4, cfg["m"]
LAGS, HOR, BACK = (1, 2, 5, 10, 20, 40), 40, 40
CPS = list(range(300, TE - HOR - 1, 120))
unit = lambda z: z / z.norm(dim=-1, keepdim=True)  # noqa: E731
R = Runner12(cfg["model"], band=tuple(cfg["band"]), read_layer=cfg["read_layer"], window=cfg["window"], revision=cfg.get("model_revision"))
f = Interface(R, m=m, p=cfg["p"], R=cfg["R"], bound=cfg["bound"], steer_reader=cfg.get("steer_reader", True)).to(R.dev)
set_picture_map(f, cfg.get("picture_map"))
set_picture_geometry(f, cfg.get("picture", "cube"), cfg.get("reader_route", "operation"))   # builds the content item first
sd_ = torch.load(IFACE, map_location=R.dev)
sd_.pop("lam_raw", None)
f.load_state_dict(sd_)
f.eval()
offs_fn, spot_live, _ = channel(f, cfg)                   # (v24) the stance acting on the model sets the spotlight
CONTENT = cfg.get("reader_route") == "content"
corpus = json.load(open(os.environ.get("JSPACE_CORPUS", "data/corpus.json")))
ids, gen, genres = make_streams(corpus, R.tok, B, TE + 5, 99)
ids = ids.to(R.dev)


def read_att(s, hist, item=None):
    """The reader's picture and its attention over the 16 read-layer states (B, heads, window)."""
    N, Rp, _ = hist.shape
    a = f.rln(hist)
    k = f.rk(a).view(N, Rp, f.rh, -1)
    stance_q = f.rq(s) if (f.rq is not None and getattr(f, "route", "operation") == "operation") else 0.0
    q = (f.rq_bias.expand(N, -1) + stance_q).view(N, f.rh, -1)
    att = torch.softmax(torch.einsum("nhd,nlhd->nhl", q, k) / math.sqrt(q.shape[-1]) + f.rpos[-Rp:][None, None], -1)
    P = f.read(s, hist, item=item) if CONTENT else f.read(s, hist)
    return P, att


def step(st, s_band, s_read, t, rec=False):
    h_in = R.lower(st, ids[:, t])
    h_read, logits = R.upper_live(st, h_in, offs_fn, s_band, sb=spot_live(s_band, st["hist"])[0])
    st["hist"] = torch.cat([st["hist"], h_read.float()], 1)[:, -f.R:]
    P, att = read_att(s_read, st["hist"], st.get("P_prev"))
    st["P_prev"] = P                                                    # the "content" route reads it at the next token
    out = (P, f.next_stance(P))
    if rec:
        loss = F.cross_entropy(logits[:, 0].float(), ids[:, t + 1], reduction="none")
        return out + (att, loss)
    return out


def clone(st):
    return dict(K=[k.clone() for k in st["K"]], V=[v.clone() for v in st["V"]], pos=st["pos"], hist=st["hist"].clone(),
                P_prev=None if st.get("P_prev") is None else st["P_prev"].clone())


def jump_stats(S):
    """S: (T, B, m) stance path; second half."""
    S = S[len(S) // 2:]
    stp = np.linalg.norm(S[1:] - S[:-1], axis=-1)                    # (n, B)
    rests = []
    for b in range(S.shape[1]):
        run = 0
        for x in stp[:, b]:
            if x < 0.01:
                run += 1
            elif run:
                rests.append(run)
                run = 0
        if run:
            rests.append(run)
    lr = np.log(np.array(rests, float)) if rests else np.zeros(1)
    return dict(jumps_over_1=float((stp > 1.0).mean()), jumps_over_half=float((stp > 0.5).mean()),
                rest_share=float((stp < 0.01).mean()), rest_log_spread=float(lr.std()), n_rests=len(rests),
                long_rests_over_100=float(np.mean(np.array(rests) > 100)) if rests else 0.0,
                mean_step=float(stp.mean()))


t0 = time.time()
res = {"life": os.path.basename(RUN.rstrip("/\\")), "arm": cfg.get("arm"), "picture": cfg.get("picture", "cube"),
       "checkpoints": CPS}
with torch.no_grad():
    # ---------------------------------------------------------------- 1. reference run (state saved at checkpoints)
    st = R.new_state(B)
    s = unit(torch.randn(B, m, generator=torch.Generator().manual_seed(0))).to(R.dev)
    ref = {k: [] for k in ("s", "P", "att_entropy", "att_age", "loss")}
    saved = {}
    for t in range(TE):
        if t in CPS:
            saved[t] = (clone(st), s.clone())
        P, s_new, att, loss = step(st, s, s, t, rec=True)
        ref["s"].append(s.cpu().numpy())
        ref["P"].append(P.cpu().numpy())
        a = att.mean(1)                                                  # (B, window), heads averaged
        ref["att_entropy"].append((-(a.clamp_min(1e-9).log() * a).sum(-1)).cpu().numpy())
        ages = torch.arange(a.shape[-1] - 1, -1, -1, device=R.dev, dtype=torch.float32)   # 0 = the newest position
        ref["att_age"].append((a * ages).sum(-1).cpu().numpy())
        ref["loss"].append(loss.cpu().numpy())
        s = s_new
    ref = {k: np.array(v) for k, v in ref.items()}                       # s[t] = stance used at token t; P[t] after it
    S_ref, P_ref = ref["s"], ref["P"]
    spread = float(np.linalg.norm(P_ref[TE // 2:] - P_ref[TE // 2:].mean((0, 1)), axis=-1).mean())
    print(f"reference run {time.time() - t0:.0f}s", flush=True)
    # ---------------------------------------------------------------- 2. interventions
    kinds = {("foreign", "full"): [], ("foreign", "processing"): [], ("foreign", "reading"): [], ("own past", "full"): []}
    kinds_s = {k: [] for k in kinds}
    donor_follow = []
    for t in CPS:
        st0, s0 = saved[t]
        for (kind, route) in kinds:
            stB, sB = clone(st0), s0.clone()
            P_t, _ = step(stB, sB, sB, t)                                # token t as in the reference
            if kind == "foreign":
                X = torch.roll(P_t, -1, 0)                               # copy b gets copy b+1's picture
            else:
                X = torch.tensor(P_ref[t - BACK], device=R.dev, dtype=P_t.dtype)
            d0 = (X - torch.tensor(P_ref[t], device=R.dev)).norm(dim=-1)  # injected difference (B,)
            s_inj = f.next_stance(X)
            stB["P_prev"] = X                                            # for the content route: the item read next
            s_refn = torch.tensor(S_ref[t + 1], device=R.dev)
            s_band, s_read = {"full": (s_inj, s_inj), "processing": (s_inj, s_refn), "reading": (s_refn, s_inj)}[route]
            P1, sN = step(stB, s_band, s_read, t + 1)
            rows, rows_s = {1: P1}, {1: sN}
            for k in range(2, HOR + 1):
                Pk, sN_next = step(stB, sN, sN, t + k)
                rows[k], rows_s[k] = Pk, sN_next
                sN = sN_next
            frac = [((rows[L] - torch.tensor(P_ref[t + L], device=R.dev)).norm(dim=-1) / d0.clamp_min(1e-6)).cpu().numpy() for L in LAGS]
            kinds[(kind, route)].append(np.array(frac))                  # (lags, B)
            kinds_s[(kind, route)].append(np.array([(rows_s[L] - torch.tensor(S_ref[t + L + 1], device=R.dev)).norm(dim=-1).cpu().numpy() for L in LAGS]))
            if (kind, route) == ("foreign", "full"):
                donor = np.roll(P_ref, -1, axis=1)                        # each copy's donor's own reference pictures
                donor_follow.append(np.array([np.linalg.norm(rows[L].cpu().numpy() - donor[t + L], axis=-1) /
                                              np.maximum(np.linalg.norm(rows[L].cpu().numpy() - P_ref[t + L], axis=-1), 1e-6)
                                              for L in LAGS]))
    raw = {f"remaining_{k[0].replace(' ', '_')}_{k[1]}": np.array(v) for k, v in kinds.items()}       # (checkpoints, lags, B)
    raw.update({f"stance_{k[0].replace(' ', '_')}_{k[1]}": np.array(v) for k, v in kinds_s.items()})
    raw["donor_ratio"] = np.array(donor_follow)
    res["picture_difference_remaining"] = {f"{k[0]} / {k[1]}": {str(L): float(np.median(np.concatenate([v[i] for v in vals])))
                                                                 for i, L in enumerate(LAGS)} for k, vals in kinds.items()}
    res["stance_distance_after"] = {f"{k[0]} / {k[1]}": {str(L): float(np.median(np.concatenate([v[i] for v in vals])))
                                                          for i, L in enumerate(LAGS)} for k, vals in kinds_s.items()}
    res["foreign_full_closer_to_donor_than_to_own"] = {str(L): float(np.mean(np.concatenate([v[i] for v in donor_follow]) < 1.0))
                                                       for i, L in enumerate(LAGS)}
    res["picture_spread"] = spread
    print(f"interventions {time.time() - t0:.0f}s", flush=True)
    # ---------------------------------------------------------------- 3. route cuts over the whole run
    s_const = unit(torch.tensor(S_ref[TE // 4:].mean(0), device=R.dev))
    res["jumps"] = {"reference": jump_stats(S_ref)}
    for cut in ("reading cut", "processing cut"):
        st = R.new_state(B)
        s = unit(torch.randn(B, m, generator=torch.Generator().manual_seed(0))).to(R.dev)
        path = []
        for t in range(TE):
            sb, sr = (s, s_const) if cut == "reading cut" else (s_const, s)
            P, s_new = step(st, sb, sr, t)
            path.append(s.cpu().numpy())
            s = s_new
        res["jumps"][cut] = jump_stats(np.array(path))
    print(f"route cuts {time.time() - t0:.0f}s", flush=True)
np.savez_compressed(os.path.join(RUN, "self_formation_ref.npz"), genre=gen[:, :TE].T, **ref, **raw)
print(json.dumps(res, indent=1), flush=True)
json.dump(res, open(os.path.join(RUN, "self_formation.json"), "w"), indent=1)
