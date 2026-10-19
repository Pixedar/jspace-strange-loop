"""Laptop check of the shared spotlight (../METHODS.md section 8), before any GPU run. No training.
For each given interface (developmental starts and rich v22 lives):
1. Calibration (the rule used for every stance strength): random unit stances (new every 25 tokens) set the reader's
   attention, whose spotlight biases the model's middle-layer attention; old query shift off; against the untouched
   model on the same texts: read-layer change (relative), next-token loss change, and the share of the model's
   middle-layer attention the spotlight moves (total variation), for kappa 0.5, 1, 2, 4, 8.
2. Consequences at the chosen kappa, with the loop running on its own (frozen) on four new texts, 600 tokens, four runs
   stepped together: its own spotlight, no channel at all, the neighbour copy's spotlight, the old query shift only
   (v22). Against no channel, from token 30: divergence of the full next-token distribution (KL), share of tokens whose
   top prediction changes, next-token loss, read-layer change, and the loop's own rests and big steps. Within the
   own-spotlight run, the immediate effect (the same state, the current token with and without its spotlight) and the
   share of the model's attention it moves.
3. Which path the field's state dependence takes now (own-spotlight run, every 10 tokens of the last two thirds; 12
   fixed probe stances and the copies' own): probe stance on both paths, on the reader's query only, on the spotlight
   only.
4. Where usefulness and self-description meet: at 6 checkpoints, the gradients of the next-token loss (through the
   spotlight) and of the self-description game (identity by how it moves) on the reader's parameters, each loss
   divided by its value as the loop's loss weighting does; their sizes and cosine.
Writes OUT_JSON and, per interface, OUT_<name>_tokens.npz (per token and copy, from token 30: the own spotlight's
loss difference and KL against no channel, for comparing lives on the same texts).
Usage: python spotlight_check.py OUT_JSON KAPPA|auto|KAPPA:nocalib NAME=RUN_DIR::IFACE [...]
(names starting with "start" decide kappa by the rule when KAPPA is "auto"; ":nocalib" skips the calibration)."""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_loop import Interface, contrastive, make_streams  # noqa: E402
from loop12 import Runner12, set_picture_geometry, set_picture_map, spotlight  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv  # noqa: E402

OUT, KARG = sys.argv[1], sys.argv[2]
ITEMS = [(a.split("=", 1)[0], *a.split("=", 1)[1].split("::")) for a in sys.argv[3:]]
KAPPAS = (0.5, 1.0, 2.0, 4.0, 8.0)
B, NP, T0 = 4, 12, 30
TE = int(os.environ.get("SPOT_TE", 600))          # tokens per consequence run (600; smaller only for smoke tests)
GRAD_T = (220, 290, 360, 430, 500, 570)
unit = lambda z: z / z.norm(dim=-1, keepdim=True)  # noqa: E731
NO_SHIFT = lambda l, x: None  # noqa: E731
corpus = json.load(open(os.environ.get("JSPACE_CORPUS", "data/corpus.json")))
R, results = None, {}


def load(run, iface):
    global R
    cfg = json.load(open(os.path.join(run, "run.json")))
    if R is None:
        R = Runner12(cfg["model"], band=tuple(cfg["band"]), read_layer=cfg["read_layer"], window=cfg["window"], revision=cfg.get("model_revision"))
    f = Interface(R, m=cfg["m"], p=cfg["p"], R=cfg["R"], bound=cfg["bound"], steer_reader=cfg.get("steer_reader", True)).to(R.dev)
    set_picture_map(f, cfg.get("picture_map"))
    set_picture_geometry(f, cfg.get("picture", "cube"), cfg.get("reader_route", "operation"))
    sd_ = torch.load(iface, map_location=R.dev)
    sd_.pop("lam_raw", None)
    f.load_state_dict(sd_)
    return cfg, f


def attention_moved(l, h, Kp, pos, sb):
    """Total variation between band layer l's attention with and without the spotlight bias (mean over heads)."""
    lay = R.layers[l]
    a = lay.self_attn
    N = h.shape[0]
    x = lay.input_layernorm(h)
    q = a.q_norm(a.q_proj(x).view(N, 1, R.nh, R.hd)).transpose(1, 2)
    k = a.k_norm(a.k_proj(x).view(N, 1, R.kvh, R.hd)).transpose(1, 2)
    cos, sin = R.model.model.rotary_emb(x, torch.full((N, 1), pos, device=R.dev, dtype=torch.long))
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    K = repeat_kv(torch.cat([Kp, k], 2), R.nh // R.kvh)
    sc = (q.float() @ K.float().transpose(-1, -2)) * a.scaling
    Lk, Rp = K.shape[2], sb.shape[1]
    bias = torch.cat([torch.zeros(N, Lk - 1 - Rp, device=R.dev), sb.float(), torch.zeros(N, 1, device=R.dev)], 1)
    w0, w1 = torch.softmax(sc, -1), torch.softmax(sc + bias[:, None, None], -1)
    return float(0.5 * (w1 - w0).abs().sum(-1).mean())


def upper_eval(st, h_in, offs_fn, s, sb, att=False):
    """The layers above the lower ones for the current token, without writing to the cache."""
    h, h_read, tv = h_in, None, []
    for l in range(R.band[0], R.L):
        band = l in R.band
        if att and band and sb is not None:
            tv.append(attention_moved(l, h, st["K"][l], st["pos"], sb))
        h, _, _ = R.layer(l, h, st["K"][l], st["V"][l], st["pos"], offs_fn(l, s) if band else None, None, sb if band else None)
        if l == R.read_layer:
            h_read = h
    return h_read, R.model.lm_head(R.model.model.norm(h)), (float(np.mean(tv)) if tv else None)


def calibrate(f, cfg):
    ids, _, _ = make_streams(corpus, R.tok, B, 300, 123)
    ids = ids.to(R.dev)
    rows = {}
    with torch.no_grad():
        for kappa in (0.0,) + KAPPAS:
            st = R.new_state(B)
            g = torch.Generator().manual_seed(0)
            s = unit(torch.randn(B, cfg["m"], generator=g)).to(R.dev)
            hs, ls, tvs = [], [], []
            for t in range(250):
                h_in = R.lower(st, ids[:, t])
                sb = spotlight(f, s, st["hist"], kappa)[0] if kappa > 0 else None
                if t > 30 and sb is not None:
                    tvs.append(upper_eval(st, h_in, NO_SHIFT, s, sb, att=True)[2])
                hr, lg = R.upper_live(st, h_in, NO_SHIFT, s, sb=sb)
                st["hist"] = torch.cat([st["hist"], hr.float()], 1)[:, -f.R:]
                if t > 30:
                    hs.append(hr.float())
                    ls.append(F.cross_entropy(lg[:, 0].float(), ids[:, t + 1]).item())
                if t % 25 == 0:
                    s = unit(torch.randn(B, cfg["m"], generator=g)).to(R.dev)
            rows[kappa] = (torch.cat(hs, 1), float(np.mean(ls)), float(np.mean(tvs)) if tvs else 0.0)
    h0, l0, _ = rows[0.0]
    return {str(k): dict(read_layer_change=float(((rows[k][0] - h0).norm(dim=-1) / h0.norm(dim=-1)).mean()),
                         loss_change_pct=100 * (rows[k][1] / l0 - 1), attention_moved=rows[k][2]) for k in KAPPAS}


def route_split_step(f, cfg, kappa, st, s, h_in, acc, S0):
    S = torch.cat([S0, s], 0)
    b_idx = torch.arange(B, device=R.dev).repeat_interleave(S.shape[0])
    SS, hist_b, s_b = S.repeat(B, 1), st["hist"][b_idx], s[b_idx]
    sb_probe = spotlight(f, SS, hist_b, kappa)[0]
    sb_live = spotlight(f, s, st["hist"], kappa)[0][b_idx]
    for route, (sb_, q_) in {"both": (sb_probe, SS), "reading": (sb_live, SS), "spotlight": (sb_probe, s_b)}.items():
        hr = R.upper_probe(st, h_in, NO_SHIFT, SS, b_idx, sb=sb_).float()
        hp = torch.cat([hist_b, hr], 1)[:, -f.R:]
        dest = f.next_stance(f.read(q_, hp)).view(B, -1, cfg["m"])
        Dp = dest - S[None]
        acc[route][0] += float(((dest - dest.mean(1, keepdim=True)) ** 2).sum())
        acc[route][1] += float(((Dp - Dp.mean((0, 1), keepdim=True)) ** 2).sum())


def gradient_step(f, cfg, kappa, st, s, h_in, target, pg):
    names = [n for n, _ in f.named_parameters()]
    params = list(f.parameters())
    reader = [i for i, n in enumerate(names) if n.split(".")[0] in ("rq", "rk", "rln", "rpos", "rq_bias")]
    query = [i for i, n in enumerate(names) if n.startswith("rq.")]
    with torch.enable_grad():
        hist = st["hist"].detach()
        sb = spotlight(f, s, hist, kappa)[0]
        hr, lg, _ = upper_eval(st, h_in, NO_SHIFT, s, sb)
        L_ext = F.cross_entropy(lg[:, 0].float(), target)
        with torch.no_grad():
            S = torch.cat([unit(torch.randn(cfg["K"], cfg["m"], generator=pg)).to(R.dev), s], 0)
            b_idx = torch.arange(B, device=R.dev).repeat_interleave(S.shape[0])
            SS = S.repeat(B, 1)
            hrp = R.upper_probe(st, h_in, NO_SHIFT, SS, b_idx, sb=spotlight(f, SS, hist[b_idx], kappa)[0]).float()
            hist_p = torch.cat([hist[b_idx], hrp], 1)[:, -f.R:]
            hist_live = torch.cat([hist, hr.detach().float()], 1)[:, -f.R:]
        dest = f.next_stance(f.read(SS, hist_p)).view(B, -1, cfg["m"])
        res = dest - dest.mean(1, keepdim=True)
        P = f.read(s, hist_live)
        L_dyn = 0.0
        for b in range(B):
            pd_ = f.predict_field(P[b:b + 1], S) + S
            L_dyn = L_dyn + contrastive(pd_ - pd_.mean(0, keepdim=True), res, b)[0] / B
        g_e = torch.autograd.grad(L_ext / L_ext.detach(), params, retain_graph=True, allow_unused=True)
        g_d = torch.autograd.grad(L_dyn / L_dyn.detach(), params, allow_unused=True)

    def flat(gs, idx):
        return torch.cat([(gs[i] if gs[i] is not None else torch.zeros_like(params[i])).reshape(-1) for i in idx])
    ge, gd = flat(g_e, reader), flat(g_d, reader)
    return dict(ext_on_reader=float(ge.norm()), game_on_reader=float(gd.norm()), ext_on_query=float(flat(g_e, query).norm()),
                game_on_query=float(flat(g_d, query).norm()), cosine=float((ge @ gd) / (ge.norm() * gd.norm()).clamp_min(1e-12)),
                L_ext=float(L_ext), L_dyn=float(L_dyn))


def consequences(f, cfg, kappa, ids):
    modes = ("own", "none", "neighbor", "shift")
    st = {m_: R.new_state(B) for m_ in modes}
    s = {m_: unit(torch.randn(B, cfg["m"], generator=torch.Generator().manual_seed(0))).to(R.dev) for m_ in modes}
    path = {m_: [] for m_ in modes}
    cmp = {m_: dict(kl=[], top=[], loss=[], loss_none=[], rl=[]) for m_ in ("own", "neighbor", "shift")}
    tok_dloss, tok_kl = [], []
    imm, tv, grads = [], [], []
    acc = {r_: [0.0, 0.0] for r_ in ("both", "reading", "spotlight")}
    S0 = unit(torch.randn(NP, cfg["m"], generator=torch.Generator().manual_seed(4242))).to(R.dev)
    pg = torch.Generator().manual_seed(77)
    with torch.no_grad():
        for t in range(TE):
            out = {}
            for m_ in modes:
                h_in = R.lower(st[m_], ids[:, t])
                sb = None
                if m_ in ("own", "neighbor"):
                    sb = spotlight(f, s[m_], st[m_]["hist"], kappa)[0]
                    if m_ == "neighbor" and sb is not None:
                        sb = torch.roll(sb, -1, 0)
                offs = f.offs if m_ == "shift" else NO_SHIFT
                if m_ == "own" and sb is not None and t >= T0:
                    _, lg0, _ = upper_eval(st[m_], h_in, offs, s[m_], None)
                    _, lg1, tv_ = upper_eval(st[m_], h_in, offs, s[m_], sb, att=True)
                    p0, p1 = F.log_softmax(lg0[:, 0].float(), -1), F.log_softmax(lg1[:, 0].float(), -1)
                    imm.append((float((p1.exp() * (p1 - p0)).sum(-1).mean()), float((p1.argmax(-1) != p0.argmax(-1)).float().mean())))
                    tv.append(tv_)
                if m_ == "own" and t >= TE // 3 and t % 10 == 0:
                    route_split_step(f, cfg, kappa, st[m_], s[m_], h_in, acc, S0)
                if m_ == "own" and t in GRAD_T:
                    grads.append(dict(t=t, **gradient_step(f, cfg, kappa, st[m_], s[m_], h_in, ids[:, t + 1], pg)))
                hr, lg = R.upper_live(st[m_], h_in, offs, s[m_], sb=sb)
                st[m_]["hist"] = torch.cat([st[m_]["hist"], hr.float()], 1)[:, -f.R:]
                out[m_] = (F.log_softmax(lg[:, 0].float(), -1), hr[:, 0].float())
                path[m_].append(s[m_].cpu().numpy())
                s[m_] = f.next_stance(f.read(s[m_], st[m_]["hist"]))
            if t >= T0:
                lb, hb = out["none"]
                tgt = ids[:, t + 1]
                la_, lb_ = out["own"][0], lb
                tok_dloss.append((-la_.gather(-1, tgt[:, None]) + lb_.gather(-1, tgt[:, None]))[:, 0].cpu().numpy())
                tok_kl.append((la_.exp() * (la_ - lb_)).sum(-1).cpu().numpy())
                for m_ in ("own", "neighbor", "shift"):
                    la, ha = out[m_]
                    c = cmp[m_]
                    c["kl"].append(float((la.exp() * (la - lb)).sum(-1).mean()))
                    c["top"].append(float((la.argmax(-1) != lb.argmax(-1)).float().mean()))
                    c["loss"].append(float(-la.gather(-1, tgt[:, None]).mean()))
                    c["loss_none"].append(float(-lb.gather(-1, tgt[:, None]).mean()))
                    c["rl"].append(float(((ha - hb).norm(dim=-1) / hb.norm(dim=-1)).mean()))

    def dyn(p):
        S_ = np.array(p)[TE // 3:]
        stp = np.linalg.norm(S_[1:] - S_[:-1], axis=-1)
        return dict(rests=float((stp < 0.01).mean()), big_steps=float((stp > 0.5).mean()))
    res = {"against_no_channel": {m_: dict(kl=float(np.mean(c["kl"])), top_prediction_changed=float(np.mean(c["top"])),
                                           loss=float(np.mean(c["loss"])), loss_without=float(np.mean(c["loss_none"])),
                                           loss_change_pct=100 * (np.mean(c["loss"]) / np.mean(c["loss_none"]) - 1),
                                           read_layer_change=float(np.mean(c["rl"]))) for m_, c in cmp.items()},
           "loop_dynamics": {m_: dyn(path[m_]) for m_ in modes},
           "immediate": dict(kl=float(np.mean([a for a, _ in imm])), top_prediction_changed=float(np.mean([b for _, b in imm])),
                             attention_moved=float(np.mean(tv))),
           "field_routes": {r_: a[0] / max(a[1], 1e-12) for r_, a in acc.items()},
           "gradients": grads, "_tokens": dict(dloss=np.array(tok_dloss, np.float32), kl=np.array(tok_kl, np.float32))}
    return res


NOCALIB = KARG.endswith(":nocalib")
KARG = KARG.split(":")[0]
for name, run_dir, iface in ITEMS:
    if NOCALIB:
        results[name] = {"interface": iface}
        continue
    cfg, f = load(run_dir, iface)
    results[name] = {"interface": iface, "calibration": calibrate(f, cfg)}
    print(name, "calibration", json.dumps({k: {a: round(b, 3) for a, b in v.items()} for k, v in results[name]["calibration"].items()}), flush=True)
    json.dump(results, open(OUT, "w"), indent=1)
if KARG == "auto":                    # the rule: read-layer change 0.2-0.4, loss < +10%, closest to 0.3; on the starts
    picks = []
    for name, res in results.items():
        if name.startswith("start"):
            ok = [(abs(v["read_layer_change"] - 0.3), float(k)) for k, v in res["calibration"].items()
                  if 0.2 <= v["read_layer_change"] <= 0.4 and v["loss_change_pct"] < 10]
            picks.append(min(ok)[1] if ok else None)
    good = [p_ for p_ in picks if p_ is not None]
    kappa = float(np.median(good)) if good else None
else:
    kappa = float(KARG)
results["chosen_kappa"] = kappa
print("chosen kappa", kappa, flush=True)
json.dump(results, open(OUT, "w"), indent=1)
if kappa is None:
    print("no kappa satisfies the rule: stop and report", flush=True)
    sys.exit(0)
if R is None:                              # with ":nocalib" nothing has loaded the model yet
    load(ITEMS[0][1], ITEMS[0][2])
ids, _, _ = make_streams(corpus, R.tok, B, TE + 5, 99)
ids = ids.to(R.dev)
for name, run_dir, iface in ITEMS:
    t0 = time.time()
    cfg, f = load(run_dir, iface)
    results[name].update(consequences(f, cfg, kappa, ids))
    np.savez_compressed(OUT.replace(".json", f"_{name}_tokens.npz"), **results[name].pop("_tokens"))
    r_ = results[name]
    print(name, json.dumps({k: r_[k] for k in ("against_no_channel", "loop_dynamics", "immediate", "field_routes")}), flush=True)
    print(name, "gradients", json.dumps(r_["gradients"]), f"{time.time() - t0:.0f}s", flush=True)
    json.dump(results, open(OUT, "w"), indent=1)
