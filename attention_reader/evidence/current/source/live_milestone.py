"""WHERE_ITS_OWN_STATE_GOES.md section 65 (runs on a GPU with the 8B model): the milestone configuration (turned stance,
habit, own attention schema; process_self.py) on the live model, with one route by which a copy's way of looking acts
on the model's processing and the consequences come back to what it reads.
- Layers 0..act_layer-1 read the shared text once, as in the recorded core (whose warm-up reads each copy's recorded own
  text, as here). From the act layer up, each copy has its own row of the model; one more row is plain.
- The act (from token `on`): at the input of the act layer, each copy's row receives the content its way of looking
  adds to its selection at its last reading, sum_l (a_l - a0_l) x_l. Here a is its attention over its window (heads
  averaged; with its stance, habit and schema), a0 the same reader's attention without any of them (its query bias
  alone), and x_l the shared act-layer inputs at the window's positions. Its size is set once per life, over the
  `calib` readings before `on`: typically `strength` times the size of the model's own update at the act layer
  (medians; section 65 ran with means, live.calib_stat "mean", which a single massive-activation token can inflate).
- The return: the copy reads its window from its own row's read-layer states. Its acts reach them through the model's
  layers from the act layer to the read layer, as well as passing up the residual stream.
Arms (live.arm): N no act (the recorded milestone loop on live states); A its own act; An its row receives its
neighbour's act; Ae (the echo) its act's content, made from the shared read-layer states with the same weights and
the same size factor, is added to the read-layer state it reads, with no layers between (the read-layer states are
larger than the act layer's, roughly as much as the model's processing amplifies A's act on its way up, so the echo
comes out about the size of A's departure: 0.6B smoke test, 11 against 12).
The rest is process_self.py's, unchanged: the turned stance; develop (the habit); schema (use own | neighbor; twin
swap | shift for [at, until)); without either, the turned stance alone.
(section 68) process_self "consequence" (arm A with the milestone): a model of what its own acts bring back. A copy of
its recent acts (their content, fading over `ema` tokens, projected to 16 numbers) predicts the part of its attention
pull its acts made: its pull minus the pull the same reading would have on the plain window (a target only, never an
input), learned online (recursive least squares, memory `tau`). use "none": learned only; "own": from `from` on, its
habit develops from what the world brought, its pull minus the predicted part; "neighbor": the model is learned from
its neighbour's acts and pulls. twin "swap" | "random" for [at, until): copies 0 and 1 take each other's record of
acts for their own, or a record of acts no one made, of the same size. Without the option, lives are as section 65's.
(section 73) consequence "target": "picture": a model of what its own acts put into its picture. Its record: the acts
it made at its window's positions, each in its own reader's value code (the reader's value map of the act, with the
reader's norm gain, scaled once per life so a typical act is 1), weighted by its attention there (heads averaged):
what it did where it looks. Its target: its picture minus the picture the same reading (stance, habit, schema) makes of
the plain window (a target only, never an input). Learned online: least squares on the 256 numbers with memory `tau`
and a fixed `ridge` (without one, a 1e-3 that fades with the memory). use "own" | "neighbor": from `from` on, its
picture is what it read less `strength` times what it expects its acts put there (back on the picture's sphere), and
its next stance comes from that picture; so the knowledge changes its interpretation, and the habit, the schema and the
knowledge itself then learn from the activity it changed. twin "swap": copies 0 and 1 weigh each other's acts by their
own attention; "random": acts no one made, of the same sizes. Logged: the self-caused part, its prediction and its
picture per token; the knowledge and the habit every 500 tokens.
(section 77) consequence "target": "keys": a model of its own echo in what draws its attention, position by
position. Its record: the act it made at each window position in its reader's key code (scaled once per life so a
typical act is 1). Its target, at the newest position: the key its own row has there minus the plain row's (target
only). Learned online (least squares on 256 numbers, memory `tau`, fixed `ridge`). use "own" | "neighbor": from `from`
on, it attends with its expected echo taken out of every position's key (the world's keys, as far as it knows), and
its schema, its habit's pull and the stance-free baseline of its act use the same keys; so the knowledge changes where
it looks, what it acts with and what its habit learns, and the knowledge then learns from the acts this changed. twin
"swap" | "random" as in section 73, on the per-position record. Logged: its prediction error at the newest position,
how far its attention is from the attention the same reader has on the plain window, the knowledge and the habit
every 500 tokens.
(section 83) consequence target "keys", "form": "gain": the knowledge is one number per copy, learned online
(fading least squares, memory `tau`): how strongly its acts come back in its keys, as its own reader sees them. Its
record at a position: the key of the state it observes there minus the key of that state without its act there (its
own reader's maps; no plain row). Its expected echo there: the gain times that. Used as in section 77. Logged: the gain.
(section 79, controls with the true counterfactual, not self-models) live "oracle": "keys": from `oracle_from` (default
`on`) it attends with the plain row's keys at every position, its own echo removed from what draws its attention (its
schema, habit pull and act baseline use the same keys, as in section 77); "values": it attends with its own keys but
reads the plain row's values, its own echo removed from what it reads.
(section 69) live "log_vocab": also sum, over the acting tokens, each copy's change in every next-token probability
and log-probability against the plain row (life.npz voc_dp, voc_dlp), for case studies of what its act changes. In
arms A and An every life also logs, per segment, the act as it entered (act_seg) and the rest of its row's read-layer
departure, the model's response (proc_seg).
(section 70) live "part" (arm A): the act uses only one part of its content, or all but it. "J": the part in the span
of its `part_k` best J-lens atoms at the act layer's input (layer `part_layer`, default act_layer - 1; the full
vocabulary; orthogonal matching pursuit, signed, least squares); "noJ": the content minus that part; "D", "noD": the
same with a random dictionary of the same size. The size factor is the full act's.
Logged per token: stances, attention and the stance-free attention, how differently the copies attend, the shared live
state against the recorded one, the act's size, how far its row's read-layer state departs from the plain row's (and
the part that is not its current act passing up), its prediction's change against the plain row, the schema's errors,
the exact self-caused change of its destination, and each row's loss on the text's actual next token (does the
model still read the text). Per 100-token segment: what each copy takes up (for the lens), its
row's change in each kind's next-token log-probability, its row's read-layer departure, the part of what it takes up
that its acts made, and its act's content at the read layer.
Usage: python live_milestone.py RUN_JSON [RUN_JSON ...]   (one model load for all; finished lives are skipped)"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core32 import DEV, build, carry, load_states, read_trace, unit  # noqa: E402
from lexicons import token_sets  # noqa: E402
from live_return import stance_free  # noqa: E402
from loop12 import Runner12, squash  # noqa: E402
from process_self import read_q  # noqa: E402

SEG = 100
_DICTS = {}


def part_dictionary(Rm, kind, path, layer):
    """(section 70) Unit atoms at the act layer's input: the J lens's (the output map centred, times the final norm, times
    that layer's J; the full vocabulary, as section 68) or a random dictionary of the same size. Built once."""
    key = (kind, path, layer)
    if key not in _DICTS:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
        sys.path.insert(0, "/workspace/code")
        from strange_loop import JSpace
        Js = torch.load(path, map_location="cpu", weights_only=True)["J"]
        J = Js[str(layer)] if str(layer) in Js else Js[layer]
        D = JSpace(Rm.model, Rm.tok, J, k=25, vocab="full", center=True).D
        if kind == "D":
            R = torch.randn(D.shape, generator=torch.Generator().manual_seed(71)).to(D.device)
            D = (R / R.norm(dim=-1, keepdim=True)).half()
        _DICTS[key] = D
    return _DICTS[key]


def sparse_part(X, D, k):
    """(section 70) The part of each row of X (N, d) in the span of its k best atoms of D: orthogonal matching pursuit,
    signed, refit by least squares at every step."""
    X = X.to(D.device).float()
    r, idx = X.clone(), torch.zeros((X.shape[0], 0), dtype=torch.long, device=X.device)
    fit = torch.zeros_like(X)
    for j in range(k):
        c = (r.half() @ D.T).float().abs()
        if idx.shape[1]:
            c.scatter_(1, idx, -1.0)
        idx = torch.cat([idx, c.argmax(-1, keepdim=True)], 1)
        A = D[idx].float()
        G = A @ A.transpose(1, 2) + 1e-4 * torch.eye(j + 1, device=X.device)
        coef = torch.linalg.solve(G, A @ X[:, :, None])
        fit = (coef * A).sum(1)
        r = X - fit
    return fit


def read_q_off(f, s, hist, Qh, q_add, k_off):
    """(section 77) process_self.read_q with an expected echo taken out of every position's key."""
    N, Rp, _ = hist.shape
    a = f.rln(hist)
    k = f.rk(a).view(N, Rp, f.rh, -1) - k_off.view(N, Rp, f.rh, -1)
    v = f.rv(a).view(N, Rp, f.rh, -1)
    q = (f.rq_bias.expand(N, -1) + f.rq(s)).view(N, f.rh, -1) + q_add
    if Qh is not None:
        q = q + torch.einsum("nhdm,nm->nhd", Qh, s)
    att = torch.softmax(torch.einsum("nhd,nlhd->nhl", q, k) / math.sqrt(q.shape[-1]) + f.rpos[-Rp:][None, None], -1)
    r = torch.einsum("nhl,nlhd->nhd", att, v).reshape(N, -1)
    return squash(f.to_pic(r), f.geometry), att, k


def read_q_val(f, s, hist, Qh, q_add, vals):
    """(section 79) process_self.read_q with its keys from hist but the values read from vals (the plain row's states,
    (window, d))."""
    N, Rp, _ = hist.shape
    k = f.rk(f.rln(hist)).view(N, Rp, f.rh, -1)
    v = f.rv(f.rln(vals)).view(1, Rp, f.rh, -1).expand(N, -1, -1, -1)
    q = (f.rq_bias.expand(N, -1) + f.rq(s)).view(N, f.rh, -1) + q_add
    if Qh is not None:
        q = q + torch.einsum("nhdm,nm->nhd", Qh, s)
    att = torch.softmax(torch.einsum("nhd,nlhd->nhl", q, k) / math.sqrt(q.shape[-1]) + f.rpos[-Rp:][None, None], -1)
    r = torch.einsum("nhl,nlhd->nhd", att, v).reshape(N, -1)
    return squash(f.to_pic(r), f.geometry), att, k


def stance_free_off(f, hist, k_off):
    """(section 77) live_return.stance_free on the same keys (the expected echo taken out)."""
    N, Rp, _ = hist.shape
    k = f.rk(f.rln(hist)).view(N, Rp, f.rh, -1) - k_off.view(N, Rp, f.rh, -1)
    q = f.rq_bias.expand(N, -1).view(N, f.rh, -1)
    return torch.softmax(torch.einsum("nhd,nlhd->nhl", q, k) / math.sqrt(q.shape[-1]) + f.rpos[-Rp:][None, None], -1).mean(1)


def life(cfg, Rm, kind_ix):
    t0 = time.time()
    out = cfg["out"]
    os.makedirs(out, exist_ok=True)
    Z = load_states(cfg["states"])
    B, T, m, warm = cfg["B"], cfg["T"], cfg["m"], int(cfg["warmup"])
    lv = {"act_layer": 16, "on": 300, "calib": 28, "strength": 0.5} | cfg["live"]
    arm, AL, on, ncal = lv["arm"], int(lv["act_layer"]), int(lv["on"]), int(lv["calib"])
    oracle, o_from = lv.get("oracle"), int(lv.get("oracle_from", lv.get("on", 300)))   # (79) controls
    assert oracle in (None, "keys", "values") and (oracle is None or arm == "A")
    RL, L, NR = Rm.read_layer, Rm.L, B + 1                       # rows: the copies, then the plain row
    assert arm in ("N", "A", "An", "Ae") and 0 < AL <= RL
    Hrec = torch.tensor(Z["h"][:B, :T + 1], dtype=torch.float32, device=DEV)   # own texts (warm-up); stream 0 (check)
    ids0 = torch.tensor(Z["ids"][0, :T + 1], dtype=torch.long, device=Rm.dev)
    torch.manual_seed(cfg["init_seed"])
    f = build(cfg, Z)
    for p_ in f.parameters():
        p_.requires_grad_(False)
    assert on - ncal >= warm + f.R - 1                           # the calibration reads live windows only
    s = unit(torch.randn(B, m, generator=torch.Generator().manual_seed(cfg["init_seed"]))).to(DEV)
    s = s[:1].repeat(B, 1)                                       # every copy from the same first stance
    ps = cfg.get("process_self", {})
    dv = ps.get("develop")                                       # the habit (process_self.py section 41, weight none)
    Qh = None
    if dv is not None:
        dv = {"weight": "none", "size": 0.5, "fill": 1000.0} | dv
        assert dv["weight"] == "none"
        Qh = torch.zeros(B, f.rh, f.rq_bias.shape[0] // f.rh, m, device=DEV)
        q_cap, q_rate = float(dv["size"] * f.rq.weight.norm()), None
    sch = ps.get("schema")                                       # the attention schema (process_self.py section 45)
    if sch is not None:
        sch = {"use": "own", "twin": None, "from": 300, "at": 1200, "until": 1500, "tau": 500.0, "strength": 0.5} | sch
        assert sch["use"] in ("own", "neighbor") and sch["twin"] in (None, "swap", "shift")
        dq = f.rq_bias.shape[0]
        Pc = torch.randn(16, dq, generator=torch.Generator().manual_seed(1616)).to(DEV) / math.sqrt(dq)
        Zs, Cs = torch.zeros(B, 16, 16, device=DEV), torch.zeros(B, dq, 16, device=DEV)
        lam_s, eye16, kappa, sch_shift = 1.0 - 1.0 / sch["tau"], torch.eye(16, device=DEV), None, None
    perm_ = torch.tensor([1, 0] + list(range(2, B)), device=DEV)
    nseg, d, nk = (T + SEG - 1) // SEG, Hrec.shape[-1], len(kind_ix)
    cq = ps.get("consequence")                                   # (section 68) a model of what its own acts bring back:
    if cq is not None:                                           # from a copy of its recent acts to the part of its
        cq = {"use": "own", "twin": None, "at": 1000, "until": 2000, "tau": 500.0, "ema": 8.0, "from": 400,
              "target": "pull", "strength": 1.0} | cq
        assert cq["use"] in ("none", "own", "neighbor") and cq["twin"] in (None, "swap", "random")
        assert cq["target"] in ("pull", "picture", "keys") and cq.get("record", "mean") in ("mean", "heads")
        assert sch is not None and Qh is not None and arm == "A"  # attention pull its acts made (target only)
        lam_c, c_hat = 1.0 - 1.0 / cq["tau"], None
        if cq["target"] == "pull":
            Pu = torch.randn(16, d, generator=torch.Generator().manual_seed(1717)).to(DEV) / math.sqrt(d)
            Zc, Cc = torch.zeros(B, 16, 16, device=DEV), torch.zeros(B, f.rq_bias.shape[0], 16, device=DEV)
            ubar = torch.zeros(B, d, device=DEV)
            u_rand = unit(torch.randn(B, 16, generator=torch.Generator().manual_seed(6868))).to(DEV) * 4.0
        elif cq["target"] == "keys":                             # (77) its echo in what draws its attention
            n_key = f.rk.weight.shape[0]
            kbuf = torch.zeros(B, f.R, n_key, device=DEV)        # its acts at its window's positions, in its key code
            Zk = torch.zeros(B, n_key, n_key, dtype=torch.float64, device=DEV)
            Ck = torch.zeros(B, n_key, n_key, dtype=torch.float64, device=DEV)
            ridge_k, eye_k = cq.get("ridge", 0.1), torch.eye(n_key, dtype=torch.float64, device=DEV)
            know_k = lambda: torch.linalg.solve(Zk + ridge_k * eye_k, Ck.transpose(1, 2)).transpose(1, 2).float()  # noqa: E731
            k_scale, cal_k, Mk = None, [], None
            g_rnd = torch.Generator().manual_seed(6868)
            kcode = lambda a: ((a - a.mean(-1, keepdim=True)) * f.rln.weight) @ f.rk.weight.T  # noqa: E731
            form = cq.get("form", "map")                         # (83) "gain": one learned gain on its acts as seen
            assert form in ("map", "gain")
            abuf = torch.zeros(B, f.R, d, device=DEV)            # its acts at its window's positions, as applied
            g_num, g_den, g_now = torch.zeros(B, device=DEV), torch.full((B,), 1e-6, device=DEV), None
            keyf = lambda h: f.rk(f.rln(h))  # noqa: E731
        else:                                                    # (73) its acts where it looks, in its value code
            n_val = f.rv.weight.shape[0]
            ebuf = torch.zeros(B, f.R, n_val, device=DEV)        # its acts at its window's positions
            Pc_inv = torch.eye(n_val, dtype=torch.float64, device=DEV).repeat(B, 1, 1) / 1e-3
            Cc = torch.zeros(B, f.p, n_val, dtype=torch.float64, device=DEV)
            ridge, eye_v = cq.get("ridge"), torch.eye(n_val, dtype=torch.float64, device=DEV)
            Zc = torch.zeros(B, n_val, n_val, dtype=torch.float64, device=DEV)
            know = lambda: ((Cc @ Pc_inv) if ridge is None else torch.linalg.solve(    # noqa: E731  its knowledge now:
                Zc + ridge * eye_v, Cc.transpose(1, 2)).transpose(1, 2)).float()        # least squares on what it has
            #                                     seen (forgetting) with a fixed ridge; without one, a 1e-3 that fades
            e_scale, cal_e, x_own = None, [], None
            g_rnd = torch.Generator().manual_seed(6868)
            vcode = lambda a: ((a - a.mean(-1, keepdim=True)) * f.rln.weight) @ f.rv.weight.T  # noqa: E731
    rec = {k: np.zeros((T, B, m), np.float32) for k in ("s", "g", "self_dg")}
    rec.update({k: np.full((T, B), np.nan, np.float32) for k in ("act", "dep", "dep_proc", "tvd", "q_size", "sch_err",
                                                                 "sch_err_other", "sch_tot", "align_used", "align_own")})
    rec["differ"] = np.zeros((T, B), bool)
    rec.update({k: np.full(T, np.nan, np.float32) for k in ("att_apart", "hdiff", "dev_ref", "upd", "upd24")})
    rec["nll"] = np.full((T, NR), np.nan, np.float32)
    rec.update({k: np.full((T, B, f.R), np.nan, np.float16) for k in ("att", "att0")})
    rec.update({k: np.zeros((nseg, B, d), np.float32) for k in ("lens_seg", "dep_seg", "tk_seg", "u24_seg")})
    if arm in ("A", "An"):
        rec.update({k: np.zeros((nseg, B, d), np.float32) for k in ("act_seg", "proc_seg")})
    rec["eff_seg"], rec["n_live"] = np.zeros((nseg, B, nk), np.float32), np.zeros(nseg, np.int32)
    if cq is not None:
        rec.update({k: np.full((T, B), np.nan, np.float32) for k in ("c_err", "c_err_other", "c_tot", "c_used", "kap_tot")})
        if cq["target"] == "picture":                            # (73) the self-caused part and its prediction (used)
            rec.update({k: np.full((T, B, f.p), np.nan, np.float32) for k in ("c_vec", "c_hat_vec", "pic")})
            rec.update({k: np.full((T, B), np.nan, np.float32) for k in ("c_err_rec", "x_size")})  # own knowledge with
            rec["Bc_snap"] = np.zeros((T // 500, B, f.p, n_val), np.float32)   # another copy's record; its size
            rec["Q_snap"] = np.zeros((T // 500,) + tuple(Qh.shape), np.float32)
        if cq["target"] == "keys":                               # (77) its errors, the record's size, how far its
            rec.update({k: np.full((T, B), np.nan, np.float32) for k in ("c_err_rec", "x_size", "atv", "gain")})
            if form == "map":                                    # attention is from the plain window's; (83) its gain
                rec["Bc_snap"] = np.zeros((T // 500, B, n_key, n_key), np.float16)
            rec["Q_snap"] = np.zeros((T // 500,) + tuple(Qh.shape), np.float32)
    part = lv.get("part")                                        # (section 70) only one part of its act, or all but it
    assert part in (None, "J", "noJ", "D", "noD")
    if part is not None:
        assert arm == "A"
        Dpart = part_dictionary(Rm, part[-1], lv["jlens"], int(lv.get("part_layer", AL - 1)))
        rec["part_share"] = np.full((T, B), np.nan, np.float32)
    voc = None                                                   # (section 69) live.log_vocab: what its act changes in
    if lv.get("log_vocab") and arm != "N":                       # the model's next-token prediction, over the vocabulary
        nv = Rm.model.lm_head.weight.shape[0]
        voc = {k: torch.zeros(B, nv, device=Rm.dev) for k in ("dp", "dlp")}
    lens_e, lens_l = torch.zeros(B, d, device=DEV), torch.zeros(B, d, device=DEV)

    z0 = lambda n: torch.zeros(n, Rm.kvh, 0, Rm.hd, device=Rm.dev, dtype=Rm.dtype)  # noqa: E731
    rows_from = AL if arm in ("A", "An") else RL + 1             # where the copies' own rows begin
    up = arm != "N"                                              # rows above the read layer, for the predictions
    K = {l: z0(1 if l < rows_from else NR) for l in range(rows_from if not up else L)}
    V = {l: z0(1 if l < rows_from else NR) for l in range(rows_from if not up else L)}
    hist = torch.zeros(B, 0, d, device=DEV)                      # what each copy reads (its row after the warm-up)
    xwin = torch.zeros(0, d, device=DEV)                         # the shared act-layer inputs at the window's positions
    pwin = torch.zeros(0, d, device=DEV)                         # the plain read-layer states there
    U = U24 = alpha = alpha24 = Ek = None
    cal = []
    pos, s_prev, g_prev = 0, None, None

    def run(l, h):
        h2, k_, v_ = Rm.layer(l, h, K[l], V[l], pos)
        K[l] = Rm.crop(torch.cat([K[l], k_], 2))
        V[l] = Rm.crop(torch.cat([V[l], v_], 2))
        return h2

    with torch.no_grad():
        for t in range(T):
            acting = up and t >= on and U is not None
            h = Rm.model.model.embed_tokens(ids0[t:t + 1].view(-1, 1))   # (1, 1, d) the real token, shared
            x_act = upd = upd24 = None
            for l in range(rows_from):
                if l == AL:
                    x_act = h
                h_ = run(l, h)
                if l == AL:
                    upd = float((h_ - h)[0, 0].float().norm())
                if l == RL:
                    upd24 = float((h_ - h)[0, 0].float().norm())
                h = h_
            Urow = None
            if rows_from == AL:                                  # A, An: each copy's row from the act layer
                x_act = h
                hr = h.expand(NR, -1, -1).clone()
                if acting:
                    Urow = U if arm == "A" else torch.roll(U, -1, 0)
                    if part is not None:                         # (70) its act's part in the span of its best atoms
                        fit = sparse_part(Urow, Dpart, int(lv.get("part_k", 25))).to(DEV)
                        rec["part_share"][t] = ((fit ** 2).sum(-1) / (Urow ** 2).sum(-1).clamp_min(1e-12)).cpu().numpy()
                        Urow = fit if part in ("J", "D") else Urow - fit
                    hr[:B] = hr[:B] + (alpha * Urow).to(Rm.dtype)[:, None].to(Rm.dev)
                    if cq is not None and cq["target"] == "pull":    # (68) a copy of what it did, fading over ~ema tokens
                        ubar = (1.0 - 1.0 / cq["ema"]) * ubar + U / cq["ema"]
                for l in range(AL, L):
                    h_ = run(l, hr)
                    if l == AL:
                        upd = float((h_[B] - hr[B])[0].float().norm())
                    if l == RL:
                        upd24 = float((h_[B] - hr[B])[0].float().norm())
                        h24 = h_[:, 0].float().to(DEV)
                    hr = h_
                logits = Rm.model.lm_head(Rm.model.model.norm(hr))[:, 0].float()
                h_plain, h_own = h24[B], h24[:B]
            else:                                                # N, Ae: shared up to the read layer
                h_plain = h[0, 0].float().to(DEV)
                h_own = h_plain[None].expand(B, -1)
                if acting:                                       # Ae: its act's content added where it reads
                    Urow = U24
                    h_own = h_plain[None] + alpha * U24
                if up:
                    hr = torch.cat([h_own, h_plain[None]], 0)[:, None].to(Rm.dtype).to(Rm.dev)
                    for l in range(RL + 1, L):
                        hr = run(l, hr)
                    logits = Rm.model.lm_head(Rm.model.model.norm(hr))[:, 0].float()
            pos += 1
            x_act = x_act[0, 0].float().to(DEV)
            rec["hdiff"][t] = float((h_plain - Hrec[0, t]).norm() / Hrec[0, t].norm().clamp_min(1e-6))
            rec["upd"][t], rec["upd24"][t] = upd, upd24
            k_s = t // SEG
            if acting:                                           # what its act does to the model
                rec["act"][t] = (alpha * Urow).norm(dim=-1).cpu().numpy()
                dep_v = h_own - h_plain[None]
                rec["dep"][t] = dep_v.norm(dim=-1).cpu().numpy()
                if arm in ("A", "An"):                           # the part that is not its current act passing up
                    rec["dep_proc"][t] = (dep_v - alpha * Urow.to(DEV)).norm(dim=-1).cpu().numpy()
                    rec["act_seg"][k_s] += (alpha * Urow).cpu().numpy()          # (69) the act as it entered, and
                    rec["proc_seg"][k_s] += (dep_v - alpha * Urow.to(DEV)).cpu().numpy()   # the model's response
                pr = torch.softmax(logits, -1)
                rec["tvd"][t] = (0.5 * (pr[:B] - pr[B:]).abs().sum(-1)).cpu().numpy()
                rec["differ"][t] = (logits[:B].argmax(-1) != logits[B].argmax(-1)).cpu().numpy()
                lp = torch.log_softmax(logits, -1)
                rec["nll"][t] = (-lp[:, ids0[t + 1]]).cpu().numpy()
                kl = torch.stack([torch.logsumexp(lp[:, ix], -1) for ix in kind_ix], -1)
                rec["eff_seg"][k_s] += (kl[:B] - kl[B:]).cpu().numpy()
                if voc is not None:                              # (section 69) the change in every next-token
                    voc["dp"] += (pr[:B] - pr[B:]).float()       # probability against the plain row, summed
                    voc["dlp"] += (lp[:B] - lp[B:]).float()
                rec["dep_seg"][k_s] += dep_v.cpu().numpy()
                rec["n_live"][k_s] += 1
            h_in = Hrec[:, t] if t < warm else h_own             # the warm-up: each copy's own text
            hist = torch.cat([hist, h_in[:, None]], 1)[:, -f.R:]
            if cq is not None and cq["target"] == "picture":    # (73) what it did at this position, in its value code
                e_new = vcode(alpha * Urow) * e_scale if acting and arm == "A" else torch.zeros(B, n_val, device=DEV)
                ebuf = torch.cat([ebuf, e_new[:, None]], 1)[:, -f.R:]
            if cq is not None and cq["target"] == "keys":       # (77) what it did at this position, in its key code,
                e_k = kcode(alpha * Urow) * k_scale if acting and arm == "A" else torch.zeros(B, n_key, device=DEV)
                kbuf = torch.cat([kbuf, e_k[:, None]], 1)[:, -f.R:]
                a_new = alpha * Urow if acting and arm == "A" else torch.zeros(B, d, device=DEV)
                abuf = torch.cat([abuf, a_new[:, None].to(abuf.dtype)], 1)[:, -f.R:]
                Ek = None                                        # and what it expects that made of each key
                if acting and hist.shape[1] == f.R and form == "gain":   # (83) its gain on its acts as it sees them
                    g_now = g_num / g_den
                    k_obs = keyf(hist)
                    seen_own = k_obs - keyf(hist - abuf)
                    seen_used = seen_own
                    if cq["at"] <= t < cq["until"] and cq["twin"] == "swap":
                        seen_used = k_obs - keyf(hist - abuf[perm_])     # copies 0 and 1 take each other's acts
                    elif cq["at"] <= t < cq["until"] and cq["twin"] == "random":
                        rnd = torch.randn(abuf.shape, generator=g_rnd).to(DEV)   # or acts no one made, same sizes
                        rnd = rnd * abuf.norm(dim=-1, keepdim=True) / rnd.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                        seen_used = torch.cat([(k_obs - keyf(hist - rnd))[:2], seen_own[2:]], 0)
                    E_exp = g_now[:, None, None] * seen_used
                    rec["c_used"][t] = (E_exp ** 2).sum(-1).mean(-1).cpu().numpy()
                    if cq["use"] != "none" and t >= cq["from"]:
                        Ek = E_exp
                elif acting and hist.shape[1] == f.R:
                    Mk = know_k()
                    kb = kbuf
                    if cq["at"] <= t < cq["until"] and cq["twin"] == "swap":
                        kb = kbuf[perm_]                         # copies 0 and 1 take each other's acts for their own
                    elif cq["at"] <= t < cq["until"] and cq["twin"] == "random":
                        rnd = torch.randn(kbuf.shape, generator=g_rnd).to(DEV)   # or acts no one made, same sizes
                        rnd = rnd * kbuf.norm(dim=-1, keepdim=True) / rnd.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                        kb = torch.cat([rnd[:2], kbuf[2:]], 0)
                    E_exp = torch.einsum("bij,blj->bli", Mk, kb)
                    rec["c_used"][t] = (E_exp ** 2).sum(-1).mean(-1).cpu().numpy()
                    if cq["use"] != "none" and t >= cq["from"]:
                        Ek = E_exp
            xwin = torch.cat([xwin, x_act[None]], 0)[-f.R:]
            pwin = torch.cat([pwin, h_plain[None]], 0)[-f.R:]
            if pwin.shape[0] > 1:
                rec["dev_ref"][t] = float((h_plain - pwin.mean(0)).norm())
            o_on = oracle is not None and acting and t >= o_from and hist.shape[1] == f.R
            if oracle == "keys":                                 # (79) its whole echo out of what draws its attention
                Ek = (f.rk(f.rln(hist)) - f.rk(f.rln(pwin))[None]) if o_on else None

            q_add = None                                         # from here as process_self.py (milestone arms)
            if sch is not None:
                kk = f.rk(f.rln(hist)).view(B, hist.shape[1], f.rh, -1)
                if Ek is not None:                               # (77) the world's keys, as far as it knows
                    kk = kk - Ek.view(B, hist.shape[1], f.rh, -1)
                z = unit(torch.einsum("ij,bj->bi", Pc, kk.mean(1).reshape(B, -1))) * 4.0
                A_s = Cs @ torch.linalg.inv(Zs + 1e-3 * eye16)
                A_u = A_s
                in_win = sch["at"] <= t < sch["until"]
                if in_win and sch_shift is None:                 # a random change of the size of copies 0 and 1's
                    gs_ = torch.Generator().manual_seed(4646)    # difference
                    dn_ = (A_s[0] - A_s[1]).norm()
                    sch_shift = torch.zeros_like(A_s)
                    for b_ in (0, 1):
                        r_ = torch.randn(A_s.shape[1:], generator=gs_).to(DEV)
                        sch_shift[b_] = r_ * dn_ / r_.norm().clamp_min(1e-12)
                if sch["twin"] is not None and in_win:
                    A_u = A_s[perm_] if sch["twin"] == "swap" else A_s + sch_shift
                pull_u = torch.einsum("bij,bj->bi", A_u, z)
                if t >= sch["from"]:
                    if kappa is None:                            # its strength, set once (the milestone's rule)
                        if sch.get("calib", "habit") == "habit":
                            kappa = float(sch["strength"] * q_cap / pull_u.norm(dim=-1).mean().clamp_min(1e-12))
                        else:
                            ws_ = (s @ f.rq.weight.T).norm(dim=-1).mean()
                            kappa = float(sch["rho"] * ws_ / pull_u.norm(dim=-1).mean().clamp_min(1e-12))
                    q_add = (kappa * pull_u).view(B, f.rh, -1)
            if q_add is not None:
                if oracle == "values" and o_on:                 # (79) its whole echo out of what it reads
                    P_, att_, k_ = read_q_val(f, s, hist, Qh, q_add, pwin)
                else:
                    P_, att_, k_ = read_q(f, s, hist, Qh, q_add) if Ek is None else read_q_off(f, s, hist, Qh, q_add, Ek)
            elif Qh is None:
                P_ = f.read(s, hist)
                att_, k_ = f._last_att, None
            else:
                P_, att_, k_ = read_trace(f, s, hist, Qh)
            P_used = P_
            if cq is not None and cq["target"] == "picture" and acting and hist.shape[1] == f.R:
                if cq.get("record", "mean") == "heads":          # (77) each head's attention on its own part of the code
                    weigh = lambda E: torch.einsum("nhl,nlhd->nhd", att_, E.view(B, f.R, f.rh, -1)).reshape(B, -1)  # noqa: E731
                else:                                            # (73) what it did where it looks (heads averaged)
                    a_h = att_.mean(1)
                    weigh = lambda E: torch.einsum("nl,nlv->nv", a_h, E)  # noqa: E731
                x_own = weigh(ebuf)
                x_used = x_own
                if cq["at"] <= t < cq["until"] and cq["twin"] == "swap":       # copies 0 and 1 weigh each other's acts
                    x_used = weigh(ebuf[perm_])                  # by their own attention
                elif cq["at"] <= t < cq["until"] and cq["twin"] == "random":   # or acts no one made, of the same sizes
                    rnd = torch.randn(ebuf.shape, generator=g_rnd).to(DEV)
                    rnd = rnd * ebuf.norm(dim=-1, keepdim=True) / rnd.norm(dim=-1, keepdim=True)
                    x_used = torch.cat([weigh(rnd)[:2], x_own[2:]], 0)
                Bc = know()
                c_hat = torch.einsum("bij,bj->bi", Bc, x_used)   # what it expects its acts put into its picture
                if cq["use"] != "none" and t >= cq["from"]:      # its picture: what it read, less that
                    P_used = squash(P_ - cq["strength"] * c_hat, f.geometry)
            g = f.next_stance(P_used)
            if sch is not None:                                  # where its attention did pull; it learns
                kp = (torch.einsum("nhl,nlhd->nhd", att_, kk) - kk.mean(1)).reshape(B, -1)
                pred = torch.einsum("bij,bj->bi", A_s, z)
                rec["sch_err"][t] = ((pred - kp) ** 2).sum(-1).cpu().numpy()
                rec["sch_err_other"][t] = ((torch.einsum("bij,bj->bi", torch.roll(A_s, -1, 0), z) - kp) ** 2).sum(-1).cpu().numpy()
                rec["sch_tot"][t] = (kp ** 2).sum(-1).cpu().numpy()
                cosu = (pull_u * kp).sum(-1) / (pull_u.norm(dim=-1) * kp.norm(dim=-1)).clamp_min(1e-12)
                coso = (pred * kp).sum(-1) / (pred.norm(dim=-1) * kp.norm(dim=-1)).clamp_min(1e-12)
                rec["align_used"][t], rec["align_own"][t] = cosu.cpu().numpy(), coso.cpu().numpy()
                if in_win:
                    for key_, Aw in (("align_swap", A_s[perm_]), ("align_shift", A_s + sch_shift)):
                        pw = torch.einsum("bij,bj->bi", Aw, z)
                        rec.setdefault(key_, np.full((T, B), np.nan, np.float32))[t] = (
                            (pw * kp).sum(-1) / (pw.norm(dim=-1) * kp.norm(dim=-1)).clamp_min(1e-12)).cpu().numpy()
                zl, kl_ = (torch.roll(z, -1, 0), torch.roll(kp, -1, 0)) if sch["use"] == "neighbor" else (z, kp)
                Zs = lam_s * Zs + torch.einsum("bi,bj->bij", zl, zl)
                Cs = lam_s * Cs + torch.einsum("bi,bj->bij", kl_, zl)
            if cq is not None and t >= on:                       # (68) the part of its pull its own acts made: its pull
                hp = pwin[None].expand(B, -1, -1)                # minus the same reading of the plain window (a target
                kk0 = f.rk(f.rln(hp)).view(B, hp.shape[1], f.rh, -1)   # only, never an input); (73) or of where its
                if q_add is not None:                            # reading points
                    z0_ = unit(torch.einsum("ij,bj->bi", Pc, kk0.mean(1).reshape(B, -1))) * 4.0
                    P0_, att0_ = read_q(f, s, hp, Qh, (kappa * torch.einsum("bij,bj->bi", A_u, z0_)).view(B, f.rh, -1))[:2]
                else:
                    P0_, att0_ = read_trace(f, s, hp, Qh)[:2]
            if cq is not None and t >= on and cq["target"] == "picture" and x_own is not None:
                c_t = P_ - P0_                                   # (73) what its acts put into its picture (target only)
                Bc_ = Bc                                         # (the same knowledge: nothing learned since)
                rec["c_err"][t] = ((torch.einsum("bij,bj->bi", Bc_, x_own) - c_t) ** 2).sum(-1).cpu().numpy()
                rec["c_err_other"][t] = ((torch.einsum("bij,bj->bi", torch.roll(Bc_, -1, 0), x_own) - c_t) ** 2).sum(-1).cpu().numpy()
                rec["c_err_rec"][t] = ((torch.einsum("bij,bj->bi", Bc_, torch.roll(x_own, -1, 0)) - c_t) ** 2).sum(-1).cpu().numpy()
                rec["c_tot"][t] = (c_t ** 2).sum(-1).cpu().numpy()
                rec["c_used"][t] = (c_hat ** 2).sum(-1).cpu().numpy()
                rec["x_size"][t] = x_own.norm(dim=-1).cpu().numpy()
                rec["c_vec"][t], rec["c_hat_vec"][t] = c_t.cpu().numpy(), c_hat.cpu().numpy()
                rec["pic"][t] = P_.cpu().numpy()                 # its picture as read (before any subtraction)
                if t % 500 == 499:
                    rec["Bc_snap"][t // 500], rec["Q_snap"][t // 500] = Bc_.cpu().numpy(), Qh.cpu().numpy()
                xl, cl = (torch.roll(x_own, -1, 0), torch.roll(c_t, -1, 0)) if cq["use"] == "neighbor" else (x_own, c_t)
                xl, cl = xl.double(), cl.double()                # it learns (forgetting 1 - 1/tau)
                if ridge is None:                                # (Sherman-Morrison; the 1e-3 fades too)
                    Pc_inv = Pc_inv / lam_c
                    px = torch.einsum("bij,bj->bi", Pc_inv, xl)
                    Pc_inv = Pc_inv - torch.einsum("bi,bj->bij", px, px) / (1.0 + (xl * px).sum(-1))[:, None, None]
                else:
                    Zc = lam_c * Zc + torch.einsum("bi,bj->bij", xl, xl)
                Cc = lam_c * Cc + torch.einsum("bi,bj->bij", cl, xl)
            if cq is not None and t >= on and cq["target"] == "keys" and form == "gain" and g_now is not None:
                dk_t = keyf(hist[:, -1]) - keyf(pwin[-1:])      # (83) its echo at the newest position (target only)
                x_k = seen_own[:, -1]                            # and its act there, as it sees it
                rec["c_err"][t] = ((g_now[:, None] * x_k - dk_t) ** 2).sum(-1).cpu().numpy()
                rec["c_err_other"][t] = ((torch.roll(g_now, -1, 0)[:, None] * x_k - dk_t) ** 2).sum(-1).cpu().numpy()
                rec["c_err_rec"][t] = ((g_now[:, None] * torch.roll(x_k, -1, 0) - dk_t) ** 2).sum(-1).cpu().numpy()
                rec["c_tot"][t] = (dk_t ** 2).sum(-1).cpu().numpy()
                rec["x_size"][t] = x_k.norm(dim=-1).cpu().numpy()
                rec["atv"][t] = (0.5 * (att_ - att0_).abs().sum(-1)).mean(-1).cpu().numpy()
                rec["gain"][t] = g_now.cpu().numpy()
                if t % 500 == 499:
                    rec["Q_snap"][t // 500] = Qh.cpu().numpy()
                xl, cl = (torch.roll(x_k, -1, 0), torch.roll(dk_t, -1, 0)) if cq["use"] == "neighbor" else (x_k, dk_t)
                g_num = lam_c * g_num + (cl * xl).sum(-1)
                g_den = lam_c * g_den + (xl * xl).sum(-1)
            if cq is not None and t >= on and cq["target"] == "keys" and Mk is not None:
                dk_t = f.rk(f.rln(hist[:, -1])) - f.rk(f.rln(pwin[-1:]))   # (77) its echo at the newest position
                x_k = kbuf[:, -1]                                # (target only), and what it did there
                rec["c_err"][t] = ((torch.einsum("bij,bj->bi", Mk, x_k) - dk_t) ** 2).sum(-1).cpu().numpy()
                rec["c_err_other"][t] = ((torch.einsum("bij,bj->bi", torch.roll(Mk, -1, 0), x_k) - dk_t) ** 2).sum(-1).cpu().numpy()
                rec["c_err_rec"][t] = ((torch.einsum("bij,bj->bi", Mk, torch.roll(x_k, -1, 0)) - dk_t) ** 2).sum(-1).cpu().numpy()
                rec["c_tot"][t] = (dk_t ** 2).sum(-1).cpu().numpy()
                rec["x_size"][t] = x_k.norm(dim=-1).cpu().numpy()
                rec["atv"][t] = (0.5 * (att_ - att0_).abs().sum(-1)).mean(-1).cpu().numpy()
                if t % 500 == 499:
                    rec["Bc_snap"][t // 500], rec["Q_snap"][t // 500] = Mk.cpu().numpy(), Qh.cpu().numpy()
                xl, cl = (torch.roll(x_k, -1, 0), torch.roll(dk_t, -1, 0)) if cq["use"] == "neighbor" else (x_k, dk_t)
                xl, cl = xl.double(), cl.double()
                Zk = lam_c * Zk + torch.einsum("bi,bj->bij", xl, xl)
                Ck = lam_c * Ck + torch.einsum("bi,bj->bij", cl, xl)
            if cq is not None and t >= on and cq["target"] == "pull":
                c_t = kp - (torch.einsum("nhl,nlhd->nhd", att0_, kk0) - kk0.mean(1)).reshape(B, -1)
                u = unit(torch.einsum("ij,bj->bi", Pu, ubar)) * 4.0  # what it knows of its own recent acts (16 numbers)
                u_used = u
                if cq["at"] <= t < cq["until"] and cq["twin"] == "swap":
                    u_used = u[perm_]                            # copies 0 and 1 take each other's acts for their own
                elif cq["at"] <= t < cq["until"] and cq["twin"] == "random":
                    u_used = torch.cat([u_rand[:2], u[2:]], 0)   # or a record of acts no one made, of the same size
                Bc = Cc @ torch.linalg.inv(Zc + 1e-3 * eye16)
                c_hat = torch.einsum("bij,bj->bi", Bc, u_used)   # what it expects its acts brought back
                c_own = torch.einsum("bij,bj->bi", Bc, u)
                rec["c_err"][t] = ((c_own - c_t) ** 2).sum(-1).cpu().numpy()
                rec["c_err_other"][t] = ((torch.einsum("bij,bj->bi", torch.roll(Bc, -1, 0), u) - c_t) ** 2).sum(-1).cpu().numpy()
                rec["c_tot"][t] = (c_t ** 2).sum(-1).cpu().numpy()
                rec["c_used"][t] = (c_hat ** 2).sum(-1).cpu().numpy()
                rec["kap_tot"][t] = (kp ** 2).sum(-1).cpu().numpy()
                ul, cl = (torch.roll(u, -1, 0), torch.roll(c_t, -1, 0)) if cq["use"] == "neighbor" else (u, c_t)
                Zc = lam_c * Zc + torch.einsum("bi,bj->bij", ul, ul)
                Cc = lam_c * Cc + torch.einsum("bi,bj->bij", cl, ul)
            if lv.get("log_cdiag") and acting and q_add is not None:   # (73, diagnostic) what its acts bring back at
                hp = pwin[None].expand(B, -1, -1)                # every level of its reader, against the same reading
                kk0 = f.rk(f.rln(hp)).view(B, hp.shape[1], f.rh, -1)   # of the plain window; and its act, its content
                z0_ = unit(torch.einsum("ij,bj->bi", Pc, kk0.mean(1).reshape(B, -1))) * 4.0   # summary, the newest
                P0, a0_ = read_q(f, s, hp, Qh, (kappa * torch.einsum("bij,bj->bi", A_u, z0_)).view(B, f.rh, -1))[:2]
                v_ = f.rv(f.rln(hist)).view(B, hist.shape[1], f.rh, -1)  # departure
                v0 = f.rv(f.rln(hp)).view(B, hp.shape[1], f.rh, -1)
                cd = dict(dr=(torch.einsum("nhl,nlhd->nhd", att_, v_) - torch.einsum("nhl,nlhd->nhd", a0_, v0)).reshape(B, -1),
                          dP=P_ - P0, dg=g - f.next_stance(P0), U=U, z=z, dep=h_own - h_plain[None],
                          dk=(f.rk(f.rln(hist[:, -1])) - f.rk(f.rln(hp[:, -1]))),      # (77) its echo at the newest
                          dv=(f.rv(f.rln(hist[:, -1])) - f.rv(f.rln(hp[:, -1]))),      # position: key and value change
                          ek=((alpha * U - (alpha * U).mean(-1, keepdim=True)) * f.rln.weight) @ f.rk.weight.T,  # its act
                          ev=((alpha * U - (alpha * U).mean(-1, keepdim=True)) * f.rln.weight) @ f.rv.weight.T,  # coded
                          atv=0.5 * (att_ - a0_).abs().sum(-1),           # how far its echo moved its attention (heads)
                          hp=h_plain[None])                               # (82) the plain state at the newest position
                for nm_, x_ in cd.items():
                    if "cd_" + nm_ not in rec:
                        rec["cd_" + nm_] = np.full((T,) + tuple(x_.shape), np.nan, np.float16 if x_.shape[-1] > 300 else np.float32)
                    rec["cd_" + nm_][t] = x_.float().cpu().numpy()
            if lv.get("log_gself") and acting:                   # (diagnostic) where its reading would point on the
                hp = pwin[None].expand(B, -1, -1)                # plain window, with the same stance, habit and schema
                if q_add is not None:
                    kk0 = f.rk(f.rln(hp)).view(B, hp.shape[1], f.rh, -1)
                    z0_ = unit(torch.einsum("ij,bj->bi", Pc, kk0.mean(1).reshape(B, -1))) * 4.0
                    P0 = read_q(f, s, hp, Qh, (kappa * torch.einsum("bij,bj->bi", A_u, z0_)).view(B, f.rh, -1))[0]
                elif Qh is None:
                    P0 = f.read(s, hp)
                else:
                    P0 = read_trace(f, s, hp, Qh)[0]
                rec.setdefault("g_self", np.full((T, B), np.nan, np.float32))[t] = (g - f.next_stance(P0)).norm(dim=-1).cpu().numpy()
                if g_prev is not None:
                    rec.setdefault("g_step", np.full((T, B), np.nan, np.float32))[t] = (g - g_prev).norm(dim=-1).cpu().numpy()
            rec["s"][t], rec["g"][t] = s.cpu().numpy(), g.cpu().numpy()
            A_ = att_.mean(1)                                    # (B, window)
            a0 = stance_free(f, hist) if Ek is None else stance_free_off(f, hist, Ek)   # the same reader without
            #                                                      stance, habit or schema (on the same keys)
            if A_.shape[1] == f.R:
                rec["att"][t], rec["att0"][t] = A_.cpu().numpy(), a0.cpu().numpy()
            rec["att_apart"][t] = float((0.5 * (A_[:, None] - A_[None]).abs().sum(-1)).sum() / (B * (B - 1)))
            a_ = torch.einsum("bl,bld->bd", A_, hist)            # what it takes up (its lens, per segment, early, late)
            v_ = (a_ - a_.mean(0, keepdim=True)) / a_.pow(2).mean(-1, keepdim=True).sqrt()
            rec["lens_seg"][k_s] += (v_ / SEG).cpu().numpy()
            if warm + 16 <= t < warm + 716:
                lens_e += v_ / 700.0
            if t >= T - 700:
                lens_l += v_ / 700.0
            if t >= warm + f.R - 1:                              # its window is live: the content its way of looking
                da = A_ - a0                                     # adds, for the next token's act
                U = torch.einsum("bl,ld->bd", da, xwin)
                U24 = torch.einsum("bl,ld->bd", da, pwin)
                if t >= on:
                    rec["tk_seg"][k_s] += torch.einsum("bl,bld->bd", A_, hist - pwin[None]).cpu().numpy()
                    rec["u24_seg"][k_s] += U24.cpu().numpy()
                if on - ncal <= t < on:
                    cal.append((upd, upd24, float(U.norm(dim=-1).mean()), float(U24.norm(dim=-1).mean())))
                    if cq is not None and cq["target"] == "picture":
                        cal_e.append(float(vcode(U).norm(dim=-1).mean()))
                    if cq is not None and cq["target"] == "keys":
                        cal_k.append(float(kcode(U).norm(dim=-1).mean()))
                if t == on - 1:                                  # its size, set once: the model's own update there
                    c_ = np.array(cal)                           # (section 69) medians: one massive-activation token
                    ctr = np.median if lv.get("calib_stat", "median") == "median" else np.mean   # (update 12,669
                    alpha = float(lv["strength"] * ctr(c_[:, 0]) / max(ctr(c_[:, 2]), 1e-12))   # against ~28) made
                    alpha24 = float(lv["strength"] * ctr(c_[:, 1]) / max(ctr(c_[:, 3]), 1e-12))  # the mean's act 16x
                    if cq is not None and cq["target"] == "picture":   # (73) its record's scale: a typical act is 1
                        e_scale = 1.0 / max(alpha * float(np.median(cal_e)), 1e-12)
                    if cq is not None and cq["target"] == "keys":      # (77) the same for its key code
                        k_scale = 1.0 / max(alpha * float(np.median(cal_k)), 1e-12)
            if g_prev is None:                                   # the first token: it stays
                new_s = s
            else:
                g_cf = f.next_stance((read_q(f, s_prev, hist, Qh, q_add) if Ek is None else
                                      read_q_off(f, s_prev, hist, Qh, q_add, Ek))[0] if q_add is not None else
                                     f.read(s_prev, hist) if Qh is None else read_trace(f, s_prev, hist, Qh)[0])
                rec["self_dg"][t] = (g - g_cf).cpu().numpy()     # the exact self-caused change (analysis only)
                new_s = carry(g_prev, g, s)                      # the turned stance
                if Qh is not None and hist.shape[1] == f.R:      # the habit
                    kap = torch.einsum("nhl,nlhd->nhd", att_, k_) - k_.mean(1)
                    if (cq is not None and cq["target"] == "pull" and cq["use"] != "none" and t >= cq["from"]
                            and c_hat is not None):
                        kap = kap - c_hat.view(B, f.rh, -1)      # (68) it develops from what the world brought
                    w_ = torch.einsum("nhd,nm->nhdm", kap, s)
                    if q_rate is None:
                        q_rate = float(q_cap / (dv["fill"] * float(w_.flatten(1).norm(dim=-1).mean().clamp_min(1e-12))))
                    Qh = Qh + q_rate * w_
                    Qh = Qh * (q_cap / Qh.flatten(1).norm(dim=-1).clamp_min(1e-12)).clamp(max=1.0).view(B, 1, 1, 1)
                    rec["q_size"][t] = (Qh.flatten(1).norm(dim=-1) / f.rq.weight.norm()).cpu().numpy()
            s_prev, s, g_prev = s, new_s, g
            if t % 300 == 0 or t == T - 1:
                print(f"[live {arm}] t={t} {time.time() - t0:.0f}s state vs recorded {rec['hdiff'][t]:.4f} apart "
                      f"{np.nanmean(rec['att_apart'][max(0, t - 99):t + 1]):.3f} act {np.nanmean(rec['act'][max(0, t - 99):t + 1]):.2f} "
                      f"dep {np.nanmean(rec['dep'][max(0, t - 99):t + 1]):.2f} proc {np.nanmean(rec['dep_proc'][max(0, t - 99):t + 1]):.2f} "
                      f"tvd {np.nanmean(rec['tvd'][max(0, t - 99):t + 1]):.3f}", flush=True)
    extra = dict(lens_early=lens_e.cpu().numpy(), lens_late=lens_l.cpu().numpy(), alpha=np.float32(alpha or np.nan),
                 alpha24=np.float32(alpha24 or np.nan), cal=np.array(cal, np.float32))
    if Qh is not None:
        extra["Q"] = Qh.cpu().numpy()
    if sch is not None:
        extra["A_s"], extra["kappa"] = (Cs @ torch.linalg.inv(Zs + 1e-3 * eye16)).cpu().numpy(), np.float32(kappa)
    if cq is not None:
        extra["Bc"] = ((Cc @ torch.linalg.inv(Zc + 1e-3 * eye16)) if cq["target"] == "pull" else
                       (g_num / g_den if form == "gain" else know_k()) if cq["target"] == "keys" else know()).cpu().numpy()
        if cq["target"] == "keys":
            extra["k_scale"] = np.float32(k_scale or np.nan)
        if cq["target"] == "picture":
            extra["e_scale"] = np.float32(e_scale or np.nan)
    if voc is not None:
        extra["voc_dp"], extra["voc_dlp"] = voc["dp"].cpu().numpy(), voc["dlp"].cpu().numpy()
    np.savez_compressed(os.path.join(out, "life.npz"), **rec, **extra)
    open(os.path.join(out, "DONE"), "w").write(f"{time.time() - t0:.1f}\n")


if __name__ == "__main__":
    cfgs = [json.load(open(p)) for p in sys.argv[1:]]
    c0 = cfgs[0]
    Rm = Runner12(c0["model"], device=DEV, dtype=torch.float32 if DEV == "cpu" else torch.bfloat16,
                  band=tuple(c0["band"]), read_layer=c0["read_layer"], window=c0["window"])
    kind_ix = [torch.tensor(v, device=Rm.dev) for v in token_sets(Rm.tok).values()]
    for cfg in cfgs:
        if os.path.exists(os.path.join(cfg["out"], "DONE")):
            continue
        print("life", os.path.basename(cfg["out"]), flush=True)
        life(cfg, Rm, kind_ix)
