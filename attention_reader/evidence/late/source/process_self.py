"""WHERE_ITS_OWN_STATE_GOES.md section 30: a self-model of its own interpreting process, in the shared fixed text with
token-by-token turning (the 8B one-world lives of sections 15-19, reproduced exactly when the self-model is off).
At each token every copy reads its window with its stance; its reading points to a destination g; the stance turns
(core32.carry). In T it turns by the motion of its destinations, g(t-1) -> g(t): the world's change and its own.
The self-model J (per copy, 8 x 8) is a learned model of how its own stance moves its reading: a change ds of its
stance moves its destination by about J ds. It is learned online (recursive least squares with forgetting) from
occasional self-observation: every `every`-th token it also reads the same window with its previous stance, and the
difference is the training target (the counterfactual is only a target, never an input). In the arms that use it,
the stance turns only by what the world brought: from the destination its own change alone would have given,
g(t-1) + J ds, to the actual one. Arms:
- T: no self-model; Sx: J learned, not used; S: its own J used; Sn: J learned from the neighbour's self-observation;
- Sr: a random J of the same size as its learned one (fixed random pattern per copy);
- Sswap / Sshift: twins of S in which copies 0 and 1 exchange J, or have J moved by a random change of the size of
  their difference, for tokens [swap_at, back_at).
For analysis only (never used by the loop) it logs at every token the exact self-caused change of the destination.
Usage: python process_self.py RUN_JSON | python process_self.py configs ROOT"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core32 import DEV, build, carry, load_states, read_trace, unit  # noqa: E402
from loop12 import squash  # noqa: E402

ARMS = ("T", "Sx", "S", "Sn", "Sr", "Sswap", "Sshift")


def read_q(f, s, hist, Qh, q_add):
    """(section 45) core32.read_trace with one more term added to the query (N, heads, dk): its schema's expected pull."""
    N, Rp, _ = hist.shape
    a = f.rln(hist)
    k = f.rk(a).view(N, Rp, f.rh, -1)
    v = f.rv(a).view(N, Rp, f.rh, -1)
    q = (f.rq_bias.expand(N, -1) + f.rq(s)).view(N, f.rh, -1) + q_add
    if Qh is not None:
        q = q + torch.einsum("nhdm,nm->nhd", Qh, s)
    att = torch.softmax(torch.einsum("nhd,nlhd->nhl", q, k) / math.sqrt(q.shape[-1]) + f.rpos[-Rp:][None, None], -1)
    r = torch.einsum("nhl,nlhd->nhd", att, v).reshape(N, -1)
    return squash(f.to_pic(r), f.geometry), att, k


def read_q_look(f, s, hist, Qh, q_add, e, scale, strength=1.0, eta=0.1):
    """(section 90) read_q, then its attention leaning toward the positions whose content would move its picture toward
    what it expects to make of this content (e, in picture space). How much leaning toward a position would move the
    picture is measured with its own picture map. scale None: set it so the lean's spread over positions is `strength`
    times its attention schema's. Returns the picture, the attention, the keys, the scale and how far the lean moved
    its attention."""
    N, Rp, _ = hist.shape
    a = f.rln(hist)
    k = f.rk(a).view(N, Rp, f.rh, -1)
    v = f.rv(a).view(N, Rp, f.rh, -1)
    q = (f.rq_bias.expand(N, -1) + f.rq(s)).view(N, f.rh, -1) + q_add
    if Qh is not None:
        q = q + torch.einsum("nhdm,nm->nhd", Qh, s)
    sc = torch.einsum("nhd,nlhd->nhl", q, k) / math.sqrt(q.shape[-1]) + f.rpos[-Rp:][None, None]
    att0 = torch.softmax(sc, -1)
    r0 = torch.einsum("nhl,nlhd->nhd", att0, v)
    P0 = squash(f.to_pic(r0.reshape(N, -1)), f.geometry)
    Hh, dv = r0.shape[1], r0.shape[2]
    D = torch.zeros(N, Hh, Rp, Hh, dv, device=hist.device)
    for h in range(Hh):                                          # head h leaning a little toward position l
        D[:, h, :, h] = v[:, :, h] - r0[:, h, None]
    Pl = squash(f.to_pic((r0[:, None, None] + eta * D).reshape(N, Hh, Rp, -1)), f.geometry)
    lean = torch.einsum("nhlp,np->nhl", (Pl - P0[:, None, None]) / eta, e)
    if scale is None:
        sch_term = torch.einsum("nhd,nlhd->nhl", q_add, k) / math.sqrt(q.shape[-1])
        scale = float(strength * sch_term.std(-1).mean() / lean.std(-1).mean().clamp_min(1e-12))
    att = torch.softmax(sc + scale * lean, -1)
    r = torch.einsum("nhl,nlhd->nhd", att, v).reshape(N, -1)
    return squash(f.to_pic(r), f.geometry), att, k, scale, 0.5 * (att - att0).abs().sum(-1).mean(-1)


def toward(a, b, k):
    """(section 36) The point a fraction k of the way from unit vector a to unit vector b along their great circle."""
    th = torch.arccos((a * b).sum(-1, keepdim=True).clamp(-1.0, 1.0))
    sn = torch.sin(th)
    out = (torch.sin((1.0 - k) * th) * a + torch.sin(k * th) * b) / sn.clamp_min(1e-12)
    return unit(torch.where(sn > 1e-6, out, a + k * (b - a)))


def life(cfg):
    t0 = time.time()
    out = cfg["out"]
    os.makedirs(out, exist_ok=True)
    Z = load_states(cfg["states"])
    B, T, m = cfg["B"], cfg["T"], cfg["m"]
    H = torch.tensor(Z["h"][:B, :T + 1], dtype=torch.float32, device=DEV)
    warm = int(cfg.get("warmup", 0))
    H = torch.cat([H[:, :warm], H[:1, warm:].repeat(B, 1, 1)], 1)    # one world after different pasts
    torch.manual_seed(cfg["init_seed"])
    f = build(cfg, Z)
    for p_ in f.parameters():
        p_.requires_grad_(False)
    temp = float(cfg.get("process_self", {}).get("temp", 1.0))
    if temp != 1.0:                                              # (section 38) softer choices among positions: the
        with torch.no_grad():                                    # reader's scores divided by a temperature
            for p_ in (f.rq_bias, f.rq.weight, f.rq.bias, f.rpos):
                p_.div_(temp)
    arm = cfg.get("arm", "T")
    assert arm in ARMS
    ps = {"every": 8, "tau": 64.0, "ridge": 1e-4, "swap_at": 1000, "back_at": 1400, "seed": 31, "use": 1.0, "gain": 1.0} | cfg.get("process_self", {})
    s = unit(torch.randn(B, m, generator=torch.Generator().manual_seed(cfg["init_seed"]))).to(DEV)
    s = s[:1].repeat(B, 1)                                       # every copy from the same first stance
    hist = torch.zeros(B, 0, H.shape[-1], device=DEV)
    A = torch.zeros(B, m, m, device=DEV)                         # its self-observations: sum ds ds' and sum dg ds'
    C = torch.zeros(B, m, m, device=DEV)
    lam = 1.0 - 1.0 / ps["tau"]
    eye = torch.eye(m, device=DEV)
    gR = torch.Generator().manual_seed(int(ps["seed"]) + int(cfg["init_seed"]))
    Rpat = torch.randn(B, m, m, generator=gR).to(DEV)
    Rpat = Rpat / Rpat.flatten(1).norm(dim=-1).view(B, 1, 1)     # (Sr) a fixed random pattern per copy
    shift = None
    ab = ps.get("ablate")                                        # (section 34) one mode of its self-model removed for
    trig = {"after": 400, "level": 0.1, "span": 50, "window": 300} | ps.get("trigger", {})   # `window` tokens, from
    t_on, sc_hist = -1, []                                       # the first chase under way after `after` tokens
    gA = torch.Generator().manual_seed(5151 + int(cfg["init_seed"]))
    ra, rb = unit(torch.randn(B, m, generator=gA)).to(DEV), unit(torch.randn(B, m, generator=gA)).to(DEV)
    rec = {k: np.zeros((T, B, m), np.float32) for k in ("s", "g", "self_dg")}
    rec["ghat"] = np.full((T, B, m), np.nan, np.float32)
    rec.update({k: np.full((T, B), np.nan, np.float32) for k in ("turn", "J_norm", "J_err", "J_err_other", "on_top",
                                                                 "on_second", "q_size", "q_weight")})
    rec["att_apart"] = np.full(T, np.nan, np.float32)
    keep = bool(ps.get("keep_att", False))                       # (section 58) for case studies: each copy's attention
    if keep:                                                     # over its window, and where its schemas expected it
        rec.update({k: np.full((T, B, f.R), np.nan, np.float32) for k in ("att", "exp_own", "exp_used")})
    dv = ps.get("develop")                                       # (section 41) a way of looking that develops: a slow
    Qh = None                                                    # per-copy change Q of its stance-to-query map
    if dv is not None:
        dv = {"weight": "none", "size": 0.5, "fill": 1000.0, "avg": 200.0} | dv
        assert dv["weight"] in ("none", "own", "neighbor", "own_rank", "neighbor_rank")
        err_win = []                                             # (section 43) its recent self-model errors
        Qh = torch.zeros(B, f.rh, f.rq_bias.shape[0] // f.rh, m, device=DEV)
        q_cap, q_rate, e_avg = float(dv["size"] * f.rq.weight.norm()), None, None
    lens_e, lens_l = torch.zeros(B, H.shape[-1], device=DEV), torch.zeros(B, H.shape[-1], device=DEV)
    SEG = 100                                                    # (section 43) what each copy takes up, per segment
    lens_seg = torch.zeros((T + SEG - 1) // SEG, B, H.shape[-1], device=DEV)
    sch = ps.get("schema")                                       # (section 45) a model of its own way of looking: from
    if sch is not None:                                          # the window's content (16 numbers) to where its
        sch = {"use": "own", "twin": None, "from": 300, "at": 1200, "until": 1500, "tau": 500.0, "strength": 0.5,
               "lose": None, "mix": None} | sch                  # (section 51) mix m: it looks with (1-m) own + m
        Zn, Cn = None, None                                      # a schema learned from its neighbour's attending
        assert sch["use"] in ("own", "none", "neighbor") and sch["twin"] in (None, "swap", "shift")
        assert sch["lose"] in (None, "top", "third", "random")    # (section 49) one distinction lost while in the window
        gL = torch.Generator().manual_seed(4949)
        lose_u = unit(torch.randn(B, f.rq_bias.shape[0], generator=gL)).to(DEV)
        lose_v = unit(torch.randn(B, 16, generator=gL)).to(DEV)
        rec["eng"] = np.full((T, B), np.nan, np.float32)
        rec["sch_sv"] = np.full(((T + 99) // 100, B, 16), np.nan, np.float32)
        dq = f.rq_bias.shape[0]                                  # attention pulls, learned online from its attending
        Pc = torch.randn(16, dq, generator=torch.Generator().manual_seed(1616)).to(DEV) / math.sqrt(dq)
        Zs, Cs = torch.zeros(B, 16, 16, device=DEV), torch.zeros(B, dq, 16, device=DEV)
        if sch["mix"] is not None:
            assert sch["use"] == "own"
            Zn, Cn = torch.zeros(B, 16, 16, device=DEV), torch.zeros(B, dq, 16, device=DEV)
        lam_s, eye16, kappa, sch_shift = 1.0 - 1.0 / sch["tau"], torch.eye(16, device=DEV), None, None
        rec.update({k: np.full((T, B), np.nan, np.float32) for k in ("sch_err", "sch_err_other", "sch_tot", "align_used",
                                                                     "align_own")})
    snap = ps.get("snap")                                        # (section 86) its picture replaced by the nearer of a
    protos = None                                                # few prototypes shared by the copies, before it turns
    if snap is not None:                                         # the stance
        protos = torch.tensor(np.load(snap["protos"]), dtype=torch.float32, device=DEV)
        protos = math.sqrt(protos.shape[-1]) * protos / protos.norm(dim=-1, keepdim=True)
    rec["pic"] = np.zeros((T, B, cfg["p"]), np.float16)          # the picture as read (before any replacement)
    rec["proto"] = np.full((T, B), -1, np.int8)

    def snapped(P, t):
        if protos is None or t < snap["from"]:
            return P, None
        ix = (P @ protos.T).argmax(-1)
        return protos[ix], ix

    isc = ps.get("interp")                                       # (section 88) a model of its own interpreting: from
    if isc is not None:                                          # the window's content (the attention schema's 16
        assert sch is not None                                   # numbers) to what its way of looking makes of its
        isc = {"use": "own", "twin": None, "from": 300, "at": 1000, "until": 2000, "tau": 500.0,   # picture, learned
               "strength": 0.5, "act_on": "picture"} | isc       # online from its own interpreting; used in its picture,
        assert isc["use"] in ("own", "none", "neighbor") and isc["twin"] in (None, "swap", "shift")   # or (section 90)
        assert isc["act_on"] in ("picture", "looking")           # in where it looks
        lam_i, kappa_i, i_shift = 1.0 - 1.0 / isc["tau"], None, None
        Zi, Ci = torch.zeros(B, 16, 16, device=DEV), torch.zeros(B, cfg["p"], 16, device=DEV)
        rec.update({k: np.full((T, B), np.nan, np.float32) for k in ("i_err", "i_err_other", "i_tot", "i_used", "i_moved")})
    e_term = None                                                # its expectation, as added to its picture this token

    def interp_used(I_s, t):
        """The model of its interpreting it uses: its own, or in the exchange window copies 0 and 1's swapped or
        changed at random by the size of their difference."""
        nonlocal i_shift
        if isc["twin"] is None or not (isc["at"] <= t < isc["until"]):
            return I_s
        if isc["twin"] == "swap":
            return I_s[perm_]
        if i_shift is None:
            gi_ = torch.Generator().manual_seed(8888)
            dn_ = (I_s[0] - I_s[1]).norm()
            i_shift = torch.zeros_like(I_s)
            for b_ in (0, 1):
                r_ = torch.randn(I_s.shape[1:], generator=gi_).to(DEV)
                i_shift[b_] = r_ * dn_ / r_.norm().clamp_min(1e-12)
        return I_s + i_shift

    tpic = ps.get("picture")                                     # (section 97) a transparent picture: the fixed leading
    if tpic is not None:                                         # directions of what the copies read, in place of its
        assert tpic["kind"] == "transparent" and snap is None and isc is None   # trained picture map
        tp_W, tp_mu, tp_sd = (torch.tensor(np.load(tpic[k_]), dtype=torch.float32, device=DEV) for k_ in ("W", "mu", "sd"))
        tp_k = int(tpic.get("k", 8))
        tp_a = 1.0 / float(tpic.get("slow", 1.0))               # (section 99) its stance follows that direction averaged
    tp_ema = None                                                # over `slow` tokens (1: not averaged)

    def transparent(att, hist):
        """(section 97) Its picture as the fixed leading directions of what it read, and its next stance from the
        leading k of them at equal scale."""
        v_t = f.rv(f.rln(hist)).view(hist.shape[0], hist.shape[1], f.rh, -1)
        u = (torch.einsum("nhl,nlhd->nhd", att, v_t).reshape(hist.shape[0], -1) - tp_mu) @ tp_W.T
        return squash(u, f.geometry), unit(u[:, :tp_k] / tp_sd[:tp_k])

    J_rec = []
    s_prev, g_prev = None, None
    uses = arm not in ("T", "Sx")
    with torch.no_grad():
        for t in range(T):
            hist = torch.cat([hist, H[:, t:t + 1]], 1)[:, -f.R:]
            q_add = None
            if sch is not None:                                  # (section 45) the window's content, and where its
                kk = f.rk(f.rln(hist)).view(B, hist.shape[1], f.rh, -1)   # schema expects its attention to pull
                z = unit(torch.einsum("ij,bj->bi", Pc, kk.mean(1).reshape(B, -1))) * 4.0
                A_s = Cs @ torch.linalg.inv(Zs + 1e-3 * eye16)
                A_u = A_s
                if Zn is not None:                               # (section 51) partly another's way of looking
                    A_u = (1.0 - sch["mix"]) * A_s + sch["mix"] * (Cn @ torch.linalg.inv(Zn + 1e-3 * eye16))
                in_win = sch["at"] <= t < sch["until"]
                if in_win and sch_shift is None:                 # a random change of the size of copies 0 and 1's
                    gs_ = torch.Generator().manual_seed(4646)    # difference (made in every life, used only in the
                    dn_ = (A_s[0] - A_s[1]).norm()               # shift twin; logged in all, for the comparison)
                    sch_shift = torch.zeros_like(A_s)
                    for b_ in (0, 1):
                        r_ = torch.randn(A_s.shape[1:], generator=gs_).to(DEV)
                        sch_shift[b_] = r_ * dn_ / r_.norm().clamp_min(1e-12)
                perm_ = torch.tensor([1, 0] + list(range(2, B)), device=DEV)
                if sch["twin"] is not None and in_win:
                    A_u = A_s[perm_] if sch["twin"] == "swap" else A_s + sch_shift   # copies 0 and 1 look with each
                                                                                       # other's schema, or a changed one
                Us_, Ss_, Vh_ = torch.linalg.svd(A_s, full_matrices=False)       # (section 49) its distinctions
                if t % 100 == 0:
                    rec["sch_sv"][t // 100] = Ss_.cpu().numpy()
                k_l = {"top": 0, "third": 2}.get(sch["lose"] or "", None)
                v_l = Vh_[:, k_l] if k_l is not None else lose_v              # the content direction it loses (or, for
                rec["eng"][t] = (z * v_l).sum(-1).abs().cpu().numpy()         # the random change, a random one)
                if sch["lose"] is not None and in_win:
                    if k_l is not None:                                       # windows differing only along it now look
                        A_u = A_u - Ss_[:, k_l, None, None] * Us_[:, :, k_l:k_l + 1] @ Vh_[:, k_l:k_l + 1, :]   # alike to it
                    else:                                                     # (control) the same size, no distinction
                        A_u = A_u - Ss_[:, 0, None, None] * lose_u[:, :, None] @ lose_v[:, None, :]
                pull_u = torch.einsum("bij,bj->bi", A_u, z)
                if sch["use"] != "none" and t >= sch["from"]:
                    if kappa is None:                            # its strength, set once: up to half a full habit, or
                        if sch.get("calib", "habit") == "habit":  # (section 56, no habit) rho times its own stance term
                            kappa = float(sch["strength"] * q_cap / pull_u.norm(dim=-1).mean().clamp_min(1e-12))
                        else:
                            ws_ = (s @ f.rq.weight.T).norm(dim=-1).mean()
                            kappa = float(sch["rho"] * ws_ / pull_u.norm(dim=-1).mean().clamp_min(1e-12))
                    q_add = (kappa * pull_u).view(B, f.rh, -1)
            look_on = (isc is not None and isc["act_on"] == "looking" and isc["use"] != "none" and t >= isc["from"]
                       and q_add is not None)
            if look_on:                                          # (section 90) what it expects to make of this content
                e_look = torch.einsum("bij,bj->bi", interp_used(Ci @ torch.linalg.inv(Zi + 1e-3 * eye16), t), z)
                P_, att_, k_, kappa_i, moved_ = read_q_look(f, s, hist, Qh, q_add, e_look, kappa_i, isc["strength"])
                rec["i_moved"][t] = moved_.cpu().numpy()         # leans where it looks
            elif q_add is not None:
                P_, att_, k_ = read_q(f, s, hist, Qh, q_add)
            elif Qh is None:
                P_ = f.read(s, hist)
                att_, k_ = f._last_att, None
            else:                                                # (section 41) its developed way of looking
                P_, att_, k_ = read_trace(f, s, hist, Qh)
            rec["pic"][t] = P_.cpu().numpy()
            if ps.get("log_read"):                               # (section 88, diagnostic) what it took, the window's
                v_r = f.rv(f.rln(hist)).view(B, hist.shape[1], f.rh, -1)   # average, and its schema's content summary
                for key_, x_ in (("rd", torch.einsum("nhl,nlhd->nhd", att_, v_r).reshape(B, -1)),
                                 ("rbar", v_r.mean(1).reshape(B, -1)), ("zs", z if sch is not None else None)):
                    if x_ is not None:
                        rec.setdefault(key_, np.zeros((T,) + tuple(x_.shape), np.float16))[t] = x_.cpu().numpy()
            if isc is not None:                                  # (section 88) what its way of looking made of its
                v_i = f.rv(f.rln(hist)).view(B, hist.shape[1], f.rh, -1)   # picture: its picture less the picture of
                dP = P_ - squash(f.to_pic(v_i.mean(1).reshape(B, -1)), f.geometry)   # the window read evenly
                I_s = Ci @ torch.linalg.inv(Zi + 1e-3 * eye16)
                e_u = torch.einsum("bij,bj->bi", interp_used(I_s, t), z)
                e_term = None
                if isc["act_on"] == "picture" and isc["use"] != "none" and t >= isc["from"]:
                    if kappa_i is None:                          # its strength, set once: on average half its own
                        kappa_i = float(isc["strength"] * dP.norm(dim=-1).mean() / e_u.norm(dim=-1).mean().clamp_min(1e-12))
                    e_term = kappa_i * e_u                       # picture contrast
                    rec["i_used"][t] = e_term.norm(dim=-1).cpu().numpy()
                rec["i_err"][t] = ((torch.einsum("bij,bj->bi", I_s, z) - dP) ** 2).sum(-1).cpu().numpy()
                rec["i_err_other"][t] = ((torch.einsum("bij,bj->bi", torch.roll(I_s, -1, 0), z) - dP) ** 2).sum(-1).cpu().numpy()
                rec["i_tot"][t] = (dP ** 2).sum(-1).cpu().numpy()
                zl_, dl_ = (torch.roll(z, -1, 0), torch.roll(dP, -1, 0)) if isc["use"] == "neighbor" else (z, dP)
                Zi = lam_i * Zi + torch.einsum("bi,bj->bij", zl_, zl_)
                Ci = lam_i * Ci + torch.einsum("bi,bj->bij", dl_, zl_)
            P_use, ix_ = snapped(P_ if e_term is None else squash(P_ + e_term, f.geometry), t)
            if ix_ is not None:
                rec["proto"][t] = ix_.cpu().numpy()
            if tpic is not None:                                 # (section 97) its picture: what it read, compressed
                P_, g = transparent(att_, hist)
                rec["pic"][t] = P_.cpu().numpy()
                if tp_a < 1.0:                                   # (section 99) averaged, then a direction again
                    tp_ema_old = g if tp_ema is None else tp_ema
                    tp_ema = (1.0 - tp_a) * tp_ema_old + tp_a * g
                    g = unit(tp_ema)
            else:
                g = f.next_stance(P_use)
            if sch is not None:                                  # (section 45) where its attention did pull; what its
                kp = (torch.einsum("nhl,nlhd->nhd", att_, kk) - kk.mean(1)).reshape(B, -1)   # schema knew of it; it learns
                pred = torch.einsum("bij,bj->bi", A_s, z)
                if keep:
                    kc_ = (kk - kk.mean(1, keepdim=True)).reshape(B, kk.shape[1], -1)
                    rec["exp_own"][t, :, -kc_.shape[1]:] = torch.einsum("bd,bld->bl", pred, kc_).cpu().numpy()
                    rec["exp_used"][t, :, -kc_.shape[1]:] = torch.einsum("bd,bld->bl", pull_u, kc_).cpu().numpy()
                rec["sch_err"][t] = ((pred - kp) ** 2).sum(-1).cpu().numpy()
                rec["sch_err_other"][t] = ((torch.einsum("bij,bj->bi", torch.roll(A_s, -1, 0), z) - kp) ** 2).sum(-1).cpu().numpy()
                rec["sch_tot"][t] = (kp ** 2).sum(-1).cpu().numpy()
                cosu = (pull_u * kp).sum(-1) / (pull_u.norm(dim=-1) * kp.norm(dim=-1)).clamp_min(1e-12)
                coso = (pred * kp).sum(-1) / (pred.norm(dim=-1) * kp.norm(dim=-1)).clamp_min(1e-12)
                rec["align_used"][t], rec["align_own"][t] = cosu.cpu().numpy(), coso.cpu().numpy()
                if in_win:                                       # how its pull lines up with the other's schema and with
                    for key_, Aw in (("align_swap", A_s[perm_]), ("align_shift", A_s + sch_shift)):   # the changed one
                        pw = torch.einsum("bij,bj->bi", Aw, z)
                        rec.setdefault(key_, np.full((T, B), np.nan, np.float32))[t] = (
                            (pw * kp).sum(-1) / (pw.norm(dim=-1) * kp.norm(dim=-1)).clamp_min(1e-12)).cpu().numpy()
                zl, kl = (torch.roll(z, -1, 0), torch.roll(kp, -1, 0)) if sch["use"] == "neighbor" else (z, kp)
                Zs = lam_s * Zs + torch.einsum("bi,bj->bij", zl, zl)
                Cs = lam_s * Cs + torch.einsum("bi,bj->bij", kl, zl)
                if Zn is not None:
                    zn_, kn_ = torch.roll(z, -1, 0), torch.roll(kp, -1, 0)
                    Zn = lam_s * Zn + torch.einsum("bi,bj->bij", zn_, zn_)
                    Cn = lam_s * Cn + torch.einsum("bi,bj->bij", kn_, zn_)
            rec["s"][t], rec["g"][t] = s.cpu().numpy(), g.cpu().numpy()
            A_ = att_.mean(1)                                    # how differently the copies attend, and what each
            if keep:
                rec["att"][t, :, -A_.shape[1]:] = A_.cpu().numpy()
            if B > 1:                                            # takes up (its lens, early and late)
                rec["att_apart"][t] = float((0.5 * (A_[:, None] - A_[None]).abs().sum(-1)).sum() / (B * (B - 1)))
            a_ = torch.einsum("bl,bld->bd", A_, hist)
            v_ = (a_ - a_.mean(0, keepdim=True)) / a_.pow(2).mean(-1, keepdim=True).sqrt()
            lens_seg[t // SEG] += v_ / SEG
            if warm + 16 <= t < warm + 716 or t >= T - 700:
                if t >= T - 700:
                    lens_l += v_ / 700.0
                else:
                    lens_e += v_ / 700.0
            if g_prev is None:                                   # the first token: it stays (as core32's turned loop)
                new_s = s
            else:
                if tpic is not None:                             # (section 97) through the same transparent picture
                    if q_add is not None:
                        att_cf = read_q(f, s_prev, hist, Qh, q_add)[1]
                    elif Qh is None:
                        f.read(s_prev, hist)
                        att_cf = f._last_att
                    else:
                        att_cf = read_trace(f, s_prev, hist, Qh)[1]
                    g_cf = transparent(att_cf, hist)[1]
                    if tp_a < 1.0:
                        g_cf = unit((1.0 - tp_a) * tp_ema_old + tp_a * g_cf)
                else:
                    P_cf = (read_q(f, s_prev, hist, Qh, q_add)[0] if q_add is not None else
                            f.read(s_prev, hist) if Qh is None else read_trace(f, s_prev, hist, Qh)[0])
                    if e_term is not None:                       # (section 88) with the same expectation
                        P_cf = squash(P_cf + e_term, f.geometry)
                    g_cf = f.next_stance(snapped(P_cf, t)[0])    # the exact
                sdg, ds = g - g_cf, s - s_prev                   # self-caused change (for analysis, and every
                rec["self_dg"][t] = sdg.cpu().numpy()            # `every`-th token its self-observation)
                J = C @ torch.linalg.inv(A + ps["ridge"] * eye)  # what it knows of itself so far
                sc_hist.append(float((sdg ** 2).sum(-1).mean()))
                if t_on < 0 and t >= trig["after"] and np.mean(sc_hist[-trig["span"]:]) > trig["level"]:
                    t_on = t                                     # (section 34) a chase is under way: the window opens
                Ul = torch.linalg.svd(J)[0]                      # the directions its knowledge says its own moves take
                rec["on_top"][t] = ((sdg * Ul[:, :, 0]).sum(-1) ** 2).cpu().numpy()
                rec["on_second"][t] = ((sdg * Ul[:, :, 1]).sum(-1) ** 2).cpu().numpy()
                if ab and t_on >= 0 and t < t_on + trig["window"]:
                    Uj, Sj, Vh = torch.linalg.svd(J)             # one distinction of its self-model removed
                    if ab == "top":
                        J = J - Sj[:, 0, None, None] * Uj[:, :, 0:1] @ Vh[:, 0:1, :]
                    elif ab == "second":
                        J = J - Sj[:, 1, None, None] * Uj[:, :, 1:2] @ Vh[:, 1:2, :]
                    else:                                        # (control) a random part of the top mode's size
                        J = J - Sj[:, 0, None, None] * ra[:, :, None] @ rb[:, None, :]
                if arm in ("Sswap", "Sshift") and ps["swap_at"] <= t < ps["back_at"]:
                    if arm == "Sswap":                           # copies 0 and 1 use each other's knowledge
                        J = J[torch.tensor([1, 0] + list(range(2, B)), device=DEV)]
                    else:
                        if shift is None:                        # a random change of the size of their difference
                            gs = torch.Generator().manual_seed(4545)
                            dn = (J[0] - J[1]).norm()
                            shift = torch.zeros_like(J)
                            for b_ in (0, 1):
                                r_ = torch.randn(m, m, generator=gs).to(DEV)
                                shift[b_] = r_ * dn / r_.norm().clamp_min(1e-12)
                        J = J + shift
                rec["J_norm"][t] = J.flatten(1).norm(dim=-1).cpu().numpy()
                pred = torch.einsum("bij,bj->bi", J, ds)         # the change its own move alone would have made
                rec["J_err"][t] = ((pred - sdg) ** 2).sum(-1).cpu().numpy()
                rec["J_err_other"][t] = ((torch.einsum("bij,bj->bi", torch.roll(J, -1, 0), ds) - sdg) ** 2).sum(-1).cpu().numpy()
                if t % ps["every"] == 0:                         # it learns from looking at itself, now and then
                    ds_l, sdg_l = (torch.roll(ds, -1, 0), torch.roll(sdg, -1, 0)) if arm == "Sn" else (ds, sdg)
                    A = lam * A + torch.einsum("bi,bj->bij", ds_l, ds_l)
                    C = lam * C + torch.einsum("bi,bj->bij", sdg_l, ds_l)
                if arm == "Sr":                                  # (control) the same size, no content
                    pred = torch.einsum("bij,bj->bi", Rpat * J.flatten(1).norm(dim=-1).view(B, 1, 1), ds)
                ref = unit(g_prev + ps["use"] * pred) if uses else g_prev   # the arms that use it turn only by what
                                                                 # the world brought ("use" < 1: only part of its own move is set aside, section 32)
                rec["ghat"][t] = ref.cpu().numpy()
                rec["turn"][t] = torch.arccos((ref * g).sum(-1).clamp(-1.0, 1.0)).cpu().numpy()
                new_s = carry(ref, g, s) if ps["gain"] == 1.0 else carry(ref, toward(ref, g, ps["gain"]), s)   # (36) slower
                if Qh is not None and hist.shape[1] == f.R:      # (section 41) its way of looking moves toward what was
                    kap = torch.einsum("nhl,nlhd->nhd", att_, k_) - k_.mean(1)   # particular in what it attended,
                    w_ = torch.einsum("nhd,nm->nhdm", kap, s)                   # for this stance
                    if q_rate is None:                           # set once, the same in every arm
                        q_rate = float(q_cap / (dv["fill"] * float(w_.flatten(1).norm(dim=-1).mean().clamp_min(1e-12))))
                    err = torch.tensor(rec["J_err"][t], device=DEV)
                    e_avg = err.mean() if e_avg is None else (1.0 - 1.0 / dv["avg"]) * e_avg + err.mean() / dv["avg"]
                    if dv["weight"] == "none":
                        wt = torch.ones(B, device=DEV)
                    elif dv["weight"] in ("own_rank", "neighbor_rank"):   # (section 43) smoothed: 0.5 + the rank of
                        err_win = (err_win + [err.cpu().numpy()])[-200:]  # this error among its last 200, so the
                        Wn = np.stack(err_win)                            # weight lies in [0.5, 1.5] and averages 1
                        rk = (Wn < Wn[-1][None]).mean(0) if len(Wn) > 1 else np.full(B, 0.5)
                        wt = torch.tensor(0.5 + rk, dtype=torch.float32, device=DEV)
                        if dv["weight"] == "neighbor_rank":
                            wt = torch.roll(wt, -1, 0)
                    else:                                        # most where its self-model did not foresee it
                        wt = err / e_avg.clamp_min(1e-12)
                        if dv["weight"] == "neighbor":           # (control) where its neighbour's did not
                            wt = torch.roll(wt, -1, 0)
                    Qh = Qh + q_rate * wt.view(B, 1, 1, 1) * w_
                    Qh = Qh * (q_cap / Qh.flatten(1).norm(dim=-1).clamp_min(1e-12)).clamp(max=1.0).view(B, 1, 1, 1)
                    rec["q_size"][t] = (Qh.flatten(1).norm(dim=-1) / f.rq.weight.norm()).cpu().numpy()
                    rec["q_weight"][t] = wt.cpu().numpy()
            s_prev, s, g_prev = s, new_s, g
            if t % 500 == 0 or t == T - 1:
                J_rec.append((C @ torch.linalg.inv(A + ps["ridge"] * eye)).cpu().numpy())
            if (t + 1) % 1000 == 0:
                print(f"[process-self {arm} s{cfg['init_seed']}] t={t + 1} {time.time() - t0:.0f}s "
                      f"J err {np.nanmean(rec['J_err'][t - 199:t + 1]):.5f} self-caused "
                      f"{np.mean((rec['self_dg'][t - 199:t + 1] ** 2).sum(-1)):.5f} turn {np.nanmean(rec['turn'][t - 199:t + 1]):.4f}",
                      flush=True)
    extra = dict(lens_early=lens_e.cpu().numpy(), lens_late=lens_l.cpu().numpy(), lens_seg=lens_seg.half().cpu().numpy())
    if Qh is not None:
        extra["Q"] = Qh.cpu().numpy()
    np.savez_compressed(os.path.join(out, "life.npz"), J=np.stack(J_rec), t_on=np.int32(t_on), **rec, **extra)
    open(os.path.join(out, "DONE"), "w").write(f"{time.time() - t0:.1f}\n")


def configs(root):
    """Section 30's lives: the 8B one-world lives of section 18 (second texts, seeds 10 and 11), one per arm."""
    names = []
    for r_ in ("c8", "d8"):
        for sd in (10, 11):
            base = json.load(open(f"D:/jspace_v11_runs/core32_turned8b/t8s2_{r_}_T_seed{sd}/run.json"))
            for arm in ARMS:
                name = f"ps_{r_}_{arm}_seed{sd}"
                cfg = dict(base, out=f"{root}/{name}", arm=arm, process_self={})
                os.makedirs(cfg["out"], exist_ok=True)
                json.dump(cfg, open(f"{cfg['out']}/run.json", "w"), indent=1)
                names.append(name)
    print("\n".join(names))


if __name__ == "__main__":
    if sys.argv[1] == "configs":
        configs(sys.argv[2])
    else:
        life(json.load(open(sys.argv[1])))
