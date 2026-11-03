"""research_v12 loop: the v11 configuration (A7m) plus switchable additions. Usage: python loop12.py config.json

00000000000000000. (v34) cfg["warmup"] = N with "same_text": one world after different pasts. With an empty model memory
   at the first token neither the steering nor the reader's query can act (one position to attend to), so copies that
   read one text from the start are identical from there on. Each copy now reads its own stream at tokens 4..N-1 and
   world 0's after; the first four tokens (kept for good by the crop as attention sinks) are world 0's for every copy.
0000000000000000. (v33) cfg["closure"] = "loss_describe": closure through the loss as "loss", but the description is graded
   only by how well it matches its field's shape (1 - correlation per copy, scale-free), with no telling copies apart
   (the fast core's finding: all of the loop's self-dependence comes from that copy contest).
000000000000000. (v32, closure through use; DESIGN_CLOSURE_THROUGH_USE.md) cfg["closure"] = "loss" | "measured" | "use"
   (identity "dynamics", the plain loop). "loss" (default) is closure as before: the field the picture describes is
   computed with gradient, so the game's error also reshapes the reading and decoding it describes. "measured": the
   field is measured (no gradient) and the description is graded only for being right (plain regression of the
   centred field, relative to its size per copy; copy identification is still logged); the loop moves by its reading.
   "use": as "measured", and from token cfg["use_from"] (default the calibration length) the loop moves by its own
   description: its next stance is the destinations' mean over the probe stances (the part the text sets) plus what its
   description says its own part is at its own stance, instead of the reader's actual own part; usefulness keeps
   training the reading and decoding through it (straight-through). With an exact description "use" changes nothing.
   The regression is relative to a running average of the field's size per copy (decay 0.99), so a momentary dip of
   the field does not explode it; its weight is calibrated on tokens adopt..calib. A head taken over from an archived
   life was trained by the contrastive game, which ignores scale: over its first cfg["adopt_tokens"] (50) tokens, before
   any learning, one least-squares scale between its centred predictions and the measured field is fitted and its last
   layer multiplied by it once ("head_scale" in config.json).
   "use_noise" (control): as "use", but the deviation of the used stance from the reader's own next stance is replaced by
   a vector of the same size in a random direction (the same push, without the description's content).
   cfg["fixed_probes"] = True: the random probe stances are drawn once per life instead of at every token, so the part
   the text sets (their destinations' mean) is a steady reference and not resampled noise.
   Saved in life.npz: "desc_err" (relative error), "own_desc" / "own_meas" (size of the described / the reader's
   actual own part at its own stance), "use_gap" (|described - actual| / |actual|), "use_dev" (|used - reader's|).

00000000000000. (v29, a self-description over the horizon of its world; DESIGN_HORIZON.md) cfg["field_horizon"] =
   {"k": 32, "every": 16, "random": 4, "source": "own"|"neighbor"} (needs the spotlight, identity "dynamics"): every
   `every` tokens, the loop's counterfactual futures (horizon_rollout: for `random` random probe stances and the copies'
   own, each copy run on for k tokens as if it stood there, then living on as it does, with the same sampling noise for
   every probe of a copy; no gradient) give its field at the horizon: where its stance is k tokens later, centred over
   the stances. The picture describes it with m more outputs of the field head (f.predict_cons) in the same contrastive
   game (identity by how it moves), with the one-step game's calibrated weight. "neighbor" (control): copy c describes copy
   c+1's futures. Saved in life.npz: "hk_acc" (how often the horizon field identifies the copy; NaN between rollouts).
0000000000000. (v28, a world partly of its own making; DESIGN_SELF_MADE_WORLD.md) cfg["self_written"] = True or
   {"prompt": 48, "write": 208, "temperature": 1.0, "top_p": 0.9}: each copy reads stretches of its own real text
   stream (prompts) and, after each, text the model writes itself, token by token, sampled from its prediction under the
   copy's live spotlight (whatever channel the life has: own, neighbour's, or none with spotlight 0); the real stream
   resumes where it was left. The prediction loss counts only real tokens predicting real tokens. Saved in life.npz:
   "tok" (the token each copy read at each step) and "written" (whether that step's token was written by the model).
000000000000. (v26, a self-description that covers what its attending does; DESIGN_SELF_CONSEQUENCES.md)
   cfg["field_consequence"] = k (needs the spotlight and identity "dynamics", no item): every row of the field the
   picture describes (one per probe stance) gets a second part: what attending from that stance does to the model's
   processing of the current token (the read-layer state under the stance's spotlight minus the state with no
   spotlight, from one extra pass that writes nothing), projected to k numbers by a fixed random projection, centred
   over the probe stances and normalised per token to a median row size of 1 (the dynamics part likewise), predicted by
   k more outputs of the field head (starting at zero). The reader receives nothing new. cfg["field_consequence_from"]
   = "neighbor" (control): copy c describes copy c+1's consequence part. With describe "another" the twin record must
   hold "C" (record_twin saves it). Saved in life.npz: "cf_size" (|live consequence| / |state without|), "cf_scale"
   (the raw typical size of the consequence part), "cf_acc" / "df_acc" (how often the consequence part / the dynamics
   part alone identifies the copy). set_field_consequence() and load_interface() serve the frozen tools;
   f.predict_field stays the dynamics part, f.predict_cons gives the consequence part.
   (v27, a simplified model of its own attention; DESIGN_ATTENTION_SCHEMA.md) cfg["field_consequence_kind"] = "act"
   with field_consequence = 2: the second part of each row is not the detailed consequence but two coarse features
   that move on the loop's timescale: how sharply the reader would attend from that stance (the entropy of its
   attention over its 16 states, which is also the spotlight's shape) and how much attending from there would change
   the model's processing of the current token (|state under its spotlight - state with no spotlight| / |state with
   no spotlight|); each centred over the stances and scaled to a typical size of 1 before the part is normalised as
   in v26. "cf_scale" then records the raw typical spread of the second feature.
00000000000. (v25, the consequence item; DESIGN_CLOSURE.md) cfg["consequence"] = "own" | "neighbor" (needs the
   spotlight): at every token the current token is also processed from the band up to the read layer with no
   spotlight (the loop attending evenly), without writing to the model's memory; the difference between the two
   read-layer states is the consequence of the loop's attending at this token. The reader takes it in as one more item
   beside its 16 states (the same normalization, keys and values; its own position bias f.cons_pos, starting at the
   mean of the window's). Probe stances get the consequence of their own spotlight. "neighbor" (control): each copy
   reads its neighbour copy's consequence. Saved in life.npz: "cons_att" (the reader's attention on the item),
   "cons_size" (|consequence| / |state without|). load_interface() and the cons_* helpers serve the frozen tools.
0000000000. (v24, the shared spotlight; DESIGN_CONSEQUENCES.md) cfg["spotlight"] = kappa: the loop acts on the model
   through its reading. At each token the reader's attention over the previous 16 read-layer states, with the query set
   by the current stance (its heads averaged: the spotlight p), also biases the band layers' attention scores for the
   same 16 positions by kappa * clamp(log(16 p), -log 16, +log 16): a position's weight in the model's attention is
   multiplied or divided by at most 16^kappa, nothing else changes, the model's weights never change. The old stance
   query shift is off unless cfg["spotlight_keep_shift"]. Probe stances set their own spotlight for the probe's
   processing, so the field includes what the loop's attending does to the model. cfg["spotlight_from"] = "neighbor"
   (control): each copy's model is focused by its neighbour copy's reading. The live spotlight is saved as life.npz
   "spot". channel(f, cfg) gives the same channel to the frozen analyses.
000000000. (v22, identity by how it moves; cfg["init_iface"] = a saved interface to start from) cfg["identity"] = "dynamics": the self-description game compares not the
   copies' fields of moves but only the part that depends on where the system is: from every field the copy's own
   mean destination over the probe stances is removed (destinations D(S) + S minus their mean over S), in the
   copies' measured fields and in the picture's predicted field alike. Naming a position no longer tells a copy apart;
   only its own way of moving from different places can. Default "position" is the game as before.
00000000. (step B, SYNTHESIS_18OCT) how the current picture reaches the next one (sphere picture only),
   cfg["reader_route"]: "operation" (default: the stance sets the reader's query, i.e. how the next picture is read),
   "content" (the reader's query has no stance part; instead the copy's last picture is given to the reader as one more
   item beside the 16 read-layer states: ordinary recurrence, the same information without changing the operation),
   "none" (neither). The stance still sets the model's attention in every case. cfg["describe"] = "another" with
   cfg["twin"] = a directory holding twin_record.npz: the picture must describe the recorded field of another life
   (same texts, its probe stances), as the v11 "another" arm. cfg["record_twin"] = True saves this life's probe stances,
   fields and stances to twin_record.npz.
0000000. (v18) cfg["picture_map"] = "linear": the map from the reader's output to the picture is one linear layer
   instead of a two-layer network, so it cannot fold what is read into a few labels; the picture stays a graded
   projection of what the reader takes in (the reader itself, the game and every route are unchanged). In every
   earlier life a 5% change of what is read moved the picture by about 0.1% or less (DEEP_PHASE_v16).
000000. (v17) cfg["lasting"] = True (sphere picture only): the picture keeps what happens to it. Each new picture
   starts from the last one: P(t) = sqrt(p) * normalize(lam * P(t-1) + (1 - lam) * P_read(t)), P_read the picture read
   from the window as before, lam = sigmoid(lam_raw) one number learned by the loop itself (from 0.5), so the loop
   chooses its own persistence. Probe pictures start from the copy's last picture too. In every earlier life a push
   to the picture or to its input was erased within one token (DEEP_PHASE_v16).
00000. (v16) cfg["picture"] = "sphere": the picture is a point on a sphere of radius sqrt(p) (the to_pic output
   normalized) instead of tanh(to_pic output), a point in the cube [-1, 1]^p. In every life up to v15 the cube picture
   saturated (about 95% of its values at +-1 after the first quarter): a binary code word whose bit flips carried most
   of the stance's motion, and through which almost no gradient passes. Default "cube" is the v11 picture.
0000. (v15) cfg["body"]["mode"] = "stance": the fluctuation moves where the copy's stance actually lands instead of
   adding to its queries in fixed random directions: the band offsets are taken at normalize(s + size * A xi_c(t)),
   A a fixed random (m x k) map. A stance can therefore undo it, which the v13 fluctuation never allowed. The reader's
   query keeps the stance the picture chose. Default "query" is the v13 fluctuation.
000. (v15) cfg["self_forecast"] = {"dim": 16, "target": "self"}: at every token the picture also forecasts how its own
   attending will change its own next processing: the copy's next read-layer state minus the untouched model's on the
   same text, through a fixed random projection (scaled from tokens 10-29), predicted by a small head from the picture
   AND the stance actually applied next (given to every arm, so the own picture gains nothing from knowing its own
   choice). The target is measured and detached (no closure). Weight by the same calibration rule as the other
   pressures. "target": "text" is the control: forecast the untouched model's next read-layer state (the text's own
   course, nothing about oneself) with the same head and scale rule.
00. (v14) cfg["field"] = "reading": what the picture must describe and tell apart from its copies is not how the
   stance would move from each probe but how the copy's own reading would respond there (a fixed random projection of
   the read-layer state under that probe stance, scaled once at calibration). Targets come from the frozen model, so
   there is no closure: the picture can describe them, not shape them. With "local" divergence the pull is toward the
   copy's own probe where its reading would change most relative to its reading at its current stance.
0. (v13) A body with its own fluctuations, cfg["body"] = {"size", "k", "tau"}: each copy's band queries receive
   size * tanh(M_l xi_c(t)), xi_c a smooth random process (k numbers, correlation time tau tokens, its own per copy),
   M_l a fixed random map (the same body plan for every life). cfg["body_arm"]: "own" (as always: the stance comes
   from the copy's own picture), "blind" (the reader reads the untouched model's activity: the picture sees the text,
   not the copy's own attending), "swap" (each copy's stance comes from the next copy's picture).

1. Self-negation variants (PLAN v12 section 4), cfg["div"]:
   "none"            no divergence;
   "shared"          v11: pull toward the shared probe where this copy's measured move is largest;
   "local"           each copy draws its own probe stances and is pulled toward the one where it measurably moves most;
   "local_picture"   pulled toward where the copy's own picture predicts it would move most (64 fixed candidates).
2. Per-copy slow attention weights (PLAN v12 section 3), cfg["plastic"]:
   "none" | "ungated" | "own" | "other". Each copy c has a rank-r change U_c V_c^T to the query projection of every band
   layer, starting at zero (all copies identical at birth). After every chunk it takes a step on its own next-token loss
   only (Adam direction, small learning rate), scaled per rank component by a gate. "ungated": gate 0.5. "own": gate =
   sigmoid(G_l * picture_c + b_l), the copy's own mean picture over the chunk. "other": the same map fed with copy
   (c+1)'s picture. The map G is learned from usefulness: a first-order hypergradient (did opening this component at the
   last chunk help this chunk's reading?). The slow change is capped (||U||_F <= cap per copy and layer).
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "C:/Users/assas/PycharmProjects/jspace/research_v11/llm")
sys.path.insert(0, "/workspace/code")
from llm_loop import Interface, Runner, contrastive, make_streams  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv  # noqa: E402


class Runner12(Runner):
    """Runner whose band layers can add a per-row content-dependent query change (the copy's slow weights)."""

    def layer(self, l, h, Kp, Vp, pos, q_off=None, ad=None, sb=None, take=None):
        """take (WHERE_ITS_OWN_STATE_GOES.md section 22): (share, p (N, Rp)), a share of the attention replaced by the
        distribution p over the last Rp positions before this token (what the model takes up)."""
        lay = self.layers[l]
        a = lay.self_attn
        N = h.shape[0]
        x = lay.input_layernorm(h)
        q = a.q_proj(x)
        if q_off is not None:
            q = q + q_off.to(q.dtype)
        if ad is not None:                     # ad(l, x) -> (N, 1, qdim) or None
            dq = ad(l, x)
            if dq is not None:
                q = q + dq.to(q.dtype)
        q = a.q_norm(q.view(N, 1, self.nh, self.hd)).transpose(1, 2)
        k = a.k_norm(a.k_proj(x).view(N, 1, self.kvh, self.hd)).transpose(1, 2)
        v = a.v_proj(x).view(N, 1, self.kvh, self.hd).transpose(1, 2)
        pid = torch.full((N, 1), pos, device=self.dev, dtype=torch.long)
        cos, sin = self.model.model.rotary_emb(x, pid)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        K = torch.cat([Kp, k], 2)
        V = torch.cat([Vp, v], 2)
        rep = self.nh // self.kvh
        if sb is not None:                     # (v24) the shared spotlight: added to the scores of the last Rp positions;
            Lk, Rp = K.shape[2], sb.shape[1]  # explicit float32 attention, as the fused kernel cannot backpropagate this bias
            z = torch.zeros(N, Lk - 1 - Rp, device=sb.device, dtype=torch.float32)
            bias = torch.cat([z, sb.float(), torch.zeros(N, 1, device=sb.device)], 1).view(N, 1, 1, Lk)
            Kr, Vr = repeat_kv(K, rep), repeat_kv(V, rep)
            w = torch.softmax((q.float() @ Kr.float().transpose(-1, -2)) * a.scaling + bias, -1)
            o = (w @ Vr.float()).to(q.dtype)
        elif take is not None:
            share, p_ = take
            Lk, Rp = K.shape[2], p_.shape[1]
            P = torch.cat([torch.zeros(N, Lk - 1 - Rp, device=p_.device), p_.float(), torch.zeros(N, 1, device=p_.device)], 1)
            Kr, Vr = repeat_kv(K, rep), repeat_kv(V, rep)
            w = torch.softmax((q.float() @ Kr.float().transpose(-1, -2)) * a.scaling, -1)
            w = (1.0 - share) * w + share * P.view(N, 1, 1, Lk)
            o = (w @ Vr.float()).to(q.dtype)
        else:
            o = F.scaled_dot_product_attention(q, repeat_kv(K, rep), repeat_kv(V, rep), scale=a.scaling)
        h = h + a.o_proj(o.transpose(1, 2).reshape(N, 1, -1))
        h = h + lay.mlp(lay.post_attention_layernorm(h))
        return h, k, v

    def upper_probe(self, st, h_in, offs_fn, S, b_idx, ad=None, sb=None):
        h = h_in[b_idx]
        for l in range(self.band[0], self.read_layer + 1):
            off = offs_fn(l, S) if l in self.band else None
            h, _, _ = self.layer(l, h, st["K"][l][b_idx], st["V"][l][b_idx], st["pos"], off, ad if l in self.band else None,
                                 sb if l in self.band else None)
        return h

    def upper_live(self, st, h_in, offs_fn, s, ad=None, sb=None, take=None, act=None):
        """act: the layers where sb or take act (default: the band)."""
        h = h_in
        h_read = None
        act = self.band if act is None else act
        for l in range(self.band[0], self.L):
            off = offs_fn(l, s) if l in self.band else None
            h, k, v = self.layer(l, h, st["K"][l], st["V"][l], st["pos"], off, ad if l in self.band else None,
                                 sb if l in act else None, take if l in act else None)
            st["K"][l] = self.crop(torch.cat([st["K"][l], k], 2))
            st["V"][l] = self.crop(torch.cat([st["V"][l], v], 2))
            if l == self.read_layer:
                h_read = h
        logits = self.model.lm_head(self.model.model.norm(h))
        st["pos"] += 1
        return h_read, logits


def set_consequence(f):
    """(v25) The consequence item's own position bias, starting at the mean of the reader's window position biases."""
    if not hasattr(f, "cons_pos"):
        f.cons_pos = nn.Parameter(f.rpos.detach().mean().clone())
    return f


def set_field_consequence(f, k):
    """(v26) Give the field head k more outputs (the consequence part, starting at zero); f.predict_field keeps returning
    the dynamics part, f.predict_cons returns the consequence part. Idempotent."""
    lin = f.field[-1]
    if lin.out_features == f.m:
        new = nn.Linear(lin.in_features, f.m + k).to(lin.weight.device)
        with torch.no_grad():
            new.weight.zero_()
            new.bias.zero_()
            new.weight[:f.m] = lin.weight
            new.bias[:f.m] = lin.bias
        f.field[len(f.field) - 1] = new
    assert f.field[-1].out_features == f.m + k, "a field head with another consequence size"
    f.k_cons = k

    def head(P, S):
        if P.shape[0] == 1 and S.shape[0] > 1:
            P = P.expand(S.shape[0], -1)
        return f.field(torch.cat([P, S], -1))
    f.predict_field = lambda P, S: head(P, S)[..., :f.m]
    f.predict_cons = lambda P, S: head(P, S)[..., f.m:]
    return f


def cons_projection(R, k):
    """(v26) The fixed random projection of read-layer states the consequence part is described in."""
    return (torch.randn(R.d, k, generator=torch.Generator().manual_seed(2026)) / math.sqrt(R.d)).to(R.dev)


def act_features(f, kappa, SS, hist_rows, h_probe, h_even_rows):
    """(v27) Coarse features of the act from each probe row: the entropy of the reader's attention over its window from
    that stance, and the relative size of what attending from there does to the read-layer state. (N, 2)."""
    p_ = spotlight(f, SS, hist_rows, kappa)[1]
    if p_ is None:                             # no window yet (the first tokens): nothing to attend to
        ent = torch.zeros(SS.shape[0], device=SS.device)
    else:
        ent = -(p_.clamp_min(1e-9).log() * p_).sum(-1)
    eff = (h_probe - h_even_rows).norm(dim=-1) / h_even_rows.norm(dim=-1).clamp_min(1e-9)
    return torch.stack([ent, eff], -1)


def act_part(feats, B):
    """(v27) (N, 2) -> (B, rows, 2): each feature centred over the stances and scaled to a typical size of 1."""
    F_ = feats.view(B, -1, feats.shape[-1])
    F_ = F_ - F_.mean(1, keepdim=True)
    return F_ / F_.abs().flatten(0, 1).median(0).values.clamp_min(1e-12)


def written_regime(cfg):
    """(v28) The writing regime of a life (None: fixed text)."""
    sw = cfg.get("self_written")
    if not sw:
        return None
    return dict(prompt=48, write=208, temperature=1.0, top_p=0.9) | (sw if isinstance(sw, dict) else {})


def sample_shared(logits, sw, u):
    """(v29) Sample from the nucleus with given uniforms (common random numbers): rows that share u and have the same
    prediction draw the same token."""
    pr_ = torch.softmax(logits.float() / sw["temperature"], -1)
    srt, idx = pr_.sort(-1, descending=True)
    srt = srt * ((srt.cumsum(-1) - srt) < sw["top_p"])
    c_ = srt.cumsum(-1)
    j = (c_ < u[:, None] * c_[:, -1:]).sum(-1).clamp_max(c_.shape[-1] - 1)
    return idx.gather(-1, j[:, None])[:, 0]


def horizon_rollout(R, f, kappa, offs, st, b_idx, q0, sp0, cur, ptr, t, k, ids, sw, u):
    """(v29) The loop's counterfactual futures, without gradient. Rows (N) are (copy, probe) pairs: each starts from its
    copy's current state (model memory, window, input token, writing position) and runs k tokens. At the first token
    the reader's query is set by q0 and the spotlight by sp0 (N, m); afterwards each row lives on with its own stance:
    real tokens from the copy's stream, written tokens sampled with the uniforms u (k, copies) shared by the rows of a
    copy, so they differ only through where they stood. Returns the rows' stances after k tokens (N, m)."""
    stR = {"K": [x[b_idx] for x in st["K"]], "V": [x[b_idx] for x in st["V"]], "pos": st["pos"], "hist": st["hist"][b_idx]}
    curR, ptrR = cur[b_idx], ptr[b_idx]
    rowsR = torch.arange(b_idx.shape[0], device=b_idx.device)
    per = (sw["prompt"] + sw["write"]) if sw else None
    sR = None
    for h in range(k):
        h_in = R.lower(stR, curR)
        q, sp = (q0, sp0) if h == 0 else (sR, sR)
        sb = spotlight(f, sp, stR["hist"], kappa)[0] if kappa else None
        hr, logits = R.upper_live(stR, h_in, offs, sp, sb=sb)
        stR["hist"] = torch.cat([stR["hist"], hr.float()], 1)[:, -f.R:]
        sR = f.next_stance(f.read(q, stR["hist"]))
        if h == k - 1:
            break
        if sw is None or ((t + h + 1) % per) < sw["prompt"]:
            curR = ids[b_idx, ptrR.clamp_max(ids.shape[1] - 1)]
            ptrR = ptrR + 1
        else:
            curR = sample_shared(logits[:, 0], sw, u[h][b_idx])
    del rowsR
    return sR


def load_interface(f, path, cfg, dev):
    """Load a saved interface (dropping the lasting-picture weight); the consequence bias is created before loading
    when the file has one, and after loading (at the mean window bias) when the life asks for the item."""
    sd = torch.load(path, map_location=dev)
    sd.pop("lam_raw", None)
    if "cons_pos" in sd:
        set_consequence(f)
    extra = sd["field.2.weight"].shape[0] - f.m if "field.2.weight" in sd else 0
    if extra > 0:                              # (v26) a field head that also describes the consequences
        set_field_consequence(f, extra)
    f.load_state_dict(sd)
    if cfg.get("consequence"):
        set_consequence(f)
    if cfg.get("field_consequence"):
        set_field_consequence(f, int(cfg["field_consequence"]))
    return f


def cons_baseline(R, f, cfg, st, h_in, offs_fn, s, spot_live):
    """(v25) Before the live pass: the current token's read-layer state with no spotlight (and, for the neighbour
    control, the live state under the spotlight), without writing to the model's memory. None without the item."""
    mode = cfg.get("consequence")
    if not mode:
        return None
    rows = torch.arange(h_in.shape[0], device=h_in.device)
    with torch.no_grad():
        base = {"mode": mode, "h0": R.upper_probe(st, h_in, offs_fn, s, rows, sb=None).float()[:, 0]}
        if mode == "neighbor":
            h_pre = R.upper_probe(st, h_in, offs_fn, s, rows, sb=spot_live(s, st["hist"])[0]).float()[:, 0]
            base["nb"] = torch.roll(h_pre - base["h0"], -1, 0)
    return base


def cons_probe(base, hr_probe, b_idx):
    """(v25) The consequence item each probe row reads: its own spotlight's difference, or the neighbour's live one."""
    if base is None:
        return None
    if base["mode"] == "neighbor":
        return base["nb"][b_idx]
    return hr_probe[:, 0].float() - base["h0"][b_idx]


def cons_live(base, h_read):
    """(v25) The consequence item the live read takes in (with the gradient through the live state)."""
    if base is None:
        return None
    c = h_read[:, 0].float() - base["h0"]
    return torch.roll(c, -1, 0) if base["mode"] == "neighbor" else c


def spotlight(f, s, hist, kappa):
    """(v24) The reader's attention over the window hist (N, R', d) with the query set by s (N, m), its heads averaged:
    p (N, R'); and the bias it puts on the model's attention scores for those positions,
    kappa * clamp(log(R' p), -log R', +log R') (bounded both ways, so p -> 0 cannot push a score to minus infinity)."""
    N, Rp, _ = hist.shape
    if Rp < 2:
        return None, None
    k = f.rk(f.rln(hist)).view(N, Rp, f.rh, -1)
    q = (f.rq_bias.expand(N, -1) + f.rq(s)).view(N, f.rh, -1)
    p = torch.softmax(torch.einsum("nhd,nlhd->nhl", q, k) / math.sqrt(q.shape[-1]) + f.rpos[-Rp:][None, None], -1).mean(1)
    if kappa == 0:                             # strength 0: no bias at all (the loop is exactly the v22 loop)
        return None, p
    lim = math.log(Rp)
    return kappa * torch.clamp(torch.log(p * Rp + 1e-12), -lim, lim), p


def channel(f, cfg):
    """(v24) How the loop acts on the model, for the loop and the frozen analyses: (offs_fn, live(s, hist) -> (bias, p),
    probe(SS, hist_rows, s, hist, b_idx) -> bias). Without cfg["spotlight"] it is the old query shift and no bias."""
    kappa = cfg.get("spotlight")
    if kappa is None:
        return f.offs, (lambda s_, h_: (None, None)), (lambda SS, hr, s_, h_, b: None)
    offs = f.offs if cfg.get("spotlight_keep_shift") else (lambda l, x: None)
    nb = cfg.get("spotlight_from", "own") == "neighbor"

    def live(s_, h_):
        sb, p = spotlight(f, s_, h_, kappa)
        return (torch.roll(sb, -1, 0) if (nb and sb is not None) else sb), p

    def probe(SS, hr, s_, h_, b):
        if nb:                                   # the neighbour's reading focuses this copy's model, whatever its stance
            sb = live(s_, h_)[0]
            return None if sb is None else sb[b]
        return spotlight(f, SS, hr, kappa)[0]
    return offs, live, probe


def squash(z, geometry="cube"):
    """The picture's geometry: the cube [-1, 1]^p (v11, tanh) or a sphere of radius sqrt(p) (v16)."""
    if geometry == "sphere":
        return math.sqrt(z.shape[-1]) * z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.tanh(z)


def lasting_combine(P_read, P_prev, lam):
    """(v17) The lasting picture on the sphere: start from the last picture, moved by what is read."""
    if P_prev is None:
        return P_read
    z = lam * P_prev + (1 - lam) * P_read
    return math.sqrt(P_read.shape[-1]) * z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def set_picture_map(f, kind):
    """(v18) Call before loading a saved interface: 'linear' replaces the two-layer reader-to-picture map."""
    if kind == "linear":
        f.to_pic = nn.Linear(f.to_pic[0].in_features, f.to_pic[-1].out_features).to(f.rq_bias.device)
    return f


def set_picture_geometry(f, geometry, route="operation"):
    """Give an Interface (llm_loop) the picture geometry; its read() is the v11 read with the chosen squashing.
    route (step B): "operation" (the stance sets the reader's query), "content" (no stance in the query; read(...,
    item=last picture) adds the copy's last picture as one more item to read), "none". Call before building the
    optimizer (and before loading a saved "content" interface): "content" adds the item's map and position bias."""
    f.geometry = geometry or "cube"
    f.route = route or "operation"
    if f.geometry == "cube":
        assert f.route == "operation", "the reader routes are built on the sphere picture"
        return f
    if f.route == "content":
        f.pic_in = nn.Linear(f.p, f.rln.normalized_shape[0]).to(f.rq_bias.device)
        f.item_pos = nn.Parameter(torch.zeros((), device=f.rq_bias.device))

    def read(s, hist, item=None, cons=None):
        N, Rp, _ = hist.shape
        a = f.rln(hist)
        k = f.rk(a).view(N, Rp, f.rh, -1)
        v = f.rv(a).view(N, Rp, f.rh, -1)
        stance_q = f.rq(s) if (f.rq is not None and f.route == "operation") else 0.0
        q = (f.rq_bias.expand(N, -1) + stance_q).view(N, f.rh, -1)
        logit = torch.einsum("nhd,nlhd->nhl", q, k) / math.sqrt(q.shape[-1]) + f.rpos[-Rp:][None, None]
        if f.route == "content" and item is not None:
            ai = f.rln(f.pic_in(item))
            ki, vi = f.rk(ai).view(N, 1, f.rh, -1), f.rv(ai).view(N, 1, f.rh, -1)
            k, v = torch.cat([k, ki], 1), torch.cat([v, vi], 1)
            logit = torch.cat([logit, torch.einsum("nhd,nlhd->nhl", q, ki) / math.sqrt(q.shape[-1]) + f.item_pos], -1)
        if cons is not None:                   # (v25) the consequence of its attending, read like any state
            ac = f.rln(cons.unsqueeze(1))
            kc, vc = f.rk(ac).view(N, 1, f.rh, -1), f.rv(ac).view(N, 1, f.rh, -1)
            k, v = torch.cat([k, kc], 1), torch.cat([v, vc], 1)
            logit = torch.cat([logit, torch.einsum("nhd,nlhd->nhl", q, kc) / math.sqrt(q.shape[-1]) + f.cons_pos], -1)
        att = torch.softmax(logit, -1)
        f._last_att = att.detach()
        r = torch.einsum("nhl,nlhd->nhd", att, v).reshape(N, -1)
        return squash(f.to_pic(r), f.geometry)
    f.read = read
    return f


class Slow(nn.Module):
    """Per-copy slow query weights for the band: U (B, L, qdim, r) starts at zero, V (B, L, d, r) random, identical
    for every copy. The gate map G (L, r, p) and bias (L, r) are shared."""

    def __init__(self, R, B, r, p, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        L = len(R.band)
        V0 = torch.randn(1, L, R.d, r, generator=g) / math.sqrt(R.d)
        self.U = nn.Parameter(torch.zeros(B, L, R.qdim, r))
        self.V = nn.Parameter(V0.repeat(B, 1, 1, 1))
        self.G = nn.Parameter(torch.zeros(L, r, p))
        self.gb = nn.Parameter(torch.zeros(L, r))
        self.li = {l: i for i, l in enumerate(R.band)}

    def fn(self, rows):
        """rows: (N,) copy index of each row -> function (l, x) -> (N, 1, qdim)."""
        def ad(l, x):
            i = self.li[l]
            Vr, Ur = self.V[rows, i], self.U[rows, i]               # (N, d, r), (N, qdim, r)
            return torch.einsum("nqr,nr->nq", Ur, torch.einsum("nd,ndr->nr", x[:, 0].float(), Vr)).unsqueeze(1)
        return ad

    def gates(self, Pbar):
        """Pbar: (B, p) mean picture per copy -> (B, L, r) gates."""
        return torch.sigmoid(torch.einsum("lrp,bp->blr", self.G, Pbar) + self.gb[None])


class SelfSlow(nn.Module):
    """Per-copy slow changes to the self-model machinery (v12b): the writers (how the stance sets each band layer's query
    offsets; full change Ww) and the reader (query map Rq, full; key and value maps, rank-r changes Uk Vk^T, Uv Vv^T).
    All start at zero (every copy identical at birth); V parts random, identical for every copy. Gates per copy for the
    two parts (writers, reader) come from a picture: sigmoid(G P + gb)."""

    def __init__(self, R, f, B, r, p, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        L, d, m, rd = len(R.band), R.d, f.m, f.rk.out_features
        self.Ww = nn.Parameter(torch.zeros(B, L, R.qdim, m))
        self.Rq = nn.Parameter(torch.zeros(B, rd, m))
        self.Uk = nn.Parameter(torch.zeros(B, rd, r))
        self.Uv = nn.Parameter(torch.zeros(B, rd, r))
        self.Vk = nn.Parameter((torch.randn(1, d, r, generator=g) / math.sqrt(d)).repeat(B, 1, 1))
        self.Vv = nn.Parameter((torch.randn(1, d, r, generator=g) / math.sqrt(d)).repeat(B, 1, 1))
        self.G = nn.Parameter(torch.zeros(2, p))
        self.gb = nn.Parameter(torch.zeros(2))
        self.li = {l: i for i, l in enumerate(R.band)}
        self.groups = [("Ww", 0), ("Rq", 1), ("Uk", 1), ("Uv", 1), ("Vk", 1), ("Vv", 1)]

    def offs_fn(self, f, rows):
        def offs(l, S):
            i = self.li[l]
            z = f.writers[str(l)](S) + torch.einsum("nqm,nm->nq", self.Ww[rows, i], S)
            return (f.bound * torch.tanh(z)).unsqueeze(1)
        return offs

    def read(self, f, s, hist, rows):
        N, Rp, _ = hist.shape
        a = f.rln(hist)
        k = f.rk(a) + torch.einsum("nlr,nqr->nlq", torch.einsum("nld,ndr->nlr", a, self.Vk[rows]), self.Uk[rows])
        v = f.rv(a) + torch.einsum("nlr,nqr->nlq", torch.einsum("nld,ndr->nlr", a, self.Vv[rows]), self.Uv[rows])
        k, v = k.view(N, Rp, f.rh, -1), v.view(N, Rp, f.rh, -1)
        q = f.rq_bias.expand(N, -1) + (f.rq(s) if f.rq is not None else 0.0) + torch.einsum("nqm,nm->nq", self.Rq[rows], s)
        q = q.view(N, f.rh, -1)
        att = torch.softmax(torch.einsum("nhd,nlhd->nhl", q, k) / math.sqrt(q.shape[-1]) + f.rpos[-Rp:][None, None], -1)
        r = torch.einsum("nhl,nlhd->nhd", att, v).reshape(N, -1)
        return squash(f.to_pic(r), getattr(f, "geometry", "cube"))

    def gates(self, Pbar):
        return torch.sigmoid(Pbar @ self.G.T + self.gb[None])              # (B, 2)


def selfslow_step(ss, params, group, base, grads, M, Vs, n, prev, Pbar, plastic, cfg, gopt):
    """One slow step for the self-model changes: Adam direction per copy, scaled by the copy's gate for that part;
    hypergradient of reading for the gate map (as for the query weights); each copy's change capped at a fraction of
    the shared weights' size. Returns the new step count, the stored (directions, gate inputs) and log values."""
    lr, frac = cfg["slow_lr"], cfg.get("selfslow_cap", 0.3)
    B = params[0].shape[0]
    hyper = 0.0
    if prev is not None and plastic in ("own", "other"):
        dirs, Pin = prev
        hg = torch.zeros(B, 2, device=Pbar.device)
        for g_, d_, gi in zip(grads, dirs, group):
            hg[:, gi] += -lr * (g_ * d_).flatten(1).sum(1)
        hyper = float(hg.mean())
        gopt.zero_grad()
        (ss.gates(Pin) * hg).sum().backward()
        gopt.step()
    with torch.no_grad():
        n += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        dirs = []
        for g_, m_, v_ in zip(grads, M, Vs):
            m_.mul_(b1).add_(g_, alpha=1 - b1)
            v_.mul_(b2).addcmul_(g_, g_, value=1 - b2)
            dirs.append((m_ / (1 - b1 ** n)) / ((v_ / (1 - b2 ** n)).sqrt() + eps))
        Pin = Pbar if plastic == "own" else torch.roll(Pbar, -1, 0)
        gate = torch.full((B, 2), 0.5, device=Pbar.device) if plastic == "ungated" else ss.gates(Pin)
        for q_, d_, gi in zip(params, dirs, group):
            q_.sub_(lr * gate[:, gi].view(B, *([1] * (q_.dim() - 1))) * d_)
        nw = ss.Ww.norm(dim=(2, 3))                                                   # (B, L)
        ss.Ww.mul_(torch.clamp(frac * base[0][None] / (nw + 1e-9), max=1.0)[:, :, None, None])
        for q_, bn in ((ss.Rq, base[1]), (ss.Uk, base[2]), (ss.Uv, base[3])):
            nq = q_.norm(dim=(1, 2))
            q_.mul_(torch.clamp(frac * bn / (nq + 1e-9), max=1.0)[:, None, None])
        rel_w = float((ss.Ww.norm(dim=(2, 3)) / base[0][None]).mean())
        rel_r = float(np.mean([float((q_.norm(dim=(1, 2)) / bn).mean()) for q_, bn in ((ss.Rq, base[1]), (ss.Uk, base[2]), (ss.Uv, base[3]))]))
        W = ss.Ww.flatten(1)
        Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)
        ov = float(np.mean([(Wn[a] @ Wn[c]).item() for a in range(B) for c in range(a + 1, B)])) if W.norm() > 0 else 0.0
    return n, (dirs, Pin), dict(rel_w=rel_w, rel_r=rel_r, gate_mean=float(gate.mean()), gate_sd=float(gate.std()),
                                copy_overlap=ov, hyper=hyper)


def slow_stats(U, V):
    """Effective rank of each copy's slow change U V^T (mean over copies and layers) and the mean cosine similarity of
    the changes of different copies, computed from rank-sized matrices."""
    U, V = U.float(), V.float()
    _, RU = torch.linalg.qr(U)
    _, RV = torch.linalg.qr(V)
    sv = torch.linalg.svdvals(RU @ RV.transpose(-1, -2))                       # (B, L, r)
    er = float(((sv.sum(-1) ** 2) / (sv ** 2).sum(-1).clamp_min(1e-12)).mean())
    UU = torch.einsum("aldr,cldk->aclrk", U, U)                                 # (B, B, L, r, r)
    VV = torch.einsum("aldr,cldk->aclrk", V, V)
    inner = torch.einsum("aclrk,aclrk->acl", UU, VV)                           # <W_a, W_c> per layer
    nrm = torch.sqrt(torch.diagonal(inner, dim1=0, dim2=1).T.clamp_min(1e-12))   # (B, L)
    cos = inner / (nrm[:, None] * nrm[None])
    B = U.shape[0]
    ov = float(np.mean([cos[a, c].mean().item() for a in range(B) for c in range(a + 1, B)]))
    return er, ov


def run(cfg):
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x8000)
        except Exception:
            pass
    torch.manual_seed(cfg["init_seed"])
    out = cfg["out"]
    os.makedirs(out, exist_ok=True)
    R = Runner12(cfg["model"], band=tuple(cfg["band"]), read_layer=cfg["read_layer"], window=cfg["window"])
    corpus = json.load(open(os.environ.get("V11_CORPUS", cfg["corpus"])))
    B, K, m, T = cfg["B"], cfg["K"], cfg["m"], cfg["T"]
    # the stream builder draws the four worlds' documents from one random sequence, so its length must not depend on T
    ids, gen, genres = make_streams(corpus, R.tok, B, cfg.get("stream_T", T), cfg["stream_seed"])
    ids, gen = ids[:, :T + 2], gen[:, :T + 2]
    warm = int(cfg.get("warmup", 0)) if cfg.get("same_text") else 0
    if warm:                                     # (v34) one world after different pasts: each copy reads its own stream at
        c_ = lambda x: x[:1].repeat(B, 1) if torch.is_tensor(x) else np.repeat(x[:1], B, 0)  # noqa: E731
        ids = torch.cat([c_(ids[:, :4]), ids[:, 4:warm], c_(ids[:, warm:])], 1)   # tokens 4..warm-1; the first four
        gen = np.concatenate([c_(gen[:, :4]), gen[:, 4:warm], c_(gen[:, warm:])], 1)  # (kept by the crop as sinks)
    elif cfg.get("same_text"):                                 # every copy reads world 0's text
        ids, gen = ids[:1].repeat(B, 1), np.repeat(gen[:1], B, 0)
    ids = ids.to(R.dev)
    sw = cfg.get("self_written")
    if sw:                                       # (v28) the text it reads is partly written under its own spotlight
        sw = dict(prompt=48, write=208, temperature=1.0, top_p=0.9) | (sw if isinstance(sw, dict) else {})
        assert not cfg.get("same_text") and not cfg.get("body") and not cfg.get("self_forecast"), "the plain loop writes"
        cur = ids[:, 0].clone()                  # the token each copy reads now
        ptr = torch.ones(B, dtype=torch.long, device=R.dev)          # the next real token of each copy's stream
        gsamp = torch.Generator(device=R.dev).manual_seed(31337 + cfg["init_seed"])
        traj_tok = np.zeros((T, B), np.int32)
        traj_written = np.zeros(T, bool)
        n_ext_on = 0
    f = Interface(R, m=m, p=cfg["p"], R=cfg["R"], bound=cfg["bound"], steer_reader=cfg.get("steer_reader", True)).to(R.dev)
    set_picture_map(f, cfg.get("picture_map"))
    set_picture_geometry(f, cfg.get("picture", "cube"), cfg.get("reader_route", "operation"))
    if cfg.get("init_iface"):                    # (v22) start from a trained interface: a developmental start
        load_interface(f, cfg["init_iface"], cfg, R.dev)
    elif cfg.get("consequence"):
        set_consequence(f)
    cfield = int(cfg.get("field_consequence") or 0)
    if cfield:                                   # (v26) the field also describes what its attending does
        set_field_consequence(f, cfield)
    fh = cfg.get("field_horizon")
    if fh:                                       # (v29) the head also describes its field at the horizon
        assert not cfield, "one extra part of the field at a time"
        set_field_consequence(f, m)
    content = cfg.get("reader_route") == "content"
    identity = cfg.get("identity", "position")
    another = cfg.get("describe") == "another"
    if another:
        zt = np.load(os.path.join(cfg["twin"], "twin_record.npz"))
        twin = {k_: torch.tensor(zt[k_], device=R.dev) for k_ in (("S", "D", "C") if cfield else ("S", "D"))}
    rec_tw = ({"S": [], "D": [], "s": []} | ({"C": []} if cfield else {})) if cfg.get("record_twin") else None
    div = cfg.get("div", "shared")
    plastic = cfg.get("plastic", "none")
    w_rec, w_div = 1.0, (0.0 if div == "none" else float(cfg.get("div_scale", 1.0)))   # div_scale: the self-negation sweep
    clo = cfg.get("closure", "loss")             # (v32) how the description of how it moves shapes the loop
    assert clo in ("loss", "measured", "use", "use_noise", "loss_describe"), "closure: loss | measured | use | use_noise | loss_describe"
    S_fixed = None
    gnoise = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(6060 + cfg["init_seed"])
    use_from = int(cfg.get("use_from", cfg.get("calib_tokens", 150)))
    adopt_n = int(cfg.get("adopt_tokens", 50)) if clo in ("measured", "use", "use_noise") else 0
    adopt_num, adopt_den, head_scale, rec_scale = 0.0, 0.0, None, None
    if clo != "loss":
        assert (identity == "dynamics" and not another and not cfield and not fh and w_div == 0.0 and plastic == "none"
                and not cfg.get("body") and not cfg.get("self_forecast") and not cfg.get("lasting") and not cfg.get("consequence")
                and cfg.get("reader_route", "operation") == "operation"), "closure through use is defined on the plain loop"
    unit = lambda z: z / z.norm(dim=-1, keepdim=True)  # noqa: E731
    pg = torch.Generator().manual_seed(77)
    pl = torch.Generator().manual_seed(1077 + cfg["init_seed"])           # copy-local probes
    cand = unit(torch.randn(64, m, generator=torch.Generator().manual_seed(4242))).to(R.dev)
    st = R.new_state(B)
    s = unit(torch.randn(B, m, generator=torch.Generator().manual_seed(cfg["init_seed"]))).to(R.dev)
    lam = np.array(cfg.get("lam", [1.0, 1.0, 1.0]))
    lasting = bool(cfg.get("lasting"))
    if lasting:
        assert cfg.get("picture") == "sphere", "the lasting picture lives on the sphere"
        f.lam_raw = nn.Parameter(torch.zeros((), device=R.dev))        # saved with the interface
    opt = torch.optim.Adam(f.parameters(), lr=cfg["lr"])
    slow, ss = None, None
    site = cfg.get("plastic_site", "query")
    if plastic != "none" and site == "selfmodel":
        ss = SelfSlow(R, f, B, cfg.get("rank", 4), cfg["p"], cfg["init_seed"]).to(R.dev)
        ss_params = [ss.Ww, ss.Rq, ss.Uk, ss.Uv]                      # V parts stay fixed (random, shared start)
        ss_group = [0, 1, 1, 1]
        ss_base = [torch.stack([f.writers[str(l)].weight for l in R.band]).norm(dim=(1, 2)).detach(),
                   f.rq.weight.norm().detach(), f.rk.weight.norm().detach(), f.rv.weight.norm().detach()]
        ss_m = [torch.zeros_like(q_) for q_ in ss_params]
        ss_v = [torch.zeros_like(q_) for q_ in ss_params]
        ss_prev, ss_n = None, 0
        gopt = torch.optim.Adam([ss.G, ss.gb], lr=cfg.get("gate_lr", 1e-3))
    if plastic != "none" and site == "query":
        slow = Slow(R, B, cfg.get("rank", 4), cfg["p"], cfg["init_seed"]).to(R.dev)
        gopt = torch.optim.Adam([slow.G, slow.gb], lr=cfg.get("gate_lr", 1e-3))
        mU, vU = torch.zeros_like(slow.U), torch.zeros_like(slow.U)
        mV, vV = torch.zeros_like(slow.V), torch.zeros_like(slow.V)
        prev = None                       # (gate inputs, Adam directions) of the last slow step, for the hypergradient
        n_slow = 0
    rows_live = torch.arange(B, device=R.dev)
    st0, hist0 = None, None
    field = cfg.get("field", "moves")
    if field == "reading":
        Psi = (torch.randn(R.d, m, generator=torch.Generator().manual_seed(4321)) / math.sqrt(R.d)).to(R.dev)
        resp_scale = None
    body = cfg.get("body")
    body_arm = cfg.get("body_arm", "own")
    if body:
        gb_ = torch.Generator().manual_seed(9090)
        Mb = {l: (torch.randn(R.qdim, body["k"], generator=gb_) / math.sqrt(body["k"])).to(R.dev) for l in R.band}
        gx = torch.Generator().manual_seed(5000 + cfg["init_seed"])
        xi = torch.randn(B, body["k"], generator=gx).to(R.dev)
        traj_xi = np.zeros((T, B, body["k"]), np.float32)
        Ab = (torch.randn(m, body["k"], generator=torch.Generator().manual_seed(9191)) / math.sqrt(body["k"])).to(R.dev)
    offs_live = ss.offs_fn(f, rows_live) if ss is not None else f.offs
    offs_rows = (lambda rows: ss.offs_fn(f, rows)) if ss is not None else (lambda rows: f.offs)
    read_rows = (lambda s_, h_, rows: ss.read(f, s_, h_, rows)) if ss is not None else (lambda s_, h_, rows: f.read(s_, h_))
    P_prev = None
    lam_traj = []
    if content:
        read_rows = lambda s_, h_, rows: f.read(s_, h_, item=None if P_prev is None else P_prev[rows])  # noqa: E731
    if lasting:
        _read0 = read_rows
        read_rows = lambda s_, h_, rows: lasting_combine(_read0(s_, h_, rows), None if P_prev is None else P_prev[rows],  # noqa: E731
                                                         torch.sigmoid(f.lam_raw))
    if body:
        def with_body(fn, rows):
            def offs(l, S):
                if body.get("mode", "query") == "stance":    # where the stance actually lands
                    return fn(l, unit(S + body["size"] * xi[rows] @ Ab.T))
                return fn(l, S) + (body["size"] * torch.tanh(xi[rows] @ Mb[l].T)).unsqueeze(1)
            return offs
        _rows0 = offs_rows
        offs_rows = lambda rows: with_body(_rows0(rows), rows)  # noqa: E731
        offs_live = with_body(offs_live, rows_live)
        if body_arm == "blind":
            st0 = R.new_state(B)
            hist0 = torch.zeros(B, 0, R.d, device=R.dev)
    kappa = cfg.get("spotlight")
    if kappa is not None:                      # (v24) the shared spotlight
        assert (cfg.get("reader_route", "operation") == "operation" and not body and ss is None and slow is None
                and not cfg.get("self_forecast")), "the spotlight is the reader's own attention, on the plain loop"
        offs_live, spot_live, spot_probe = channel(f, cfg)
        offs_rows = lambda rows: offs_live  # noqa: E731
        traj_spot = np.zeros((T, B, cfg["R"]), np.float16)
    cmode = cfg.get("consequence")
    if cmode:                                  # (v25) the consequence item
        assert kappa is not None and cfg.get("picture") == "sphere" and not content and not lasting, "the item needs the spotlight"
        traj_cons_att = np.zeros((T, B), np.float32)
        traj_cons_size = np.zeros((T, B), np.float32)
    if cfield:                                 # (v26) the consequence part of the field
        assert (kappa is not None and identity == "dynamics" and not cmode and cfg.get("picture") == "sphere" and not content
                and not lasting and not body), "the consequence part needs the spotlight, identity by dynamics and no item"
        cf_kind = cfg.get("field_consequence_kind", "projection")
        assert cf_kind in ("projection", "act") and (cf_kind != "act" or cfield == 2), "the act part has two features"
        Psi_c = cons_projection(R, cfield) if cf_kind == "projection" else None
        cf_nb = cfg.get("field_consequence_from", "own") == "neighbor"
        traj_cf_size = np.zeros((T, B), np.float32)
        traj_cf_scale = np.zeros(T, np.float32)
        traj_cf_acc = np.zeros(T, np.float32)
        traj_df_acc = np.zeros(T, np.float32)
    if fh:                                     # (v29) the same game at the horizon of its world
        fh = dict(k=32, every=16, random=4, source="own") | (fh if isinstance(fh, dict) else {})
        assert (kappa is not None and identity == "dynamics" and not cmode and not content and not lasting and not body
                and not another and cfg.get("spotlight_from", "own") == "own"), "the horizon needs the plain spotlight loop"
        gh = torch.Generator().manual_seed(2929 + cfg["init_seed"])
        gu_h = torch.Generator(device=R.dev).manual_seed(5150 + cfg["init_seed"])
        traj_hk_acc = np.full(T, np.nan, np.float32)
        lam_k, calib_k_sum, calib_k_n, Dk, Sk = 1.0, 0.0, 0, None, None
    sf = cfg.get("self_forecast")
    if sf:
        Psi_f = (torch.randn(R.d, sf["dim"], generator=torch.Generator().manual_seed(2468)) / math.sqrt(R.d)).to(R.dev)
        fc = nn.Sequential(nn.Linear(cfg["p"] + m, 64), nn.GELU(), nn.Linear(64, sf["dim"])).to(R.dev)
        Psi_t = (torch.randn(R.d, 64, generator=torch.Generator().manual_seed(1357)) / math.sqrt(R.d)).to(R.dev)
        traj_text = np.zeros((T, B, 64), np.float16)       # what is read (untouched) and what it read through its own
        traj_act = np.zeros((T, B, 64), np.float16)        # attending, for the offline analyses
        opt.add_param_group({"params": list(fc.parameters())})
        f_scale, f_mean, F_prev, f_raw, n_self = None, None, None, [], 0
        traj_fe = np.zeros((T, B), np.float32)
        traj_delta = np.zeros((T, B, sf["dim"]), np.float32)
        if st0 is None:
            st0 = R.new_state(B)
            hist0 = torch.zeros(B, 0, R.d, device=R.dev)
    keys = ("L_ext", "L_rec", "L_div", "acc", "world_dep", "move") + (("L_self",) if sf else ())
    log = {k: [] for k in keys}
    slog = {k: [] for k in ("t", "U_norm", "gate_mean", "gate_sd", "eff_rank", "copy_overlap", "hyper")}
    traj_s = np.zeros((T, B, m), np.float32)
    traj_P = np.zeros((T, B, cfg["p"]), np.float32)
    if clo != "loss":
        traj_desc_err = np.zeros((T, B), np.float32)       # the description's relative error (1 - R^2 per copy)
        traj_own_desc = np.zeros((T, B), np.float32)       # |its described own part| at its own stance
        traj_own_meas = np.zeros((T, B), np.float32)       # |the reader's actual own part| there
        traj_use_gap = np.zeros((T, B), np.float32)        # |described - actual own part| / |actual|
        traj_use_dev = np.zeros((T, B), np.float32)        # |used stance - the reader's own next stance|
    chunk_loss, chunk_ext, t0 = 0.0, 0.0, time.time()
    gprobe = bool(cfg.get("grad_probe")) and bool(cfg.get("consequence"))   # (v25 analysis) which pressure moves the item
    gp_ext, gp_rec, gp_rows = 0.0, 0.0, []
    P_chunk = []
    calib, calib_sums = cfg.get("calib_tokens", 150), np.zeros(4 if sf else 3)
    for t in range(T):
        ad_live = slow.fn(rows_live) if slow is not None else None
        if body:
            traj_xi[t] = xi.cpu().numpy()
        if st0 is not None:
            with torch.no_grad():
                h0b = R.lower(st0, ids[:, t])
                hr0, _ = R.upper_live(st0, h0b, lambda l, x: None, s)
                hist0 = torch.cat([hist0, hr0.float()], 1)[:, -f.R:]
        with torch.no_grad():
            h_in = R.lower(st, cur if sw else ids[:, t])
            S = unit(torch.randn(K, m, generator=pg)).to(R.dev)
            if cfg.get("fixed_probes"):                  # (v32) the same random probe stances all life
                S_fixed = S if S_fixed is None else S_fixed
                S = S_fixed
            S = torch.cat([S, s.detach()], 0)
            if another:                                  # the twin's probe stances: its field is the target
                S = twin["S"][t]
            b_idx = torch.arange(B, device=R.dev).repeat_interleave(S.shape[0])
            SS = S.repeat(B, 1)
            sb_probe = spot_probe(SS, st["hist"][b_idx], s, st["hist"], b_idx) if kappa is not None else None
            hr_probe = R.upper_probe(st, h_in, offs_rows(b_idx), SS, b_idx, slow.fn(b_idx) if slow is not None else None,
                                     sb=sb_probe).float()
            cbase = cons_baseline(R, f, cfg, st, h_in, offs_live, s, spot_live) if cmode else None
            c_probe = cons_probe(cbase, hr_probe, b_idx)
            if cfield:                         # what attending from each probe stance does to this token's processing
                h_even = R.upper_probe(st, h_in, offs_live, s, rows_live, sb=None).float()[:, 0]
                if cf_kind == "act":           # (v27) how sharply it would attend, how much the model would change
                    feats = act_features(f, kappa, SS, st["hist"][b_idx], hr_probe[:, 0], h_even[b_idx])
                    fe_ = feats[:, 1].view(B, -1)
                    traj_cf_scale[t] = float((fe_ - fe_.mean(1, keepdim=True)).abs().median())
                    C_own = act_part(feats, B)
                else:
                    C_own = ((hr_probe[:, 0] - h_even[b_idx]) @ Psi_c).view(B, -1, cfield)
                    C_own = C_own - C_own.mean(1, keepdim=True)
                    traj_cf_scale[t] = float(C_own.norm(dim=-1).median())
            if fh:
                Dk = None
            if fh and t % fh["every"] == 0:      # (v29) its counterfactual futures over the horizon
                Sk = torch.cat([unit(torch.randn(fh["random"], m, generator=gh)).to(R.dev), s.detach()], 0)
                bk = torch.arange(B, device=R.dev).repeat_interleave(Sk.shape[0])
                SSk = Sk.repeat(B, 1)
                uk = torch.rand(fh["k"], B, generator=gu_h, device=R.dev)
                cur_h = cur if sw else ids[:, t]
                ptr_h = ptr if sw else torch.full((B,), t + 1, dtype=torch.long, device=R.dev)
                Dk = horizon_rollout(R, f, kappa, offs_live, st, bk, SSk, SSk, cur_h, ptr_h, t, fh["k"], ids, sw, uk).view(B, -1, m)
                Dk = Dk - Dk.mean(1, keepdim=True)
                if fh["source"] == "neighbor":
                    Dk = torch.roll(Dk, -1, 0)
            if div == "local":
                Kl = cfg.get("K_local", 4)
                SL = unit(torch.randn(B * Kl, m, generator=pl)).to(R.dev)
                bl = torch.arange(B, device=R.dev).repeat_interleave(Kl)
                hr_loc = R.upper_probe(st, h_in, offs_rows(bl), SL, bl, slow.fn(bl) if slow is not None else None).float()
        if body and body_arm == "blind":
            hist_p = hist0[b_idx]                       # the field is formed from what is read, not from its own attending
        else:
            hist_p = torch.cat([st["hist"][b_idx], hr_probe], 1)[:, -f.R:]
        Pp = f.read(SS, hist_p, cons=c_probe) if cmode else read_rows(SS, hist_p, b_idx)
        if body and body_arm == "swap":        # what actually moves copy c is copy c+1's picture
            Pp = torch.roll(Pp.view(B, -1, Pp.shape[-1]), -1, 0).reshape(-1, Pp.shape[-1])
        Dp = (f.next_stance(Pp) - SS).view(B, -1, m)
        Dt = Dp if clo in ("loss", "loss_describe") else Dp.detach()   # closure (itself arm) | (v32) measured, not optimized
        if another:
            Dt = twin["D"][t]                                    # another life's recorded field, no closure
        if rec_tw is not None:
            rec_tw["S"].append(S.detach().cpu().numpy())
            rec_tw["D"].append(Dp.detach().cpu().numpy())
            rec_tw["s"].append(s.detach().cpu().numpy())
            if cfield:
                rec_tw["C"].append(C_own.cpu().numpy())
        if field == "reading":
            with torch.no_grad():
                Rr = (hr_probe[:, 0] @ Psi).view(B, -1, m)       # how this copy's reading responds at each probe stance
                if resp_scale is None:
                    resp_scale = float(Rr.norm(dim=-1).mean().clamp_min(1e-6))
                Rr = Rr / resp_scale
            Dt = Rr
        sb_live = None
        if kappa is not None:
            sb_live, p_live = spot_live(s, st["hist"])
            if p_live is not None:
                traj_spot[t, :, -p_live.shape[1]:] = p_live.detach().cpu().numpy()
        h_read, logits = R.upper_live(st, h_in, offs_live, s, ad_live, sb=sb_live)
        hist = torch.cat([st["hist"], h_read.float()], 1)[:, -f.R:]
        if cfield:
            traj_cf_size[t] = ((h_read[:, 0].detach().float() - h_even).norm(dim=-1)
                               / h_even.norm(dim=-1).clamp_min(1e-9)).cpu().numpy()
        if cmode:
            c_live = cons_live(cbase, h_read)
            P = f.read(s, hist, cons=c_live)
            traj_cons_att[t] = f._last_att[:, :, -1].mean(1).cpu().numpy()
            traj_cons_size[t] = (c_live.detach().norm(dim=-1) / cbase["h0"].norm(dim=-1).clamp_min(1e-9)).cpu().numpy()
        else:
            P = read_rows(s, hist0 if (body and body_arm == "blind") else hist, rows_live)
        if lasting:
            P_prev = P
            lam_traj.append(float(torch.sigmoid(f.lam_raw)))
        elif content:
            P_prev = P
        s_next = f.next_stance(torch.roll(P, -1, 0) if (body and body_arm == "swap") else P)
        if sw:
            per = sw["prompt"] + sw["write"]
            real_now, real_next = (t % per) < sw["prompt"], ((t + 1) % per) < sw["prompt"]
            traj_tok[t], traj_written[t] = cur.cpu().numpy(), not real_now
            tgt = ids[torch.arange(B, device=R.dev), ptr.clamp_max(ids.shape[1] - 1)]
            ext_on = real_now and real_next
            L_ext = F.cross_entropy(logits[:, 0].float(), tgt) * (1.0 if ext_on else 0.0)
            if ext_on and t < calib:
                n_ext_on += 1
            with torch.no_grad():
                if real_next:
                    cur, ptr = tgt, ptr + 1
                else:                            # the model writes the next token, under the spotlight it was given
                    pr_ = torch.softmax(logits[:, 0].detach().float() / sw["temperature"], -1)
                    srt, idx = pr_.sort(-1, descending=True)
                    srt = srt * ((srt.cumsum(-1) - srt) < sw["top_p"])
                    pick = torch.multinomial(srt / srt.sum(-1, keepdim=True), 1, generator=gsamp)[:, 0]
                    cur = idx.gather(-1, pick[:, None])[:, 0]
        else:
            L_ext = F.cross_entropy(logits[:, 0].float(), ids[:, t + 1])
        L_self = 0.0
        if sf:
            with torch.no_grad():
                traj_text[t] = (hr0[:, 0].float() @ Psi_t).cpu().numpy()
                traj_act[t] = (h_read[:, 0].float() @ Psi_t).cpu().numpy()
                if sf.get("target", "self") == "text":
                    delta = hr0[:, 0].float() @ Psi_f                            # the text's own course (control)
                else:
                    delta = (h_read[:, 0].float() - hr0[:, 0].float()) @ Psi_f  # what its own attending did to this token
                # scale from tokens 10-29: on the first tokens the cache holds too few positions for attention to differ
                # (the text control is also centred there: the untouched state has a large constant part)
                if f_scale is None:
                    if 10 <= t < 30:
                        f_raw.append(delta.clone())
                    if t == 30:
                        Dr = torch.cat(f_raw, 0)
                        f_mean = Dr.mean(0) if sf.get("target", "self") == "text" else torch.zeros_like(Dr[0])
                        f_scale = max(float((Dr - f_mean).norm(dim=-1).mean()), 1e-6)
            if f_scale is not None:
                delta = (delta - f_mean) / f_scale
                traj_delta[t] = delta.cpu().numpy()
                if F_prev is not None:
                    err = ((F_prev - delta) ** 2).sum(-1)                       # (B,)
                    L_self = err.mean()
                    traj_fe[t] = (err / (delta ** 2).sum(-1).clamp_min(1e-6)).detach().cpu().numpy()
                    if t < calib:
                        n_self += 1
                F_prev = fc(torch.cat([P, s_next.detach()], -1))                 # forecast of the next token's effect
        L_ext_copies = (F.cross_entropy(logits[:, 0].float(), tgt, reduction="sum") * (1.0 if ext_on else 0.0) if sw
                        else F.cross_entropy(logits[:, 0].float(), ids[:, t + 1], reduction="sum"))   # per-copy sum for the slow weights
        L_rec, accs, L_div = 0.0, [], 0.0
        own_rows = []
        if div == "local":
            with torch.no_grad():
                hp = torch.cat([st["hist"][bl], hr_loc], 1)[:, -f.R:]
                DL = (f.next_stance(read_rows(SL, hp, bl)) - SL).view(B, Kl, m)
        if identity == "dynamics":                               # identity by how it moves, not where it is
            dest_t = Dt + S[None]
            Dt_cmp = dest_t - dest_t.mean(1, keepdim=True)
        else:
            Dt_cmp = Dt
        if clo != "loss":                                        # (v32) the field's typical size per copy, a running average
            with torch.no_grad():
                ms_ = (Dt_cmp ** 2).sum(-1).mean(1)
                rec_scale = ms_.clone() if rec_scale is None else 0.99 * rec_scale + 0.01 * ms_
        if cfield:                                               # (v26) both parts, each at a typical row size of 1
            Ct = twin["C"][t] if another else (torch.roll(C_own, -1, 0) if cf_nb else C_own)
            sc_d = Dt_cmp.detach().norm(dim=-1).median().clamp_min(1e-8)
            sc_c = Ct.norm(dim=-1).median().clamp_min(1e-8)
            T_joint = torch.cat([Dt_cmp / sc_d, Ct / sc_c], -1)
            pacc_d, pacc_c = [], []
        for b in range(B):
            pf = f.predict_field(P[b:b + 1], S)
            if identity == "dynamics":
                pd_ = pf + S
                pf = pd_ - pd_.mean(0, keepdim=True)
            if cfield:
                pc = f.predict_cons(P[b:b + 1], S)
                pf = torch.cat([pf / sc_d, pc - pc.mean(0, keepdim=True)], -1)
                l_, a_ = contrastive(pf, T_joint, b)
                with torch.no_grad():
                    pacc_d.append(contrastive(pf[:, :m], T_joint[..., :m], b)[1])
                    pacc_c.append(contrastive(pf[:, m:], T_joint[..., m:], b)[1])
            elif clo == "loss_describe":                         # (v33) closure through the loss, no copy contest
                u_, v_ = pf.flatten(), Dt_cmp[b].flatten()
                l_ = 1.0 - (u_ * v_).sum() / (u_.norm() * v_.norm()).clamp_min(1e-8)
                with torch.no_grad():
                    a_ = contrastive(pf, Dt_cmp, b)[1]
                    traj_desc_err[t, b] = float(l_)
                    traj_own_desc[t, b] = float(pf[K + b].norm())
                    traj_own_meas[t, b] = float(Dt_cmp[b, K + b].norm())
            elif clo != "loss":                                  # (v32) graded only for being right
                tb_ = Dt_cmp[b]
                l_ = ((pf - tb_) ** 2).sum(-1).mean() / rec_scale[b].clamp_min(1e-8)
                with torch.no_grad():
                    if cfg.get("init_iface") and t < adopt_n:        # an archived head: fit its scale before it learns
                        u_ = pf - (S - S.mean(0, keepdim=True))      # its centred predicted moves
                        v_ = tb_ - (S - S.mean(0, keepdim=True))     # the centred measured moves
                        adopt_num, adopt_den = adopt_num + float((u_ * v_).sum()), adopt_den + float((u_ * u_).sum())
                    a_ = contrastive(pf, Dt_cmp, b)[1]
                    own_d, own_m = pf[K + b], tb_[K + b]             # its own stance is probe row K + b
                    own_rows.append(own_d)
                    traj_desc_err[t, b] = float(l_)
                    traj_own_desc[t, b] = float(own_d.norm())
                    traj_own_meas[t, b] = float(own_m.norm())
                    traj_use_gap[t, b] = float((own_d - own_m).norm() / own_m.norm().clamp_min(1e-8))
            else:
                l_, a_ = contrastive(pf, Dt_cmp, b)
            L_rec, accs = L_rec + l_ / B, accs + [a_]
            if div == "shared":
                c_star = S[Dt[b].detach().norm(dim=-1).argmax()]
            elif div == "local" and field == "reading":
                with torch.no_grad():
                    Rl = (hr_loc[:, 0] @ Psi).view(B, Kl, m) / resp_scale
                    here = Rr[b, K + b]                          # its reading at its own current stance
                    c_star = SL.view(B, Kl, m)[b, (Rl[b] - here).norm(dim=-1).argmax()]
            elif div == "local":
                c_star = SL.view(B, Kl, m)[b, DL[b].norm(dim=-1).argmax()]
            elif div == "local_picture":
                with torch.no_grad():
                    c_star = cand[f.predict_field(P[b:b + 1].detach(), cand).norm(dim=-1).argmax()]
            else:
                c_star = None
            if c_star is not None:
                L_div = L_div + (s_next[b] - c_star).norm() / B
        if clo != "loss" and cfg.get("init_iface") and t == adopt_n - 1:
            head_scale = adopt_num / max(adopt_den, 1e-12)
            with torch.no_grad():
                f.field[2].weight.mul_(head_scale)
                f.field[2].bias.mul_(head_scale)
        if clo in ("use", "use_noise") and t >= use_from:        # (v32) it moves by its own description of its own part
            with torch.no_grad():
                dest_mean = (Dt + S[None]).mean(1)               # where its reading takes any stance: the part the text sets
                z_use = dest_mean + torch.stack(own_rows)
                if clo == "use_noise":                           # control: the same push in a random direction
                    dev = z_use - s_next.detach()
                    rnd = torch.randn(dev.shape, generator=gnoise, device=dev.device)
                    z_use = s_next.detach() + unit(rnd) * dev.norm(dim=-1, keepdim=True)
                s_use = unit(z_use)
                traj_use_dev[t] = (s_use - s_next.detach()).norm(dim=-1).cpu().numpy()
            s_next = s_use + (s_next - s_next.detach())         # usefulness still trains the reading and decoding through it
        L_rec_k = None
        if fh and Dk is not None:                                # (v29) the same game at the horizon
            L_rec_k, accs_k = 0.0, []
            for b in range(B):
                pk = f.predict_cons(P[b:b + 1], Sk)
                l_, a_ = contrastive(pk - pk.mean(0, keepdim=True), Dk, b)
                L_rec_k, accs_k = L_rec_k + l_ / B, accs_k + [a_]
            traj_hk_acc[t] = float(np.mean(accs_k))
        if cfield:
            traj_df_acc[t], traj_cf_acc[t] = float(np.mean(pacc_d)), float(np.mean(pacc_c))
        if t < calib:
            calib_sums += [float(L_ext), float(L_rec) if t >= adopt_n else 0.0, float(L_div) if w_div else 1.0] + ([float(L_self)] if sf else [])
            if t == calib - 1:
                lam = calib / np.maximum(calib_sums, 1e-6)
                if sw:                                   # (v28) the mean over the tokens where it counts
                    lam[0] = max(n_ext_on, 1) / max(calib_sums[0], 1e-6)
                if clo != "loss":                        # (v32) the regression's weight from the tokens after adoption
                    lam[1] = max(calib - adopt_n, 1) / max(calib_sums[1], 1e-6)
                if fh:                                   # (v29) the same game at the horizon: the one-step game's weight
                    lam_k = float(lam[1])                # (its own mean is dominated by the zero-started head's first losses)
                if sf:
                    lam[3] = max(n_self, 1) / max(calib_sums[3], 1e-6)       # mean over the tokens where the forecast ran
                json.dump(dict(cfg=cfg, loss_weights=lam.tolist(), head_scale=head_scale, horizon_weight=lam_k if fh else None,
                               horizon_first_mean=(calib_k_sum / max(calib_k_n, 1)) if fh else None),
                          open(os.path.join(out, "config.json"), "w"), indent=1)
        loss = lam[0] * L_ext + w_rec * lam[1] * L_rec + (w_div * lam[2] * L_div if w_div else 0.0) + ((lam[3] if len(lam) > 3 else 1.0) * L_self if sf else 0.0)
        if fh and L_rec_k is not None:
            if t < calib:
                calib_k_sum, calib_k_n = calib_k_sum + float(L_rec_k), calib_k_n + 1
            else:
                loss = loss + w_rec * lam_k * L_rec_k
        chunk_loss = chunk_loss + (loss if t >= calib else 0.0 * loss)
        if gprobe and t >= calib:
            gp_ext, gp_rec = gp_ext + lam[0] * L_ext, gp_rec + w_rec * lam[1] * L_rec
        chunk_ext = chunk_ext + L_ext_copies
        P_chunk.append(P.detach())
        st["hist"] = hist.detach() if (t + 1) % cfg["chunk"] == 0 else hist
        traj_s[t] = s.detach().cpu().numpy()
        traj_P[t] = P.detach().cpu().numpy()
        mv = float((s_next.detach() - s.detach()).norm(dim=-1).mean())
        for k_, v_ in zip(keys, (float(L_ext), float(L_rec), float(L_div), float(np.mean(accs)),
                                 float((Dp.detach() - Dp.detach().mean(0, keepdim=True)).norm(dim=-1).mean()), mv)
                          + ((float(L_self),) if sf else ())):
            log[k_].append(v_)
        s = s_next
        if body:
            with torch.no_grad():
                a_ = 1.0 / body["tau"]
                xi = xi * (1 - a_) + math.sqrt(2 * a_) * torch.randn(B, body["k"], generator=gx).to(R.dev)
        if (t + 1) % cfg["chunk"] == 0:
            if t >= calib:
                if slow is not None:
                    gU, gV = torch.autograd.grad(chunk_ext, [slow.U, slow.V], retain_graph=True)
                if ss is not None:
                    ss_g = torch.autograd.grad(chunk_ext, ss_params, retain_graph=True, allow_unused=True)
                    ss_g = [torch.zeros_like(q_) if g_ is None else g_ for q_, g_ in zip(ss_params, ss_g)]
                if gprobe:                     # the two weighted pressures on the item's position bias, before clipping
                    g_e = torch.autograd.grad(gp_ext, [f.cons_pos], retain_graph=True, allow_unused=True)[0]
                    g_r = torch.autograd.grad(gp_rec, [f.cons_pos], retain_graph=True, allow_unused=True)[0]
                    gp_rows.append([t, 0.0 if g_e is None else float(g_e), 0.0 if g_r is None else float(g_r)])
                opt.zero_grad()
                chunk_loss.backward()
                torch.nn.utils.clip_grad_norm_(f.parameters(), 1.0)
                opt.step()
                if ss is not None:
                    for q_ in list(ss.parameters()):
                        q_.grad = None
                    ss_n, ss_prev, srec = selfslow_step(ss, ss_params, ss_group, ss_base, ss_g, ss_m, ss_v, ss_n, ss_prev,
                                                       torch.stack(P_chunk).mean(0), plastic, cfg, gopt)
                    if (t + 1) % 200 == 0:
                        for k_, v_ in srec.items():
                            slog.setdefault(k_, []).append(v_)
                        slog["t"].append(t + 1)
                if slow is not None:
                    slow.U.grad, slow.V.grad = None, None       # the slow weights learn only from reading (gU, gV)
                    with torch.no_grad():
                        Pbar = torch.stack(P_chunk).mean(0)                         # (B, p)
                        hyper = 0.0
                        # hypergradient: d(this chunk's loss)/d(last gates) = -lr * <grad now, last direction> per component
                        if prev is not None and plastic in ("own", "other"):
                            dirU, dirV, Pin = prev
                            hg = -cfg["slow_lr"] * ((gU * dirU).sum(2) + (gV * dirV).sum(2))     # (B, L, r)
                            hyper = float(hg.mean())
                    if prev is not None and plastic in ("own", "other"):
                        gopt.zero_grad()
                        gl = slow.gates(Pin)
                        (gl * hg).sum().backward()
                        gopt.step()
                    with torch.no_grad():
                        n_slow += 1
                        b1, b2, eps = 0.9, 0.999, 1e-8
                        mU.mul_(b1).add_(gU, alpha=1 - b1); vU.mul_(b2).addcmul_(gU, gU, value=1 - b2)
                        mV.mul_(b1).add_(gV, alpha=1 - b1); vV.mul_(b2).addcmul_(gV, gV, value=1 - b2)
                        dirU = (mU / (1 - b1 ** n_slow)) / ((vU / (1 - b2 ** n_slow)).sqrt() + eps)
                        dirV = (mV / (1 - b1 ** n_slow)) / ((vV / (1 - b2 ** n_slow)).sqrt() + eps)
                        Pin = Pbar if plastic == "own" else torch.roll(Pbar, -1, 0)
                        gate = torch.full((B, len(R.band), slow.U.shape[-1]), 0.5, device=R.dev) if plastic == "ungated" else slow.gates(Pin)
                        slow.U.sub_(cfg["slow_lr"] * gate[:, :, None, :] * dirU)
                        slow.V.sub_(cfg["slow_lr"] * gate[:, :, None, :] * dirV)
                        nrm = slow.U.norm(dim=(2, 3), keepdim=True)
                        slow.U.mul_(torch.clamp(cfg.get("slow_cap", 1.0) / (nrm + 1e-9), max=1.0))
                        prev = (dirU, dirV, Pin)
                        if (t + 1) % 200 == 0:
                            er, ov = slow_stats(slow.U, slow.V)
                            for k_, v_ in (("t", t + 1), ("U_norm", float(slow.U.norm(dim=(2, 3)).mean())), ("gate_mean", float(gate.mean())),
                                           ("gate_sd", float(gate.std())), ("eff_rank", er), ("copy_overlap", ov), ("hyper", hyper)):
                                slog[k_].append(v_)
            chunk_loss, chunk_ext = 0.0, 0.0
            gp_ext, gp_rec = 0.0, 0.0
            P_chunk = []
            if sf and f_scale is not None:
                F_prev = fc(torch.cat([P.detach(), s.detach()], -1))
            s = s.detach()
            if (lasting or content) and P_prev is not None:
                P_prev = P_prev.detach()
            st["K"] = [k_.detach() for k_ in st["K"]]
            st["V"] = [v_.detach() for v_ in st["V"]]
        if (t + 1) % cfg.get("print_every", 250) == 0:
            pe = cfg.get("print_every", 250)
            w = traj_s[t - pe + 1:t + 1]
            apart = float(np.mean([np.linalg.norm(w[:, a] - w[:, c], axis=-1).mean() for a in range(B) for c in range(a + 1, B)]))
            extra = ""
            if ss is not None and slog["t"]:
                extra = (f" | self-model slow: writers {slog['rel_w'][-1]:.3f} reader {slog['rel_r'][-1]:.3f} "
                         f"gate {slog['gate_mean'][-1]:.2f}+-{slog['gate_sd'][-1]:.2f} copies' cosine {slog['copy_overlap'][-1]:.2f}")
            if slow is not None and slog["t"]:
                extra = (f" | slow |U| {slog['U_norm'][-1]:.3f} gate {slog['gate_mean'][-1]:.2f}±{slog['gate_sd'][-1]:.2f} "
                         f"rank {slog['eff_rank'][-1]:.2f} overlap {slog['copy_overlap'][-1]:.2f}")
            if sf:
                extra += f" | forecast error {traj_fe[t - pe + 1:t + 1].mean():.3f}"
            if lasting:
                extra += f" | keeps {lam_traj[-1]:.3f} of its last picture"
            if clo != "loss":
                w_ = slice(t - pe + 1, t + 1)
                extra += (f" | describes itself: error {traj_desc_err[w_].mean():.3f} own part {traj_own_desc[w_].mean():.3f}"
                          f" (reader {traj_own_meas[w_].mean():.3f}) gap {np.median(traj_use_gap[w_]):.2f}")
            print(f"[{div}/{plastic} s{cfg['init_seed']}] t={t + 1} {time.time() - t0:.0f}s Lext {np.mean(log['L_ext'][-pe:]):.3f} "
                  f"acc {np.mean(log['acc'][-pe:]):.2f} move {np.median(log['move'][-pe:]):.4f} worldsApart {apart:.3f} "
                  f"worldDep {np.mean(log['world_dep'][-pe:]):.3f}{extra} mem {torch.cuda.max_memory_allocated() / 2**30:.2f}GiB", flush=True)
        if (t + 1) in (T // 4, T // 2, 3 * T // 4, T) or (cfg.get("save_every") and (t + 1) % cfg["save_every"] == 0):
            torch.save(f.state_dict(), os.path.join(out, f"iface_{t + 1}.pt"))
            if slow is not None:
                torch.save(slow.state_dict(), os.path.join(out, f"slow_{t + 1}.pt"))
            if ss is not None:
                torch.save(ss.state_dict(), os.path.join(out, f"selfslow_{t + 1}.pt"))
            if sf:
                torch.save(fc.state_dict(), os.path.join(out, f"forecast_{t + 1}.pt"))
    if rec_tw is not None:
        np.savez_compressed(os.path.join(out, "twin_record.npz"), **{k_: np.array(v_, np.float32) for k_, v_ in rec_tw.items()})
    extra_save = {"xi": traj_xi} if body else {}
    if kappa is not None:
        extra_save["spot"] = traj_spot
    if sw:
        extra_save.update(tok=traj_tok, written=traj_written)
    if cmode:
        extra_save.update(cons_att=traj_cons_att, cons_size=traj_cons_size)
        if gprobe:
            extra_save["cons_grad"] = np.array(gp_rows, np.float64).reshape(-1, 3)   # (token, prediction, game)
    if cfield:
        extra_save.update(cf_size=traj_cf_size, cf_scale=traj_cf_scale, cf_acc=traj_cf_acc, df_acc=traj_df_acc)
    if fh:
        extra_save["hk_acc"] = traj_hk_acc
    if lasting:
        extra_save["lam"] = np.array(lam_traj, np.float32)
    if clo != "loss":
        extra_save.update(desc_err=traj_desc_err, own_desc=traj_own_desc, own_meas=traj_own_meas, use_gap=traj_use_gap,
                          use_dev=traj_use_dev)
    if sf:
        extra_save.update(forecast_err=traj_fe, delta=traj_delta, text64=traj_text, act64=traj_act,
                          f_scale=np.array(f_scale if f_scale is not None else np.nan, np.float32))
    np.savez_compressed(os.path.join(out, "life.npz"), s=traj_s, P=traj_P, genre=gen[:, :T].T, **extra_save,
                        **{k: np.array(v, np.float32) for k, v in log.items()}, **{"slow_" + k: np.array(v, np.float32) for k, v in slog.items()})
    open(os.path.join(out, "DONE"), "w").write(f"{time.time() - t0:.1f}\n")


if __name__ == "__main__":
    run(json.load(open(sys.argv[1])))
