"""The loop's self-referential core on the model's recorded states (DESIGN_CORE_SANDBOX.md).
Kept exactly: the interface (llm_loop.Interface with loop12's picture geometry and reading route), its reading of the
last R read-layer states with the stance as query, the probe stances (loop12's generator), the field, the
identity-by-dynamics game with closure, chunked online learning after calibration, archived interfaces as starts.
Dropped: the stance's offsets and spotlight on the model (the states are the untouched model's, recorded once), the
prediction loss, the written world.
  record OUT_NPZ STREAM_SEED STREAM_T T [MODEL BAND_FIRST BAND_LAST READ_LAYER]   the untouched model's read-layer
                                          states, four streams (float16)
  life CONFIG_JSON                        one life; cfg as loop12's plus "states" (a recorded file), "closure" ("loss" as
                                          loop12; "measured": the field without gradient, the description graded by
                                          regression, an archived head's scale adopted first, as loop12 v32; "use":
                                          from use_from on it moves by the part the text sets plus its own
                                          description of its own part; "use_noise": the same-sized push in a random
                                          direction; "loss_describe": closure through the loss as "loss", but the
                                          description graded only by how well it matches the field's shape
                                          (1 - correlation per copy, scale-free), no copy contest),
                                          "fixed_probes", "same_text" (every copy reads stream 0),
                                          "trace" {"strength", "fade", "from": "own"|"neighbor"}
                                          (DESIGN_OWN_WAY_OF_READING.md: each copy's private, stance-conditioned
                                          change to its reader's query, renewed after every token by the key it
                                          attended from its stance and fading by `fade`; no gradient; "neighbor"
                                          (control): copy c's trace is renewed by copy c+1's attending),
                                          "world": "written" (the live model reads and writes under each copy's
                                          spotlight, cfg "spotlight" and "self_written" as loop12 v28; the reader's
                                          window holds the live states; the probes read that window without steering
                                          the current token; no prediction loss), "shared_noise" (all copies draw
                                          written tokens with the same uniforms, so their texts differ only through
                                          their own spotlights),
                                          "warmup" N with "same_text": each copy first reads its own stream for N
                                          tokens (its own window, model context and stance), then all copies read
                                          stream 0 from the same place on (one world, different pasts; without it
                                          the copies are identical from token 1, as their first window holds one
                                          state and the query cannot act),
                                          "progress" {"slow": 200, "avg": 50, "from": "own"|"neighbor"} with closure
                                          "measured" and "fixed_probes" (DESIGN_COMING_TO_KNOW_ITSELF.md): the loop is
                                          drawn to where its picture of itself is getting better. Beside its
                                          description it keeps a slow one (the average of its recent selves over
                                          `slow` tokens); at each fixed stance its progress is how much better the
                                          present one fits the field there than the slow one (averaged over `avg`
                                          tokens); the next stance is its own next stance blended with the
                                          progress-weighted direction of the fixed stances, the blend being the mean
                                          share of the slow one's error the present one has removed (0..1). No loss,
                                          no gradient, no weight to set.
                                          "neighbor" (control): copy c is drawn by copy c+1's progress.
                                          "describe": "neighbor" (control, measured): copy c's picture describes copy
                                          c+1's field,
                                          "probe_seed" (default 77 as loop12; the core is deterministic, so lives
                                          from one start differ only through it: in the real loop they differed
                                          through the GPU's nondeterministic backward pass);
                                          closure "know" (local design review, section 4, a loop whose
                                          acting changes itself): only the self-model (the field head) learns, only
                                          for accuracy: it predicts the whole destination from every probe stance
                                          (the measured field as the target only; no gradient reaches the reader, the
                                          picture map, the decoder or the body). "know" {"move": "self" (from
                                          `move_from` on it moves to where its self-model says it goes from where it
                                          stands) | "reading" (by its reading; the self-model learns but is not used),
                                          "state": "picture" | "zero" (control: the self-model's picture held at
                                          zero) | "unbodied" (control: the picture read without its body's
                                          tiredness), "model_of": "own" | "neighbor" (control: the field of its
                                          neighbour's body on its own text)};
                                          "body" {"strength": 0.8, "time": 100}: each copy's stance acts on its reader
                                          through a body that tires with use: the reader's query takes (I - strength U)
                                          s, U the average of s s^T over the stances it acted with (over `time`
                                          tokens), so a way of acting held does less and one left alone recovers;
                                          "habit" {"size": 0.5, "fill": 1000, "from": "own"|"neighbor"}
                                          (DESIGN_A_HISTORY_OF_ITS_OWN.md, direction 1): each copy's own change to its
                                          stance-to-query map, written after every token by what was particular in
                                          what it attended (the attended key minus the window's mean key) tied to the
                                          stance that reached the reader; held to a fixed size (`size` times the map's
                                          own), so new habits displace old ones; writes held to one direction would fill
                                          it in `fill` tokens; no gradient; "neighbor" (control): written by copy c+1's
                                          attending;
                                          "moved" {"angle": 0.1, "turn": "own"|"fixed"} (direction 2): the stance is
                                          turned by a fixed angle toward where its reading would take it, never
                                          replaced; "fixed" (control): toward where the reading from its first stance
                                          would take it, so the turn does not depend on where it stands;
                                          "futures" {"every": 100, "horizon": 32, "last": 1500}: in the last `last`
                                          tokens, every `every` tokens, 8 probe stances and the copies' own live on
                                          `horizon` tokens of the copy's coming text by its present rule (habits and
                                          body as they are, nothing written): the share of their spread left and the
                                          places they reach (cosine 0.9), as the futures check of 23 October;
                                          with "interaction": true also, from the copy's own stance, one head's part
                                          of it (m / heads coordinates) changed at a time: the other parts' spread
                                          after `horizon` tokens as a share of theirs in the futures above;
                                          "reader" {"kind": ..., "contrast": 3, "time": 50, "from": "own"|"neighbor",
                                          "stride": 16, "stance_q": true} (DESIGN_A_READER_OF_ITS_OWN.md): kind
                                          "particular": what it attends is measured against the window's mean value
                                          and the difference enlarged `contrast` times; "adapted": the same against
                                          the copy's own running expectation of what it has read (over `time`
                                          tokens; "neighbor" (control): built from copy c+1's reading); "wide": its
                                          window holds every `stride`-th state ending at the present one; "heads":
                                          each head's query takes only its own part of the stance (rescaled to keep
                                          its size); "gain": the stance's part of the query (without the query
                                          map's bias) `gain` times; "stance_q": false (control) removes the stance
                                          from the query;
                                          "stance_path" {"last": 1000, "every": 13}: at those tokens 12 fixed random
                                          stances read the copy's present window with its present reader; the
                                          spread across them of where it looks, what it reads, its picture and next
                                          stance, against the spread across windows (stance_path.py, in the life);
                                          "turned" {"turn": "own"|"fixed"} (WHERE_ITS_OWN_STATE_GOES.md (b)): the
                                          stance is never replaced by its reading's destination but turned by the
                                          rotation that carries the last destination to the present one (in their
                                          plane; at the first token it stays), so its angle to its reading's last
                                          destination never changes (logged: turn_inv) and stances are kept apart
                                          except where their readings differ; "fixed" (control): the destinations
                                          of its first stance's reading, so the turn does not depend on where it
                                          stands (path integration of the text);
                                          "same_start": every copy starts from copy 0's first stance;
                                          "same_text" (every copy reads stream 0); saves life.npz, config.json, iface_T.pt
  frozen OUT_JSON STATES_NPZ T NAME=RUN_DIR::IFACE ...   frozen pass on recorded new texts: the stance's course
                                          (frozen.npz per life, for trajectory_complexity) and, every 50 tokens, the
                                          futures over 32 tokens from 4 random probe stances and the copies' own (the
                                          stances' difference left, and the places they reach), as horizon_check
Usage: python core32.py record|life|frozen ..."""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "C:/Users/assas/PycharmProjects/jspace/research_v11/llm")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llm_loop import Interface, contrastive, make_streams  # noqa: E402
from loop12 import load_interface, set_picture_geometry, set_picture_map, squash  # noqa: E402

DEV = os.environ.get("CORE_DEV") or ("cuda" if torch.cuda.is_available() else "cpu")
CORPUS = os.environ.get("V11_CORPUS", "C:/Users/assas/jspace_v11_runs/text/corpus.json")
unit = lambda z: z / z.norm(dim=-1, keepdim=True)  # noqa: E731


class Shape:
    """What Interface needs from a runner."""
    def __init__(self, d, qdim, band):
        self.d, self.qdim, self.band = int(d), int(qdim), [int(b) for b in band]


def record(out, stream_seed, stream_T, T, model="Qwen/Qwen3-0.6B", band=(12, 18), read_layer=19, window=256, B=4):
    from loop12 import Runner12
    R = Runner12(model, band=band, read_layer=read_layer, window=window)
    ids, gen, _ = make_streams(json.load(open(CORPUS)), R.tok, B, stream_T, stream_seed)
    ids = ids[:, :T + 1].to(R.dev)
    st = R.new_state(B)
    H = np.zeros((B, T + 1, R.d), np.float16)
    s0 = torch.zeros(B, 8, device=R.dev)
    t0 = time.time()
    with torch.no_grad():
        for t in range(T + 1):
            h_read, _ = R.upper_live(st, R.lower(st, ids[:, t]), lambda l, x: None, s0)
            H[:, t] = h_read[:, 0].float().cpu().numpy()
    np.savez_compressed(out, h=H, ids=ids.cpu().numpy(), genre=gen[:, :T + 1], d=R.d, qdim=R.qdim,
                        band=np.array(R.band), stream_seed=stream_seed, stream_T=stream_T)
    print(f"recorded {H.shape} in {time.time() - t0:.0f}s -> {out}", flush=True)


def load_states(path):
    """One recorded file, or a list of them joined along the copies (more copies, each with a stream of its own)."""
    if isinstance(path, str):
        return dict(np.load(path))
    Zs = [dict(np.load(x)) for x in path]
    Z = dict(Zs[0])
    for k in ("h", "ids", "genre"):
        Z[k] = np.concatenate([z[k] for z in Zs], 0)
    return Z


def build(cfg, Z):
    shape = Shape(Z["d"], Z["qdim"], Z["band"])
    f = Interface(shape, m=cfg["m"], p=cfg["p"], R=cfg["R"], bound=cfg["bound"], steer_reader=cfg.get("steer_reader", True)).to(DEV)
    set_picture_map(f, cfg.get("picture_map"))
    set_picture_geometry(f, cfg.get("picture", "cube"), cfg.get("reader_route", "operation"))
    if cfg.get("init_iface"):
        load_interface(f, cfg["init_iface"], cfg, DEV)
    return f


def read_trace(f, s, hist, Qrows):
    """loop12's read (sphere picture, operation route) with a per-row change to the query, Qrows (N, heads, dk, m)
    applied to the stance; returns the picture, the attention and the keys."""
    N, Rp, _ = hist.shape
    a = f.rln(hist)
    k = f.rk(a).view(N, Rp, f.rh, -1)
    v = f.rv(a).view(N, Rp, f.rh, -1)
    q = (f.rq_bias.expand(N, -1) + f.rq(s)).view(N, f.rh, -1) + torch.einsum("nhdm,nm->nhd", Qrows, s)
    att = torch.softmax(torch.einsum("nhd,nlhd->nhl", q, k) / math.sqrt(q.shape[-1]) + f.rpos[-Rp:][None, None], -1)
    r = torch.einsum("nhl,nlhd->nhd", att, v).reshape(N, -1)
    return squash(f.to_pic(r), f.geometry), att, k


def window(f, x, rd):
    """The reader's window from the kept history x (N, L, d): its last R states, or (wide) every stride-th state
    ending at the present one."""
    if rd is None or rd["kind"] != "wide":
        return x[:, -f.R:]
    idx = torch.arange(x.shape[1] - 1, -1, -rd["stride"], device=x.device).flip(0)[-f.R:]
    return x[:, idx]


def rd_read(f, rd, s, win, E=None):
    """loop12's read (sphere picture, operation route) by one of the readers of DESIGN_A_READER_OF_ITS_OWN.md (rd
    None: unchanged); E (N, heads, dv) the expectation of the "adapted" reader. Returns the picture, the attention, the
    keys, what it attended (N, heads, dv) and what it read after the contrast."""
    N, Rp, _ = win.shape
    a = f.rln(win)
    k = f.rk(a).view(N, Rp, f.rh, -1)
    v = f.rv(a).view(N, Rp, f.rh, -1)
    kind = rd["kind"] if rd is not None else "plain"
    if rd is not None and not rd["stance_q"]:
        sq = 0.0                                                 # (control) it attends by position and content alone
    elif kind == "heads":                                        # each head steered by its own part of the stance
        m_ = s.shape[-1]
        per = m_ // f.rh
        mask = torch.zeros(f.rh, m_, device=s.device)
        for h_ in range(f.rh):
            mask[h_, h_ * per:(h_ + 1) * per] = 1.0
        W_ = f.rq.weight.view(f.rh, -1, m_)
        sq = (math.sqrt(f.rh) * torch.einsum("hdm,nhm->nhd", W_, s[:, None] * mask[None]) + f.rq.bias.view(f.rh, -1)).reshape(N, -1)
    elif kind == "gain":                                         # the stance's part of the query, `gain` times
        sq = rd["gain"] * (f.rq(s) - f.rq.bias) + f.rq.bias
    else:
        sq = f.rq(s)
    q = (f.rq_bias.expand(N, -1) + sq).view(N, f.rh, -1)
    att = torch.softmax(torch.einsum("nhd,nlhd->nhl", q, k) / math.sqrt(q.shape[-1]) + f.rpos[-Rp:][None, None], -1)
    raw = torch.einsum("nhl,nlhd->nhd", att, v)
    base = v.mean(1) if kind == "particular" else ((raw if E is None else E) if kind == "adapted" else None)
    r = raw if base is None else base + rd["contrast"] * (raw - base)   # what is particular, enlarged
    return squash(f.to_pic(r.reshape(N, -1)), f.geometry), att, k, raw, r


def read_item(f, s, hist, item):
    """(WHERE_ITS_OWN_STATE_GOES.md section 13) loop12's read (sphere picture, operation route) with one more item to
    read beside the window: item (N, d) in the read-layer space, with the position bias of the window's present state.
    Returns the picture and the attention (N, heads, R' + 1; the item last)."""
    N, Rp, _ = hist.shape
    a = f.rln(torch.cat([hist, item[:, None].to(hist.dtype)], 1))
    k = f.rk(a).view(N, Rp + 1, f.rh, -1)
    v = f.rv(a).view(N, Rp + 1, f.rh, -1)
    q = (f.rq_bias.expand(N, -1) + f.rq(s)).view(N, f.rh, -1)
    pos = torch.cat([f.rpos[-Rp:], f.rpos[-1:]])
    att = torch.softmax(torch.einsum("nhd,nlhd->nhl", q, k) / math.sqrt(q.shape[-1]) + pos[None, None], -1)
    r = torch.einsum("nhl,nlhd->nhd", att, v).reshape(N, -1)
    return squash(f.to_pic(r), f.geometry), att


def turn(s, g, angle):
    """(direction 2) s turned by a fixed angle along the great circle toward g: moved, never replaced, never still."""
    u = g - (g * s).sum(-1, keepdim=True) * s
    return unit(math.cos(angle) * s + math.sin(angle) * u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8))


def carry(a, b, x):
    """((b) of WHERE_ITS_OWN_STATE_GOES.md) x turned by the rotation that carries the unit a to the unit b in their
    plane, left alone in every other direction (rows; where a and b coincide, x stays)."""
    c = (a * b).sum(-1, keepdim=True)
    w = b - c * a
    sn = w.norm(dim=-1, keepdim=True)                            # the sine of the angle from a to b
    w = w / sn.clamp_min(1e-8)
    xa, xw = (x * a).sum(-1, keepdim=True), (x * w).sum(-1, keepdim=True)
    out = x + (c * xa - sn * xw - xa) * a + (sn * xa + c * xw - xw) * w
    return torch.where(sn > 1e-8, out, x)


def field(f, S, hist, h_t, B, Qh=None, G=None, rd=None, E=None):
    """Where each copy's next stance would go from each probe stance (loop12's field, reading only); with its trace;
    through its body G (B, m, m) when it has one (the stance reaches the reader's query as G s); by its reader rd."""
    NS = S.shape[0]
    b_idx = torch.arange(B, device=DEV).repeat_interleave(NS)
    SS = S.repeat(B, 1)
    SA = SS if G is None else torch.einsum("nij,nj->ni", G[b_idx], SS)
    if rd is not None:
        win = window(f, torch.cat([hist[b_idx], h_t[b_idx]], 1), rd)
        Pp = rd_read(f, rd, SA, win, None if E is None else E[b_idx])[0]
        return (f.next_stance(Pp) - SS).view(B, NS, -1)
    hist_p = torch.cat([hist[b_idx], h_t[b_idx]], 1)[:, -f.R:]
    Pp = f.read(SA, hist_p) if Qh is None else read_trace(f, SA, hist_p, Qh[b_idx])[0]
    return (f.next_stance(Pp) - SS).view(B, NS, -1)


def life(cfg):
    torch.manual_seed(cfg["init_seed"])
    out = cfg["out"]
    os.makedirs(out, exist_ok=True)
    Z = load_states(cfg["states"])
    B, K, m, T = cfg["B"], cfg["K"], cfg["m"], cfg["T"]
    assert Z["h"].shape[0] >= B, "one recorded stream per copy"
    Z["h"], Z["ids"], Z["genre"] = Z["h"][:B], Z["ids"][:B], Z["genre"][:B]
    H = torch.tensor(Z["h"][:, :T + 1], dtype=torch.float32, device=DEV)
    warm = int(cfg.get("warmup", 0)) if cfg.get("same_text") else 0
    if cfg.get("same_text"):                                     # one world; with warmup, after different pasts
        H = torch.cat([H[:, :warm], H[:1, warm:].repeat(B, 1, 1)], 1)
    f = build(cfg, Z)
    assert cfg.get("identity") == "dynamics", "the core is the identity-by-dynamics loop"
    written = cfg.get("world") == "written"
    if written:                                                  # a world that answers: the live model writes
        from loop12 import Runner12, channel, sample_shared, written_regime
        Rm = Runner12(cfg["model"], device=DEV, dtype=torch.float32 if DEV == "cpu" else torch.bfloat16, band=tuple(cfg["band"]),
                      read_layer=cfg["read_layer"], window=cfg["window"])   # (on the CPU in float32, faster there than bf16)
        assert Rm.d == H.shape[-1], "the recorded states and the live model must be the same model"
        ids_w = torch.tensor(Z["ids"], dtype=torch.long, device=DEV)
        sw = written_regime(cfg)
        if cfg.get("same_text"):                                 # real tokens read by the end of the warm-up, then stream 0
            per_ = sw["prompt"] + sw["write"]
            n_real = sum(1 for t_ in range(warm) if ((t_ + 1) % per_) < sw["prompt"]) + 1
            ids_w = torch.cat([ids_w[:, :n_real], ids_w[:1, n_real:].repeat(B, 1)], 1)
        assert sw and cfg.get("spotlight") is not None, "the written world needs self_written and the spotlight"
        offs_live, spot_live, _ = channel(f, cfg)
        place = cfg.get("act_place")                             # (WHERE_ITS_OWN_STATE_GOES.md section 22) where and how
        act_l = None if (place is None or place["layers"] == "band") else list(range(Rm.read_layer + 1, Rm.L))
        nb_take = cfg.get("spotlight_from", "own") == "neighbor"   # its emphasis acts: {"layers": band|upper,

        def emphasis(s_, h_):                                    # "how": scores|take|none, "share"}
            """(bias on where its model looks, share of what its model takes up, its own selection p)."""
            sb_, p_ = spot_live(s_, h_)
            if place is None or p_ is None:
                return sb_, None, p_
            if place["how"] == "take":
                return None, (place["share"], torch.roll(p_, -1, 0) if nb_take else p_), p_
            return (sb_ if place["how"] == "scores" else None), None, p_
        ll = cfg.get("lens_log")                                 # (section 22) per segment and copy: what its selection
        if ll:                                                   # takes up, and the kinds of what its model writes
            from lexicons import kind_lens, token_sets
            ll_seg = int(ll.get("segment", 256))
            n_seg = (T + ll_seg - 1) // ll_seg
            ll_sets = [torch.tensor(v, device=DEV) for v in token_sets(Rm.tok).values()]
            ll_vec = torch.zeros(n_seg, B, Rm.d, device=DEV)
            ll_write = torch.zeros(n_seg, B, len(ll_sets), device=DEV)
            ll_n = np.zeros(n_seg, np.int32)
        st_m = Rm.new_state(B)
        cur = ids_w[:, 0].clone()
        ptr = torch.ones(B, dtype=torch.long, device=DEV)
        gsamp = torch.Generator(device=DEV).manual_seed(31337 + cfg["init_seed"])
        traj_tok, traj_written = np.zeros((T, B), np.int32), np.zeros(T, bool)
        infl = cfg.get("influence")                              # (WHERE_ITS_OWN_STATE_GOES.md section 11) would another
        if infl:                                                 # stance's spotlight have written a different token?
            g_alt = torch.Generator().manual_seed(5151 + int(cfg["init_seed"]))
            traj_infl = np.full(T, np.nan, np.float32)

        def upper_peek(st_, h_in_, s_, sb_, take_=None):
            """The model's next-token logits for this token without changing its state (before upper_live)."""
            h_ = h_in_
            acts_ = Rm.band if act_l is None else act_l
            for l_ in range(Rm.band[0], Rm.L):
                off_ = offs_live(l_, s_) if l_ in Rm.band else None
                h_, _, _ = Rm.layer(l_, h_, st_["K"][l_], st_["V"][l_], st_["pos"], off_, None, sb_ if l_ in acts_ else None,
                                    take_ if l_ in acts_ else None)
            return Rm.model.lm_head(Rm.model.model.norm(h_))

        def upper_peek_read(st_, h_in_, s_, sb_):
            """The model's read-layer state for this token without changing its state (before upper_live)."""
            h_ = h_in_
            for l_ in range(Rm.band[0], Rm.read_layer + 1):
                off_ = offs_live(l_, s_) if l_ in Rm.band else None
                h_, _, _ = Rm.layer(l_, h_, st_["K"][l_], st_["V"][l_], st_["pos"], off_, None, sb_ if l_ in Rm.band else None)
            return h_
    sc = cfg.get("schema")
    if sc is not None:                                           # (WHERE_ITS_OWN_STATE_GOES.md section 13) its model of
        assert written and cfg.get("spotlight"), "the schema models what its spotlight does to the live model"
        sc = {"read": None, "lr": 1e-3, "lr_size": 3e-3, "hidden": 64, "warm": 128, "seed": 929, "past": 256, "dump": None} | sc
        sc_dump = {"x": [], "dh": [], "hin": [], "sel": []} if sc["dump"] else None   # (analysis) inputs, targets, the
                                                                 # new token, and what its attention selects
        assert sc["read"] in (None, "present", "past")
        d_ = Rm.d
        n_sc = m + f.R + d_                                      # where it stands, how it attends, what it read last
        gsc = torch.Generator().manual_seed(int(sc["seed"]))

        def sc_params():
            W1_ = (torch.randn(1, n_sc, sc["hidden"], generator=gsc) / math.sqrt(n_sc)).repeat(B, 1, 1).to(DEV).requires_grad_()
            ps_ = [W1_, torch.zeros(B, sc["hidden"], device=DEV, requires_grad=True),
                   torch.zeros(B, sc["hidden"], d_, device=DEV, requires_grad=True),   # it starts expecting no effect
                   torch.zeros(B, d_, device=DEV, requires_grad=True)]
            return ps_, torch.optim.Adam(ps_, lr=sc["lr"])
        sc_own, sc_opt = sc_params()
        sc_nb, sc_opt_nb = sc_params()                           # (comparison) the same, learned from the neighbour's

        def sz_params():                                         # how strongly its attending acts now (the size of the
            ps_ = [(torch.randn(1, m + 2 * f.R, 32, generator=gsc) / math.sqrt(m + 2 * f.R)).repeat(B, 1, 1).to(DEV).requires_grad_(),
                   torch.zeros(B, 32, device=DEV, requires_grad=True), torch.zeros(B, 32, 1, device=DEV, requires_grad=True),
                   torch.zeros(B, 1, device=DEV, requires_grad=True)]   # effect), from where it stands and how it attends
            return ps_, torch.optim.Adam(ps_, lr=sc["lr_size"])
        sz_own, sz_opt = sz_params()
        sz_nb, sz_opt_nb = sz_params()

        def sc_predict(ps_, x_):
            h_ = torch.nn.functional.gelu(torch.einsum("bi,bih->bh", x_, ps_[0]) + ps_[1])
            return torch.einsum("bh,bhd->bd", h_, ps_[2]) + ps_[3]
        sc_scale, sc_buf, sc_item, sz_rows = None, [], None, []
        gpast = torch.Generator().manual_seed(9393 + int(cfg["init_seed"]))
        sc_log = {k_: np.full(T, np.nan, np.float32) for k_ in ("sse", "sse_nb", "dh2", "item_att", "sz", "sz_hat", "sz_hat_nb")}
        sc_sum, sc_sum2, sc_n = torch.zeros(B, d_, device=DEV), torch.zeros(B, device=DEV), 0
    clo = cfg.get("closure", "loss")
    assert clo in ("loss", "measured", "use", "use_noise", "loss_describe", "know")
    kn = clo == "know"
    if kn:
        kn_cfg = {"move": "self", "state": "picture", "model_of": "own", "move_from": 400} | (cfg.get("know") or {})
        assert not cfg.get("trace"), "the first check runs without traces"
        for mod_ in f.field:                                     # its self-model starts knowing nothing of its acting
            if hasattr(mod_, "reset_parameters"):
                mod_.reset_parameters()
    body = cfg.get("body")
    if body is not None:
        body = {"strength": 0.8, "time": 100.0} | body
        U = torch.zeros(B, m, m, device=DEV)
        traj_U = np.zeros((T, B, m, m), np.float16)
    eye = torch.eye(m, device=DEV)
    use_from = int(cfg.get("use_from", 400))
    gnoise = torch.Generator(device=DEV).manual_seed(6060 + cfg["init_seed"])
    S_fixed = None
    tr = cfg.get("trace")
    Qh, tr_rate = None, None
    if tr:
        assert cfg.get("picture") == "sphere" and cfg.get("reader_route", "operation") == "operation"
        tr = {"fade": 0.002, "strength": 0.5, "from": "own"} | tr
        Qh = torch.zeros(B, f.rh, f.rq_bias.shape[0] // f.rh, m, device=DEV)
    hb = cfg.get("habit")
    if hb is not None:                                           # (DESIGN_A_HISTORY_OF_ITS_OWN.md, direction 1)
        assert not tr and cfg.get("picture") == "sphere" and cfg.get("reader_route", "operation") == "operation"
        hb = {"size": 0.5, "fill": 1000.0, "from": "own"} | hb
        Qh = torch.zeros(B, f.rh, f.rq_bias.shape[0] // f.rh, m, device=DEV)
        hb_cap, hb_rate = float(hb["size"] * f.rq.weight.detach().norm()), None
    mv = cfg.get("moved")
    if mv is not None:                                           # (direction 2)
        mv = {"angle": 0.1, "turn": "own"} | mv
    tn = cfg.get("turned")
    if tn is not None:                                           # (WHERE_ITS_OWN_STATE_GOES.md (b))
        tn = {"turn": "own"} | tn
        assert tn["turn"] in ("own", "fixed")
        assert mv is None and not tr and hb is None and (not kn or kn_cfg["move"] == "reading")
    g_prev, inv0 = None, None                                    # (turned) where its reading pointed last; its invariant
    sm = cfg.get("selfmodel")
    P_prev = None                                                # (self-model) its last picture
    if sm is not None:                                           # (WHERE_ITS_OWN_STATE_GOES.md section 9)
        assert tn is not None and tn["turn"] == "own"
        sm = {"hidden": 64, "lr": 1e-4, "from": "own", "swap_at": None, "shift_at": None, "seed": 919} | sm
        assert sm["from"] in ("own", "neighbor")
        n_in = cfg["p"] + m                                      # every copy's self-model is born the same
        gsm = torch.Generator().manual_seed(int(sm["seed"]))
        sm_W1 = (torch.randn(1, n_in, sm["hidden"], generator=gsm) / math.sqrt(n_in)).repeat(B, 1, 1).to(DEV).requires_grad_()
        sm_b1 = torch.zeros(B, sm["hidden"], device=DEV, requires_grad=True)
        sm_W2 = torch.zeros(B, sm["hidden"], m, device=DEV, requires_grad=True)   # so it starts as persistence
        sm_b2 = torch.zeros(B, m, device=DEV, requires_grad=True)
        sm_params = [sm_W1, sm_b1, sm_W2, sm_b2]
        sm_opt = torch.optim.Adam(sm_params, lr=sm["lr"])

    def sm_predict(x, gp_, bi):
        """(self-model) where copy bi's self-model expects its reading to point, from its last picture and its stance
        (x): persistence (gp_, its reading's last destination) plus its learned correction."""
        h_ = torch.nn.functional.gelu(torch.einsum("ni,nih->nh", x, sm_W1[bi]) + sm_b1[bi])
        return unit(gp_ + torch.einsum("nh,nho->no", h_, sm_W2[bi]) + sm_b2[bi])
    fut = cfg.get("futures")
    if fut is not None:                                          # how much of its present reaches its future
        assert not written
        fut = {"every": 100, "horizon": 32, "last": 1500, "interaction": False} | fut
        S_fut = unit(torch.randn(8, m, generator=torch.Generator().manual_seed(4242))).to(DEV)
        fut_num, fut_den, fut_places = 0.0, 0.0, []
        if fut["interaction"]:                                   # one head's part of its own stance changed at a time
            per_h = m // f.rh
            n_var = 6
            Z_int = torch.randn(f.rh, n_var, per_h, generator=torch.Generator().manual_seed(4343)).to(DEV)
            Z_int = Z_int / Z_int.norm(dim=-1, keepdim=True)
            int_num, int_den = 0.0, 0.0
    rd = cfg.get("reader")
    if rd is not None:                                           # (DESIGN_A_READER_OF_ITS_OWN.md)
        rd = {"contrast": 3.0, "time": 50.0, "from": "own", "stride": 16, "stance_q": True, "gain": 1.0} | rd
        assert rd["kind"] in ("particular", "adapted", "wide", "heads", "gain")
        assert not tr and hb is None and mv is None and body is None and not written
        assert cfg.get("picture") == "sphere" and cfg.get("reader_route", "operation") == "operation"
        assert not kn or kn_cfg["state"] == "picture"
    E = None                                                     # (adapted) each copy's expectation of what it reads
    L_keep = (f.R - 1) * rd["stride"] + 1 if (rd is not None and rd["kind"] == "wide") else f.R
    sp = cfg.get("stance_path")
    if sp is not None:                                           # where its stance's influence is lost, in the life
        sp = {"last": 1000, "every": 13} | sp
        S_sp = unit(torch.randn(12, m, generator=torch.Generator().manual_seed(5))).to(DEV)
        sp_tv, sp_num, sp_base = [], {k_: [] for k_ in ("raw", "read", "picture", "next")}, {k_: [] for k_ in ("raw", "read", "picture", "next")}

    def rollout(rows, b_idx, t0, horizon):
        """Rows (N, m) of copies b_idx live on `horizon` tokens of their copy's coming text by its present rule (the
        reader's expectation renewed by each row's own reading, or the neighbour's own course's; nothing written)."""
        N0 = rows.shape[0]
        rows = torch.cat([rows, s.detach()], 0)                  # the copies' own courses ride along (the neighbour's
        b_idx = torch.cat([b_idx, torch.arange(B, device=DEV)])  # reading for the control's expectation)
        hr = hist[b_idx]
        Er = None if E is None else E[b_idx].clone()
        gp = None if (tn is None or g_prev is None) else g_prev[b_idx]   # (turned) each row keeps its copy's last
        Pp = None if (sm is None or P_prev is None) else P_prev[b_idx]   # reading destination (and last picture)
        for h_ in range(horizon):
            hr = torch.cat([hr, H[b_idx, t0 + h_:t0 + h_ + 1]], 1)[:, -L_keep:]
            ra = rows if G is None else torch.einsum("nij,nj->ni", G[b_idx], rows)
            if rd is not None:
                P_, _, _, raw_, _ = rd_read(f, rd, ra, window(f, hr, rd), Er)
                if rd["kind"] == "adapted":
                    src = raw_ if rd["from"] == "own" else raw_[N0 + (b_idx + 1) % B]
                    Er = src.clone() if Er is None else (1.0 - 1.0 / rd["time"]) * Er + src / rd["time"]
                g_ = f.next_stance(P_)
            else:
                P_ = f.read(ra, hr) if Qh is None else read_trace(f, ra, hr, Qh[b_idx])[0]
                g_ = f.next_stance(P_)
            if mv is not None:
                if mv["turn"] == "fixed":
                    g_ = f.next_stance(f.read(s_fix[b_idx], hr))
                g_ = turn(rows, g_, mv["angle"])
            if tn is not None:
                g_t = g_ if tn["turn"] == "own" else f.next_stance(f.read(s_fix[b_idx], hr))
                if gp is None:
                    g_ = rows
                elif sm is not None and Pp is not None:          # turned by what its self-model did not foresee
                    g_ = carry(sm_predict(torch.cat([Pp, rows], -1), gp, b_idx), g_t, rows)
                else:
                    g_ = carry(gp, g_t, rows)
                gp, Pp = g_t, (P_ if sm is not None else None)
            rows = g_
        return rows[:N0]
    opt = torch.optim.Adam(f.field.parameters() if kn else f.parameters(), lr=cfg["lr"])
    pg = torch.Generator().manual_seed(int(cfg.get("probe_seed", 77)))   # loop12's generator; per life for a range of lives
    s = unit(torch.randn(B, m, generator=torch.Generator().manual_seed(cfg["init_seed"]))).to(DEV)
    if cfg.get("same_start"):                                    # every copy from the same first stance
        s = s[:1].repeat(B, 1)
    s_fix = s.clone()                                            # (control of direction 2) its first stance, kept
    hist = torch.zeros(B, 0, H.shape[-1], device=DEV)
    calib, chunk = cfg.get("calib_tokens", 150), cfg["chunk"]
    adopt_n = int(cfg.get("adopt_tokens", 50)) if (clo in ("measured", "use", "use_noise") and cfg.get("init_iface")) else 0
    adopt_num, adopt_den, head_scale, rec_scale, lam1, calib_sum = 0.0, 0.0, None, None, 1.0, 0.0
    traj = {k: np.zeros((T, B, m), np.float32) for k in ("s",)}
    traj_P = np.zeros((T, B, cfg["p"]), np.float32)
    log = {k: np.zeros(T, np.float32) for k in ("acc", "world_dep", "L_rec", "own_meas", "own_desc", "use_dev", "trace_size",
                                                "trace_apart", "lp_pull", "lp_prog", "lp_err", "body_gain", "self_dev",
                                                "read_step", "model_err", "probe_err", "att_apart", "turn_inv", "turn_angle",
                                                "sm_err", "sm_pers")}
    if kn:
        traj_read, traj_self = np.zeros((T, B, m), np.float32), np.zeros((T, B, m), np.float32)
    lp = cfg.get("progress")
    if lp is not None:                                           # (DESIGN_COMING_TO_KNOW_ITSELF.md); {} means the defaults
        assert clo == "measured" and cfg.get("fixed_probes"), "learning progress is defined on the measured description at fixed stances"
        import copy
        lp = {"slow": 200.0, "avg": 50.0, "from": "own"} | lp
        slow_field = copy.deepcopy(f.field)                      # who it took itself to be a while ago
        for p_ in slow_field.parameters():
            p_.requires_grad_(False)
        lp_prog, lp_err = torch.zeros(B, K, device=DEV), torch.zeros(B, K, device=DEV)
        e_fast, e_slow = torch.zeros(B, K, device=DEV), torch.zeros(B, K, device=DEV)
        traj_lp = np.zeros((T, B, K), np.float16)
    chunk_loss, t0 = 0.0, time.time()
    for t in range(T):
        if written:
            with torch.no_grad():
                h_in = Rm.lower(st_m, cur)
                sb, take_, p_att = emphasis(s.detach(), hist)
                if sc is not None and sb is not None:            # its schema expects an effect, before it happens
                    p16 = torch.zeros(B, f.R, device=DEV)
                    p16[:, -p_att.shape[1]:] = p_att
                    x_sc = torch.cat([s.detach(), p16, f.rln(hist[:, -1]).detach()], -1)
                    with torch.no_grad():
                        sc_item = sc_predict(sc_own, x_sc)
                    h_wo = upper_peek_read(st_m, h_in, s.detach(), None)[:, 0].float()   # without its spotlight
                logits_alt = None
                if infl and ((t + 1) % (sw["prompt"] + sw["write"])) >= sw["prompt"]:   # the next token will be written
                    s_alt = unit(torch.randn(B, m, generator=g_alt)).to(DEV)
                    logits_alt = upper_peek(st_m, h_in, s_alt, *emphasis(s_alt, hist)[:2])
                h_read, logits_w = Rm.upper_live(st_m, h_in, offs_live, s.detach(), sb=sb, take=take_, act=act_l)
                if ll and p_att is not None:                     # (section 22) what its own selection takes up, against
                    hw_ = hist[:, -p_att.shape[1]:]              # the other copies'; and the kinds its model writes
                    a_ = torch.einsum("bl,bld->bd", p_att, hw_)
                    ll_vec[t // ll_seg] += (a_ - a_.mean(0, keepdim=True)) / a_.pow(2).mean(-1, keepdim=True).sqrt()
                    if ((t + 1) % (sw["prompt"] + sw["write"])) >= sw["prompt"]:   # the next token will be written
                        lp_ = torch.log_softmax(logits_w[:, 0].float(), -1)
                        ll_write[t // ll_seg] += torch.stack([torch.logsumexp(lp_[:, ix], -1) for ix in ll_sets], -1)
                        ll_n[t // ll_seg] += 1
            h_t = h_read.float()
            if sc is not None and sb is not None:                # it learns what its attending did, only to be true
                dh = h_t[:, 0] - h_wo                            # (the counterfactual: a training target, never an input)
                dh2 = (dh ** 2).sum(-1)
                sc_scale = dh2.mean().clamp_min(1e-8) if sc_scale is None else 0.99 * sc_scale + 0.01 * dh2.mean().clamp_min(1e-8)
                with torch.enable_grad():
                    for ps_, op_, xx_, yy_ in ((sc_own, sc_opt, x_sc, dh), (sc_nb, sc_opt_nb, torch.roll(x_sc, -1, 0), torch.roll(dh, -1, 0))):
                        loss_ = ((sc_predict(ps_, xx_) - yy_) ** 2).sum(-1).mean() / sc_scale
                        op_.zero_grad()
                        loss_.backward()
                        op_.step()
                y_sz = torch.log(dh2.sqrt() + 1e-6)[:, None]       # how strongly its attending acted
                x_sz = torch.cat([x_sc[:, :m + f.R], torch.log(x_sc[:, m:m + f.R] + 1e-6)], -1)   # its attention, and its log
                if not sz_rows:                                  # its first expectation: the size it first met (all copies)
                    with torch.no_grad():
                        sz_own[3].fill_(float(y_sz.mean()))
                        sz_nb[3].fill_(float(y_sz.mean()))
                with torch.no_grad():
                    sz_hat, sz_hat_nb = sc_predict(sz_own, x_sz), sc_predict(sz_nb, x_sz)   # before learning from it
                with torch.enable_grad():
                    for ps_, op_, xx_, yy_ in ((sz_own, sz_opt, x_sz, y_sz), (sz_nb, sz_opt_nb, torch.roll(x_sz, -1, 0), torch.roll(y_sz, -1, 0))):
                        loss_ = ((sc_predict(ps_, xx_) - yy_) ** 2).mean()
                        op_.zero_grad()
                        loss_.backward()
                        op_.step()
                sc_log["sz"][t], sc_log["sz_hat"][t], sc_log["sz_hat_nb"][t] = float(y_sz.mean()), float(sz_hat.mean()), float(sz_hat_nb.mean())
                sz_rows.append(torch.cat([y_sz, sz_hat, sz_hat_nb], -1).cpu().numpy())
                with torch.no_grad():
                    sc_log["sse"][t] = float(((sc_item - dh) ** 2).sum(-1).mean())
                    sc_log["sse_nb"][t] = float(((sc_predict(sc_nb, x_sc) - dh) ** 2).sum(-1).mean())
                    sc_log["dh2"][t] = float(dh2.mean())
                    if t >= T // 2:
                        sc_sum += dh
                        sc_sum2 += dh2
                        sc_n += 1
                    sc_buf = (sc_buf + [sc_item.clone()])[-sc["past"]:]
                    if sc_dump is not None:
                        sc_dump["x"].append(x_sc.half().cpu().numpy())
                        sc_dump["dh"].append(dh.half().cpu().numpy())
                        sc_dump["hin"].append(h_in[:, 0].float().half().cpu().numpy())
                        hw_ = hist[:, -p_att.shape[1]:]
                        sc_dump["sel"].append((torch.einsum("bl,bld->bd", p_att, hw_) - hw_.mean(1)).half().cpu().numpy())
        else:
            h_t = H[:, t:t + 1]
        with torch.no_grad():
            S_r = unit(torch.randn(K, m, generator=pg)).to(DEV)
            if cfg.get("fixed_probes"):
                S_fixed = S_r if S_fixed is None else S_fixed
                S_r = S_fixed
            S = torch.cat([S_r, s.detach()], 0)
        G = (eye - body["strength"] * U) if body is not None else None   # its body: how its stance reaches its reader now
        if fut is not None and t >= T - fut["last"] and t % fut["every"] == 0 and t + fut["horizon"] <= T:
            with torch.no_grad():                                # every stance lives on by the copy's present rule
                Sf = torch.cat([S_fut, s.detach()], 0)
                NS = Sf.shape[0]
                b_idx = torch.arange(B, device=DEV).repeat_interleave(NS)
                rows = rollout(Sf.repeat(B, 1), b_idx, t, fut["horizon"])
                D_ = rows.view(B, NS, m)
                if fut["interaction"]:                           # one head's part changed: do the others' futures move?
                    X0 = s.detach()[:, None, None].repeat(1, f.rh, n_var, 1)          # (B, heads, variants, m)
                    for h_ in range(f.rh):
                        sl = slice(h_ * per_h, (h_ + 1) * per_h)
                        X0[:, h_, :, sl] = Z_int[h_][None] * s.detach()[:, None, sl].norm(dim=-1, keepdim=True)
                    bi_ = torch.arange(B, device=DEV).repeat_interleave(f.rh * n_var)
                    F_ = rollout(X0.reshape(-1, m), bi_, t, fut["horizon"]).view(B, f.rh, n_var, m)
                    for h_ in range(f.rh):
                        oth = [i_ for i_ in range(m) if not (h_ * per_h <= i_ < (h_ + 1) * per_h)]
                        Fo, Do = F_[:, h_][..., oth], D_[..., oth]
                        int_num += float(((Fo - Fo.mean(1, keepdim=True)) ** 2).sum(-1).mean(1).sum())
                        int_den += float(((Do - Do.mean(1, keepdim=True)) ** 2).sum(-1).mean(1).sum())
                fut_num += float(((D_ - D_.mean(1, keepdim=True)) ** 2).sum())
                fut_den += B * float(((Sf - Sf.mean(0, keepdim=True)) ** 2).sum())
                for b in range(B):
                    first = []
                    for x in D_[b]:
                        if not first or max(float(x @ y) for y in first) < 0.9:
                            first.append(x)
                    fut_places.append(len(first))
        with torch.set_grad_enabled(not kn):                     # (know) nothing learned reaches how it reads
            Dp = field(f, S, hist, h_t, B, Qh, G, rd, E)
            if kn and kn_cfg["model_of"] == "neighbor":         # (control) its neighbour's body, on its own text
                Dp_o = field(f, S, hist, h_t, B, Qh, torch.roll(G, -1, 0))
        Dt = Dp if clo in ("loss", "loss_describe") else Dp.detach()   # closure through the loss | measured
        hist = torch.cat([hist, h_t], 1)[:, -L_keep:]
        sa = s if G is None else torch.einsum("bij,bj->bi", G, s)
        if sp is not None and t >= T - sp["last"] and t % sp["every"] == 0:
            with torch.no_grad():                                # 12 stances read the copy's present window
                bi_ = torch.arange(B, device=DEV).repeat_interleave(12)
                P_, a_, _, raw_s, r_s = rd_read(f, rd, S_sp.repeat(B, 1), window(f, hist, rd)[bi_], None if E is None else E[bi_])
                A_ = a_.view(B, 12, f.rh, -1)
                tv_ = 0.5 * (A_[:, :, None] - A_[:, None]).abs().sum(-1).mean(-1)
                sp_tv.append(float(tv_.sum() / (B * 12 * 11)))
                for key_, x_ in (("raw", raw_s.flatten(1)), ("read", r_s.flatten(1)), ("picture", P_), ("next", f.next_stance(P_))):
                    X_ = x_.reshape(B, 12, -1)
                    sp_num[key_].append(float(((X_ - X_.mean(1, keepdim=True)) ** 2).sum(-1).mean()))
                    sp_base[key_].append(X_[:, 0].cpu())
        if rd is not None:
            with torch.set_grad_enabled(not kn):
                P, att_, k_, raw_, _ = rd_read(f, rd, sa, window(f, hist, rd), E)
            if rd["kind"] == "adapted":                          # what it is used to: its own reading, lately
                with torch.no_grad():
                    src = raw_.detach() if rd["from"] == "own" else torch.roll(raw_.detach(), -1, 0)
                    E = src.clone() if E is None else (1.0 - 1.0 / rd["time"]) * E + src / rd["time"]
        elif Qh is None:
            if sc is not None and sc["read"] and t >= sc["warm"] and sc_item is not None:   # (section 13) it reads its
                if sc["read"] == "present":                      # schema of what its attending does beside its window
                    item_ = sc_item
                else:                                            # (control) a past prediction: the same kind of item,
                    j_ = int(torch.randint(max(1, len(sc_buf) - 1), (1,), generator=gpast))   # not about now
                    item_ = sc_buf[j_]
                with torch.set_grad_enabled(not kn):
                    P, att_ = read_item(f, sa, hist, item_)
                sc_log["item_att"][t] = float(att_[..., -1].mean())
            else:
                with torch.set_grad_enabled(not kn):
                    P = f.read(sa, hist)
                att_ = getattr(f, "_last_att", None)
        else:
            with torch.set_grad_enabled(not kn):
                P, att_, k_ = read_trace(f, sa, hist, Qh)
        if hb is not None and hist.shape[1] == f.R:              # what was particular in what it attended, tied to the
            with torch.no_grad():                                # stance that reached its reader, becomes a habit (from
                                                                 # a full window: in a window of one, nothing is particular)
                kap = torch.einsum("nhl,nlhd->nhd", att_, k_) - k_.mean(1)   # centred: what is everywhere is not written
                s_d = sa.detach()
                if hb["from"] == "neighbor":
                    kap, s_d = torch.roll(kap, -1, 0), torch.roll(s_d, -1, 0)
                w_ = torch.einsum("nhd,nm->nhdm", kap, s_d)
                if hb_rate is None:                              # writes held to one direction fill it in `fill` tokens
                    hb_rate = float(hb_cap / (hb["fill"] * w_.flatten(1).norm(dim=-1).mean().clamp_min(1e-8)))
                Qh = Qh + hb_rate * w_
                Qh = Qh * (hb_cap / Qh.flatten(1).norm(dim=-1).clamp_min(1e-12)).clamp(max=1.0).view(B, 1, 1, 1)   # of fixed size
                log["trace_size"][t] = float(torch.einsum("nhdm,nm->nhd", Qh, sa.detach()).flatten(1).norm(dim=-1).mean()
                                             / f.rq(sa.detach()).norm(dim=-1).mean().clamp_min(1e-8))
                Qf = Qh.flatten(1)
                log["trace_apart"][t] = float(torch.cdist(Qf, Qf).sum() / (B * (B - 1)) / Qf.norm(dim=-1).mean().clamp_min(1e-8))
        elif tr:
            with torch.no_grad():                                # what it attended, from where it stood, leaves a trace
                kap = torch.einsum("nhl,nlhd->nhd", att_, k_)
                s_d = s.detach()
                if tr["from"] == "neighbor":
                    kap, s_d = torch.roll(kap, -1, 0), torch.roll(s_d, -1, 0)
                if tr_rate is None:                              # strength: the trace's size at equilibrium vs the query's
                    q_size = f.rq(s_d).norm(dim=-1).mean()
                    tr_rate = float(tr["strength"] * tr["fade"] * q_size / kap.flatten(1).norm(dim=-1).mean().clamp_min(1e-8))
                Qh = (1.0 - tr["fade"]) * Qh + tr_rate * torch.einsum("nhd,nm->nhdm", kap, s_d)   # a new tensor: earlier reads keep theirs
                log["trace_size"][t] = float(torch.einsum("nhdm,nm->nhd", Qh, s.detach()).flatten(1).norm(dim=-1).mean()
                                             / f.rq(s.detach()).norm(dim=-1).mean().clamp_min(1e-8))
                Qf = Qh.flatten(1)
                log["trace_apart"][t] = float(torch.cdist(Qf, Qf).sum() / (B * (B - 1)) / Qf.norm(dim=-1).mean().clamp_min(1e-8))
        if att_ is not None and B > 1:                           # how differently the copies read their windows
            with torch.no_grad():
                A_ = att_.detach()
                log["att_apart"][t] = float((0.5 * (A_[:, None] - A_[None]).abs().sum(-1).mean(-1)).sum() / (B * (B - 1)))
        with torch.set_grad_enabled(not kn):
            s_next = f.next_stance(P)
        dest = Dt + S[None]
        Dt_cmp = dest - dest.mean(1, keepdim=True)
        if clo != "loss":
            with torch.no_grad():
                ms_ = (Dt_cmp ** 2).sum(-1).mean(1)
                rec_scale = ms_.clone() if rec_scale is None else 0.99 * rec_scale + 0.01 * ms_
        L_rec, accs, own_rows = 0.0, [], []
        if kn:                                                   # its self-model learns, only to be true, the whole of
            with torch.no_grad():                                # what its acting does: where it goes from each stance
                st_ = kn_cfg["state"]
                P_in = P if st_ == "picture" else (torch.zeros_like(P) if st_ == "zero" else f.read(s, hist))
                tgt = (Dp_o if kn_cfg["model_of"] == "neighbor" else Dp) + S[None]
            NS, ar = S.shape[0], torch.arange(B, device=DEV)
            pd_ = f.field(torch.cat([P_in[:, None].expand(-1, NS, -1), S[None].expand(B, -1, -1)], -1)) + S[None]
            L_rec = ((pd_ - tgt) ** 2).sum(-1).mean()
            with torch.no_grad():
                pc_ = pd_ - pd_.mean(1, keepdim=True)
                accs = [contrastive(pc_[b], Dt_cmp, b)[1] for b in range(B)]
                own_rows = list(pc_[ar, K + ar])
                s_self = unit(pd_[ar, K + ar])                   # where its self-model says it goes from where it stands
                log["self_dev"][t] = float((s_self - s_next).norm(dim=-1).mean())
                log["read_step"][t] = float((s_next - s).norm(dim=-1).mean())
                log["model_err"][t] = float((pd_[ar, K + ar] - tgt[ar, K + ar]).norm(dim=-1).mean())
                log["probe_err"][t] = float((pd_[:, :K] - tgt[:, :K]).norm(dim=-1).mean())
                traj_read[t], traj_self[t] = s_next.cpu().numpy(), s_self.cpu().numpy()
            if kn_cfg["move"] == "self" and t >= kn_cfg["move_from"]:
                s_next = s_self                                  # it moves by what it knows of itself
        if mv is not None:                                       # (direction 2) moved toward where its reading points
            with torch.no_grad():
                g_ = s_next.detach() if mv["turn"] == "own" else f.next_stance(f.read(s_fix, hist))
                s_next = turn(s.detach(), g_, mv["angle"])
        if tn is not None:                                       # ((b)) turned by its reading's motion, never replaced
            with torch.no_grad():
                g_t = s_next.detach() if tn["turn"] == "own" else f.next_stance(f.read(s_fix, hist))
                if g_prev is None:                               # first token: it stays; its angle to this reading is
                    inv0 = (s.detach() * g_t).sum(-1)            # what the turning keeps from now on
                    s_next = s.detach()
                elif sm is not None and P_prev is not None:      # (section 9) turned by what it did not foresee of itself
                    ar_ = torch.arange(B, device=DEV)
                    if sm["swap_at"] is not None and t == sm["swap_at"]:      # copies 0 and 1 exchange self-models
                        for p_ in sm_params:
                            p_[[0, 1]] = p_[[1, 0]].clone()
                            st_ = sm_opt.state.get(p_, {})
                            for k_ in ("exp_avg", "exp_avg_sq"):
                                if k_ in st_:
                                    st_[k_][[0, 1]] = st_[k_][[1, 0]].clone()
                    if sm["shift_at"] is not None and t == sm["shift_at"]:    # (control) a random change of that size
                        gsh = torch.Generator().manual_seed(4545)
                        for p_ in sm_params:
                            dn = (p_[0] - p_[1]).norm()
                            for b_ in (0, 1):
                                r_ = torch.randn(p_[b_].shape, generator=gsh).to(DEV)
                                p_[b_] += r_ * dn / r_.norm().clamp_min(1e-12)
                    x_in = torch.cat([P_prev, s.detach()], -1)
                    with torch.enable_grad():                    # it predicts before the token, then learns the truth
                        g_hat = sm_predict(x_in, g_prev, ar_)
                        if sm["from"] == "own":
                            loss_sm = ((g_hat - g_t) ** 2).sum()
                        else:                                    # (control) learned from its neighbour's experience
                            g_n = sm_predict(torch.roll(x_in, -1, 0), torch.roll(g_prev, -1, 0), ar_)
                            loss_sm = ((g_n - torch.roll(g_t, -1, 0)) ** 2).sum()
                        sm_opt.zero_grad()
                        loss_sm.backward()
                        sm_opt.step()
                    g_hat = g_hat.detach()
                    log["sm_err"][t] = float(((g_hat - g_t) ** 2).sum(-1).mean())
                    log["sm_pers"][t] = float(((g_prev - g_t) ** 2).sum(-1).mean())
                    log["turn_angle"][t] = float(torch.arccos((g_hat * g_t).sum(-1).clamp(-1.0, 1.0)).mean())
                    s_next = carry(g_hat, g_t, s.detach())
                else:
                    log["turn_inv"][t] = float(((s.detach() * g_prev).sum(-1) - inv0).abs().max())
                    log["turn_angle"][t] = float(torch.arccos((g_prev * g_t).sum(-1).clamp(-1.0, 1.0)).mean())
                    s_next = carry(g_prev, g_t, s.detach())
                g_prev = g_t
                if sm is not None:
                    P_prev = P.detach().clone()
        for b in range(0 if kn else B):
            pf = f.predict_field(P[b:b + 1], S)
            pd_ = pf + S
            pf = pd_ - pd_.mean(0, keepdim=True)
            if clo == "loss":
                l_, a_ = contrastive(pf, Dt_cmp, b)
            elif clo == "loss_describe":
                u_, v_ = pf.flatten(), Dt_cmp[b].flatten()
                l_ = 1.0 - (u_ * v_).sum() / (u_.norm() * v_.norm()).clamp_min(1e-8)
                with torch.no_grad():
                    a_ = contrastive(pf, Dt_cmp, b)[1]
            else:
                tb = (b + 1) % B if cfg.get("describe") == "neighbor" else b   # (control) it describes its neighbour's field
                l_ = ((pf - Dt_cmp[tb]) ** 2).sum(-1).mean() / rec_scale[tb].clamp_min(1e-8)
                with torch.no_grad():
                    a_ = contrastive(pf, Dt_cmp, b)[1]
                    own_rows.append(pf[K + b])
                    if t < adopt_n:
                        Sc = S - S.mean(0, keepdim=True)
                        u_, v_ = pf - Sc, Dt_cmp[tb] - Sc
                        adopt_num, adopt_den = adopt_num + float((u_ * v_).sum()), adopt_den + float((u_ * u_).sum())
                    if lp:                                       # the present and the slow description at the fixed stances
                        ps_ = slow_field(torch.cat([P[b:b + 1].detach().expand(S.shape[0], -1), S], -1)) + S
                        ps_ = ps_ - ps_.mean(0, keepdim=True)
                        sc_ = rec_scale[tb].clamp_min(1e-8)
                        e_fast[b] = ((pf[:K] - Dt_cmp[tb, :K]) ** 2).sum(-1) / sc_
                        e_slow[b] = ((ps_[:K] - Dt_cmp[tb, :K]) ** 2).sum(-1) / sc_
            L_rec, accs = L_rec + l_ / B, accs + [a_]
        if lp:                                                   # drawn to where its picture of itself is getting better
            with torch.no_grad():
                a_lp = 1.0 / lp["avg"]
                lp_prog.mul_(1.0 - a_lp).add_(a_lp * (e_slow - e_fast))
                lp_err.mul_(1.0 - a_lp).add_(a_lp * e_slow)
                w_ = (lp_prog.clamp_min(0.0) / lp_err.clamp_min(1e-8)).clamp(0.0, 1.0)   # share of the slow one's error removed
                if lp["from"] == "neighbor":                     # (control) drawn by the neighbour's coming to know itself
                    w_ = torch.roll(w_, -1, 0)
                lam = w_.mean(1, keepdim=True)                   # how much it is coming to know itself, overall (0..1)
                g_ = unit(torch.einsum("bk,km->bm", w_, S_r) + 1e-8)   # toward the stances in proportion to their progress
                bent = unit((1.0 - lam) * s_next.detach() + lam * g_)
                log["lp_pull"][t] = float((bent - unit(s_next.detach())).norm(dim=-1).mean())
                log["lp_prog"][t] = float(lp_prog.clamp_min(0.0).mean())
                log["lp_err"][t] = float(lp_err.mean())
                traj_lp[t] = w_.cpu().numpy()
                for p_s, p_f in zip(slow_field.parameters(), f.field.parameters()):
                    p_s.mul_(1.0 - 1.0 / lp["slow"]).add_(p_f.detach(), alpha=1.0 / lp["slow"])
            s_next = unit((1.0 - lam) * s_next + lam * g_)
        if clo in ("use", "use_noise") and t >= use_from:      # it moves by its own description of its own part
            with torch.no_grad():
                z_use = dest.mean(1) + torch.stack(own_rows)
                if clo == "use_noise":
                    dev_ = z_use - s_next.detach()
                    z_use = s_next.detach() + unit(torch.randn(dev_.shape, generator=gnoise, device=DEV)) * dev_.norm(dim=-1, keepdim=True)
                s_use = unit(z_use)
                log["use_dev"][t] = float((s_use - s_next.detach()).norm(dim=-1).mean())
            s_next = s_use
        if adopt_n and t == adopt_n - 1:
            head_scale = adopt_num / max(adopt_den, 1e-12)
            with torch.no_grad():
                f.field[2].weight.mul_(head_scale)
                f.field[2].bias.mul_(head_scale)
                if lp:                                           # the slow description starts from the adopted head
                    slow_field.load_state_dict(f.field.state_dict())
                    lp_prog.zero_()
                    lp_err.zero_()
        if t < calib:
            calib_sum += float(L_rec) if t >= adopt_n else 0.0
            if t == calib - 1:
                lam1 = (calib - adopt_n) / max(calib_sum, 1e-6)
                json.dump(dict(cfg=cfg, loss_weight=lam1, head_scale=head_scale, trace_rate=tr_rate), open(os.path.join(out, "config.json"), "w"), indent=1)
        chunk_loss = chunk_loss + (lam1 * L_rec if t >= calib else 0.0 * L_rec)
        traj["s"][t], traj_P[t] = s.detach().cpu().numpy(), P.detach().cpu().numpy()
        with torch.no_grad():
            log["acc"][t] = float(np.mean(accs))
            log["world_dep"][t] = float((Dp - Dp.mean(0, keepdim=True)).norm(dim=-1).mean())
            log["L_rec"][t] = float(L_rec)
            log["own_meas"][t] = float(torch.stack([Dt_cmp[b, K + b] for b in range(B)]).norm(dim=-1).mean())
            if own_rows:
                log["own_desc"][t] = float(torch.stack(own_rows).norm(dim=-1).mean())
        if written:                                              # the next token: real text, or what the model writes
            with torch.no_grad():
                per = sw["prompt"] + sw["write"]
                traj_tok[t], traj_written[t] = cur.cpu().numpy(), (t % per) >= sw["prompt"]
                if ((t + 1) % per) < sw["prompt"]:
                    cur = ids_w[torch.arange(B, device=DEV), ptr.clamp_max(ids_w.shape[1] - 1)]
                    ptr = ptr + 1
                else:
                    u_ = torch.rand(1 if cfg.get("shared_noise") else B, generator=gsamp, device=DEV).expand(B)
                    cur = sample_shared(logits_w[:, 0], sw, u_)
                    if logits_alt is not None:                   # the same random number, another stance's spotlight
                        traj_infl[t] = float((sample_shared(logits_alt[:, 0], sw, u_) != cur).float().mean())
        if body is not None:                                     # use tires its body along the way it acted
            with torch.no_grad():
                traj_U[t] = U.cpu().numpy()
                log["body_gain"][t] = float(sa.norm(dim=-1).mean())
                U = (1.0 - 1.0 / body["time"]) * U + torch.einsum("bi,bj->bij", s.detach(), s.detach()) / body["time"]
        s = s_next
        if (t + 1) % chunk == 0:
            if t >= calib:
                opt.zero_grad()
                chunk_loss.backward()
                torch.nn.utils.clip_grad_norm_(f.parameters(), 1.0)
                opt.step()
            chunk_loss = 0.0
            s = s.detach()
            hist = hist.detach()
        if (t + 1) % cfg.get("print_every", 500) == 0:
            w = slice(t + 1 - cfg.get("print_every", 500), t + 1)
            st_ = traj["s"][w]
            flips = float(((st_[1:] * st_[:-1]).sum(-1) < 0).mean())
            print(f"[core {clo} s{cfg['init_seed']}] t={t + 1} {time.time() - t0:.0f}s acc {log['acc'][w].mean():.2f} "
                  f"flips {flips:.2f} worldDep {log['world_dep'][w].mean():.3f} own {log['own_meas'][w].mean():.3f}"
                  + (f" pull {log['lp_pull'][w].mean():.3f} progress {log['lp_prog'][w].mean():.4f}" if lp else "")
                  + (f" gain {log['body_gain'][w].mean():.2f}" if body is not None else "")
                  + (f" readStep {log['read_step'][w].mean():.3f} selfDev {log['self_dev'][w].mean():.3f} "
                     f"modelErr {log['model_err'][w].mean():.3f} probeErr {log['probe_err'][w].mean():.3f}" if kn else ""), flush=True)
    torch.save(f.state_dict(), os.path.join(out, f"iface_{T}.pt"))
    if Qh is not None:
        torch.save(Qh.cpu(), os.path.join(out, f"trace_{T}.pt"))
    extra = dict(tok=traj_tok, written=traj_written) if written else {}
    if written and infl:
        extra["infl"] = traj_infl
    if written and ll:                                           # (section 22) through the model's own output map
        with torch.no_grad():
            kl = kind_lens(Rm.model.lm_head.weight, Rm.model.model.norm.weight, Rm.tok)
            half = n_seg // 2
            extra.update(lens_kind_seg=kl(ll_vec.reshape(-1, Rm.d)).view(n_seg, B, -1).cpu().numpy(),
                         lens_kind_late=kl(ll_vec[half:].sum(0)).cpu().numpy(),
                         lens_kind_early=kl(ll_vec[1:half].sum(0)).cpu().numpy(),
                         write_kind_seg=(ll_write / torch.tensor(np.maximum(ll_n, 1), device=DEV).view(-1, 1, 1)).cpu().numpy(),
                         write_n=ll_n, lens_vec_seg=ll_vec.half().cpu().numpy())
    if sc is not None:                                           # (section 13) what it came to know of its attending
        sst = float((sc_sum2 - (sc_sum ** 2).sum(-1) / max(sc_n, 1)).sum())
        h2_ = slice(T // 2, T)
        extra.update({f"sc_{k_}": v_ for k_, v_ in sc_log.items()})
        extra["sc_r2"] = np.float32(1.0 - np.nansum(sc_log["sse"][h2_]) * B / max(sst, 1e-12))
        extra["sc_r2_nb"] = np.float32(1.0 - np.nansum(sc_log["sse_nb"][h2_]) * B / max(sst, 1e-12))
        if sz_rows:                                              # per copy and token: size, its schema's, the neighbour-trained
            extra["sz_rows"] = np.stack(sz_rows).astype(np.float32)   # (tokens, B, 3)
        if sc_dump is not None:
            np.savez_compressed(sc["dump"], **{k_: np.stack(v_) for k_, v_ in sc_dump.items()})
    if lp:
        extra["lp_map"] = traj_lp
    if body is not None:
        extra["U"] = traj_U
    if kn:
        extra.update(s_read=traj_read, s_self=traj_self)
    if fut is not None:
        extra.update(fut_left=np.float32(fut_num / max(fut_den, 1e-12)), fut_places=np.array(fut_places, np.int16))
        if fut["interaction"]:
            extra["fut_interaction"] = np.float32(int_num / max(int_den, 1e-12))
    if sm is not None:                                           # (self-model) what each copy came to know
        extra.update({f"sm_{k_}": p_.detach().cpu().numpy() for k_, p_ in zip(("W1", "b1", "W2", "b2"), sm_params)})
    if sp is not None:                                           # the stance's spread against the windows' spread
        extra["sp_attention"] = np.float32(np.mean(sp_tv))
        for key_ in sp_num:
            Xb = torch.cat(sp_base[key_], 0)
            extra[f"sp_{key_}"] = np.float32(np.mean(sp_num[key_]) / max(float(((Xb - Xb.mean(0)) ** 2).sum(-1).mean()), 1e-12))
    np.savez_compressed(os.path.join(out, "life.npz"), s=traj["s"], P=traj_P, genre=Z["genre"][:, :T].T, **log, **extra)
    open(os.path.join(out, "DONE"), "w").write(f"{time.time() - t0:.1f}\n")


def frozen(out_json, states, T, items, K=4, every=50, k_h=32):
    """Frozen pass on new texts: the stance's course, and the futures from probe stances (no learning)."""
    Z = np.load(states)
    H = torch.tensor(Z["h"][:, :T + 1], dtype=torch.float32, device=DEV)
    res = {}
    for item in items:
        name, rest = item.split("=", 1)
        run_dir, iface = rest.split("::")
        cfg = json.load(open(os.path.join(run_dir, "run.json")))
        f = build(dict(cfg, init_iface=iface), Z)
        B, m = H.shape[0], cfg["m"]
        g = torch.Generator().manual_seed(4242)
        S0 = unit(torch.randn(K, m, generator=g)).to(DEV)
        s = unit(torch.randn(B, m, generator=torch.Generator().manual_seed(0))).to(DEV)
        hist = torch.zeros(B, 0, H.shape[-1], device=DEV)
        traj = np.zeros((T, B, m), np.float32)
        kept, kept1, places = [0.0, 0.0], [0.0, 0.0], []
        with torch.no_grad():
            for t in range(T):
                if t >= 150 and t % every == 0 and t + k_h < T:
                    S = torch.cat([S0, s], 0)
                    NS = S.shape[0]
                    rows = S.repeat(B, 1)
                    b_idx = torch.arange(B, device=DEV).repeat_interleave(NS)
                    hr = hist[b_idx]
                    for h in range(k_h):           # each row lives on by its own reading of the same text
                        hr = torch.cat([hr, H[b_idx, t + h:t + h + 1]], 1)[:, -f.R:]
                        rows = f.next_stance(f.read(rows, hr))
                        if h == 0:
                            D1 = rows.view(B, NS, m)
                    D = rows.view(B, NS, m)
                    sp = lambda X: float(((X - X.mean(1, keepdim=True)) ** 2).sum())  # noqa: E731
                    sprobe = B * float(((S - S.mean(0, keepdim=True)) ** 2).sum())
                    kept[0] += sp(D)
                    kept[1] += sprobe
                    kept1[0] += sp(D1)
                    kept1[1] += sprobe
                    for b in range(B):
                        first = []
                        for x in D[b]:
                            if not first or max(float(x @ y) for y in first) < 0.9:
                                first.append(x)
                        places.append(len(first))
                hist = torch.cat([hist, H[:, t:t + 1]], 1)[:, -f.R:]
                traj[t] = s.cpu().numpy()
                s = f.next_stance(f.read(s, hist))
        os.makedirs(os.path.join(run_dir, "frozen32"), exist_ok=True)
        np.savez_compressed(os.path.join(run_dir, "frozen32", f"frozen_{os.path.basename(iface).replace('.pt', '')}.npz"), s=traj)
        res[name] = dict(interface=iface, left_after_32=kept[0] / max(kept[1], 1e-12), left_after_1=kept1[0] / max(kept1[1], 1e-12),
                         places=float(np.median(places)), measurements=len(places) // B)
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "record":                    # optional: MODEL BAND_FIRST BAND_LAST READ_LAYER (default Qwen3-0.6B 12 18 19)
        extra = {}
        if len(sys.argv) > 6:
            extra = dict(model=sys.argv[6], band=(int(sys.argv[7]), int(sys.argv[8])), read_layer=int(sys.argv[9]))
        record(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), **extra)
    elif cmd == "life":
        life(json.load(open(sys.argv[2])))
    elif cmd == "frozen":
        frozen(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5:])
