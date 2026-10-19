"""Self-describing attention loop, including centered-field identity and the shared spotlight.
Run with: python loop12.py config.json. See ../METHODS.md for definitions and numerical settings."""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from llm_loop import Interface, Runner, contrastive, make_streams  # noqa: E402
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv  # noqa: E402


class Runner12(Runner):
    """Runner whose band layers can add a per-row content-dependent query change (the copy's slow weights)."""

    def layer(self, l, h, Kp, Vp, pos, q_off=None, ad=None, sb=None):
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

    def upper_live(self, st, h_in, offs_fn, s, ad=None, sb=None):
        h = h_in
        h_read = None
        for l in range(self.band[0], self.L):
            off = offs_fn(l, s) if l in self.band else None
            h, k, v = self.layer(l, h, st["K"][l], st["V"][l], st["pos"], off, ad if l in self.band else None,
                                 sb if l in self.band else None)
            st["K"][l] = self.crop(torch.cat([st["K"][l], k], 2))
            st["V"][l] = self.crop(torch.cat([st["V"][l], v], 2))
            if l == self.read_layer:
                h_read = h
        logits = self.model.lm_head(self.model.model.norm(h))
        st["pos"] += 1
        return h_read, logits


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

    def read(s, hist, item=None):
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
        att = torch.softmax(logit, -1)
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
    R = Runner12(cfg["model"], band=tuple(cfg["band"]), read_layer=cfg["read_layer"], window=cfg["window"], revision=cfg.get("model_revision"))
    corpus = json.load(open(os.environ.get("JSPACE_CORPUS", os.environ.get("V11_CORPUS", cfg.get("corpus", "data/corpus.json")))))
    B, K, m, T = cfg["B"], cfg["K"], cfg["m"], cfg["T"]
    # the stream builder draws the four worlds' documents from one random sequence, so its length must not depend on T
    ids, gen, genres = make_streams(corpus, R.tok, B, cfg.get("stream_T", T), cfg["stream_seed"])
    ids, gen = ids[:, :T + 2], gen[:, :T + 2]
    if cfg.get("same_text"):                                   # every copy reads world 0's text
        ids, gen = ids[:1].repeat(B, 1), np.repeat(gen[:1], B, 0)
    ids = ids.to(R.dev)
    f = Interface(R, m=m, p=cfg["p"], R=cfg["R"], bound=cfg["bound"], steer_reader=cfg.get("steer_reader", True)).to(R.dev)
    set_picture_map(f, cfg.get("picture_map"))
    set_picture_geometry(f, cfg.get("picture", "cube"), cfg.get("reader_route", "operation"))
    if cfg.get("init_iface"):                    # (v22) start from a trained interface: a developmental start
        sd0 = torch.load(cfg["init_iface"], map_location=R.dev)
        sd0.pop("lam_raw", None)
        f.load_state_dict(sd0)
    content = cfg.get("reader_route") == "content"
    identity = cfg.get("identity", "position")
    another = cfg.get("describe") == "another"
    if another:
        zt = np.load(os.path.join(cfg["twin"], "twin_record.npz"))
        twin = {k_: torch.tensor(zt[k_], device=R.dev) for k_ in ("S", "D")}
    rec_tw = {"S": [], "D": [], "s": []} if cfg.get("record_twin") else None
    div = cfg.get("div", "shared")
    plastic = cfg.get("plastic", "none")
    w_rec, w_div = 1.0, (0.0 if div == "none" else float(cfg.get("div_scale", 1.0)))   # div_scale: the self-negation sweep
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
    chunk_loss, chunk_ext, t0 = 0.0, 0.0, time.time()
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
            h_in = R.lower(st, ids[:, t])
            S = unit(torch.randn(K, m, generator=pg)).to(R.dev)
            S = torch.cat([S, s.detach()], 0)
            if another:                                  # the twin's probe stances: its field is the target
                S = twin["S"][t]
            b_idx = torch.arange(B, device=R.dev).repeat_interleave(S.shape[0])
            SS = S.repeat(B, 1)
            sb_probe = spot_probe(SS, st["hist"][b_idx], s, st["hist"], b_idx) if kappa is not None else None
            hr_probe = R.upper_probe(st, h_in, offs_rows(b_idx), SS, b_idx, slow.fn(b_idx) if slow is not None else None,
                                     sb=sb_probe).float()
            if div == "local":
                Kl = cfg.get("K_local", 4)
                SL = unit(torch.randn(B * Kl, m, generator=pl)).to(R.dev)
                bl = torch.arange(B, device=R.dev).repeat_interleave(Kl)
                hr_loc = R.upper_probe(st, h_in, offs_rows(bl), SL, bl, slow.fn(bl) if slow is not None else None).float()
        if body and body_arm == "blind":
            hist_p = hist0[b_idx]                       # the field is formed from what is read, not from its own attending
        else:
            hist_p = torch.cat([st["hist"][b_idx], hr_probe], 1)[:, -f.R:]
        Pp = read_rows(SS, hist_p, b_idx)
        if body and body_arm == "swap":        # what actually moves copy c is copy c+1's picture
            Pp = torch.roll(Pp.view(B, -1, Pp.shape[-1]), -1, 0).reshape(-1, Pp.shape[-1])
        Dp = (f.next_stance(Pp) - SS).view(B, -1, m)
        Dt = Dp                                                  # closure (itself arm)
        if another:
            Dt = twin["D"][t]                                    # another life's recorded field, no closure
        if rec_tw is not None:
            rec_tw["S"].append(S.detach().cpu().numpy())
            rec_tw["D"].append(Dp.detach().cpu().numpy())
            rec_tw["s"].append(s.detach().cpu().numpy())
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
        P = read_rows(s, hist0 if (body and body_arm == "blind") else hist, rows_live)
        if lasting:
            P_prev = P
            lam_traj.append(float(torch.sigmoid(f.lam_raw)))
        elif content:
            P_prev = P
        s_next = f.next_stance(torch.roll(P, -1, 0) if (body and body_arm == "swap") else P)
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
        L_ext_copies = F.cross_entropy(logits[:, 0].float(), ids[:, t + 1], reduction="sum")   # per-copy sum for the slow weights
        L_rec, accs, L_div = 0.0, [], 0.0
        if div == "local":
            with torch.no_grad():
                hp = torch.cat([st["hist"][bl], hr_loc], 1)[:, -f.R:]
                DL = (f.next_stance(read_rows(SL, hp, bl)) - SL).view(B, Kl, m)
        if identity == "dynamics":                               # identity by how it moves, not where it is
            dest_t = Dt + S[None]
            Dt_cmp = dest_t - dest_t.mean(1, keepdim=True)
        else:
            Dt_cmp = Dt
        for b in range(B):
            pf = f.predict_field(P[b:b + 1], S)
            if identity == "dynamics":
                pd_ = pf + S
                pf = pd_ - pd_.mean(0, keepdim=True)
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
        if t < calib:
            calib_sums += [float(L_ext), float(L_rec), float(L_div) if w_div else 1.0] + ([float(L_self)] if sf else [])
            if t == calib - 1:
                lam = calib / np.maximum(calib_sums, 1e-6)
                if sf:
                    lam[3] = max(n_self, 1) / max(calib_sums[3], 1e-6)       # mean over the tokens where the forecast ran
                json.dump(dict(cfg=cfg, loss_weights=lam.tolist()), open(os.path.join(out, "config.json"), "w"), indent=1)
        loss = lam[0] * L_ext + w_rec * lam[1] * L_rec + (w_div * lam[2] * L_div if w_div else 0.0) + ((lam[3] if len(lam) > 3 else 1.0) * L_self if sf else 0.0)
        chunk_loss = chunk_loss + (loss if t >= calib else 0.0 * loss)
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
    if lasting:
        extra_save["lam"] = np.array(lam_traj, np.float32)
    if sf:
        extra_save.update(forecast_err=traj_fe, delta=traj_delta, text64=traj_text, act64=traj_act,
                          f_scale=np.array(f_scale if f_scale is not None else np.nan, np.float32))
    np.savez_compressed(os.path.join(out, "life.npz"), s=traj_s, P=traj_P, genre=gen[:, :T].T, **extra_save,
                        **{k: np.array(v, np.float32) for k, v in log.items()}, **{"slow_" + k: np.array(v, np.float32) for k, v in slog.items()})
    open(os.path.join(out, "DONE"), "w").write(f"{time.time() - t0:.1f}\n")


if __name__ == "__main__":
    run(json.load(open(sys.argv[1])))
