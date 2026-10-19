"""Frozen Qwen3 runner and shared reader, picture, stance and field interface."""
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv


class Runner:
    def __init__(self, name, device="cuda", dtype=torch.bfloat16, band=(12, 18), read_layer=19, window=256, revision=None):
        self.tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, revision=revision).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dev, self.dtype = device, dtype
        self.layers = self.model.model.layers
        self.L = len(self.layers)
        self.band = list(range(band[0], band[1] + 1))
        self.read_layer = read_layer
        self.window = window
        a = self.layers[0].self_attn
        self.hd = a.head_dim
        self.nh = self.model.config.num_attention_heads
        self.kvh = self.model.config.num_key_value_heads
        self.qdim = self.nh * self.hd
        self.d = self.model.config.hidden_size

    def crop(self, X, sink=4):
        """Keep the first `sink` positions (attention sinks; dropping them wrecks Qwen's predictions, as in StreamingLLM)
        and the most recent ones, window in total."""
        if X.shape[2] <= self.window:
            return X
        return torch.cat([X[:, :, :sink], X[:, :, -(self.window - sink):]], 2)

    def new_state(self, B):
        z = torch.zeros(B, self.kvh, 0, self.hd, device=self.dev, dtype=self.dtype)
        return dict(K=[z.clone() for _ in range(self.L)], V=[z.clone() for _ in range(self.L)], pos=0,
                    hist=torch.zeros(B, 0, self.d, device=self.dev, dtype=torch.float32))

    def layer(self, l, h, Kp, Vp, pos, q_off=None):
        """h: (N, 1, d); Kp, Vp: (N, kvh, Lp, hd) earlier positions. Returns new h and this token's k, v."""
        lay = self.layers[l]
        a = lay.self_attn
        N = h.shape[0]
        x = lay.input_layernorm(h)
        q = a.q_proj(x)
        if q_off is not None:
            q = q + q_off.to(q.dtype)
        q = a.q_norm(q.view(N, 1, self.nh, self.hd)).transpose(1, 2)
        k = a.k_norm(a.k_proj(x).view(N, 1, self.kvh, self.hd)).transpose(1, 2)
        v = a.v_proj(x).view(N, 1, self.kvh, self.hd).transpose(1, 2)
        pid = torch.full((N, 1), pos, device=self.dev, dtype=torch.long)
        cos, sin = self.model.model.rotary_emb(x, pid)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        K = torch.cat([Kp, k], 2)
        V = torch.cat([Vp, v], 2)
        rep = self.nh // self.kvh
        o = F.scaled_dot_product_attention(q, repeat_kv(K, rep), repeat_kv(V, rep), scale=a.scaling)
        h = h + a.o_proj(o.transpose(1, 2).reshape(N, 1, -1))
        h = h + lay.mlp(lay.post_attention_layernorm(h))
        return h, k, v

    def lower(self, st, ids):
        """Layers below the write band for all B worlds (stance-independent); appends their k, v."""
        h = self.model.model.embed_tokens(ids.view(-1, 1))
        for l in range(self.band[0]):
            h, k, v = self.layer(l, h, st["K"][l], st["V"][l], st["pos"])
            st["K"][l] = self.crop(torch.cat([st["K"][l], k], 2))
            st["V"][l] = self.crop(torch.cat([st["V"][l], v], 2))
        return h

    def upper_probe(self, st, h_in, offs_fn, S, b_idx):
        """Probe variants: for world b (repeated), run the band up to the read layer with stance offsets. No cache update.
        h_in: (B, 1, d); S: (N, m) stances per variant; b_idx: (N,) world index per variant."""
        h = h_in[b_idx]
        for l in range(self.band[0], self.read_layer + 1):
            off = offs_fn(l, S) if l in self.band else None
            h, _, _ = self.layer(l, h, st["K"][l][b_idx], st["V"][l][b_idx], st["pos"], off)
        return h

    def upper_live(self, st, h_in, offs_fn, s):
        """Live path for all B worlds: band and the rest of the model; appends k, v; returns read-layer state, logits."""
        h = h_in
        h_read = None
        for l in range(self.band[0], self.L):
            off = offs_fn(l, s) if l in self.band else None
            h, k, v = self.layer(l, h, st["K"][l], st["V"][l], st["pos"], off)
            st["K"][l] = self.crop(torch.cat([st["K"][l], k], 2))
            st["V"][l] = self.crop(torch.cat([st["V"][l], v], 2))
            if l == self.read_layer:
                h_read = h
        logits = self.model.lm_head(self.model.model.norm(h))
        st["pos"] += 1
        return h_read, logits


class Interface(nn.Module):
    def __init__(self, runner, m=8, p=32, R=16, rh=4, bound=1.0, steer_reader=True):
        super().__init__()
        d, qdim = runner.d, runner.qdim
        self.m, self.p, self.R, self.rh, self.bound = m, p, R, rh, bound
        self.writers = nn.ModuleDict({str(l): nn.Linear(m, qdim) for l in runner.band})
        for w in self.writers.values():
            nn.init.normal_(w.weight, std=1.0 / math.sqrt(m))
            nn.init.zeros_(w.bias)
        self.rln = nn.LayerNorm(d)
        self.rk, self.rv = nn.Linear(d, 256), nn.Linear(d, 256)
        self.rq = nn.Linear(m, 256) if steer_reader else None
        self.rq_bias = nn.Parameter(torch.randn(256) * 0.1)
        self.rpos = nn.Parameter(torch.zeros(R))
        self.to_pic = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, p))
        self.dec = nn.Sequential(nn.Linear(p, 64), nn.GELU(), nn.Linear(64, m))
        self.field = nn.Sequential(nn.Linear(p + m, 128), nn.GELU(), nn.Linear(128, m))

    def offs(self, l, s):
        return (self.bound * torch.tanh(self.writers[str(l)](s))).unsqueeze(1)    # (N, 1, qdim)

    def read(self, s, hist):
        """s: (N, m); hist: (N, R', d) float32, current position last."""
        N, Rp, _ = hist.shape
        a = self.rln(hist)
        k = self.rk(a).view(N, Rp, self.rh, -1)
        v = self.rv(a).view(N, Rp, self.rh, -1)
        q = (self.rq_bias.expand(N, -1) + (self.rq(s) if self.rq is not None else 0.0)).view(N, self.rh, -1)
        att = torch.softmax(torch.einsum("nhd,nlhd->nhl", q, k) / math.sqrt(q.shape[-1]) + self.rpos[-Rp:][None, None], -1)
        r = torch.einsum("nhl,nlhd->nhd", att, v).reshape(N, -1)
        return torch.tanh(self.to_pic(r))

    def next_stance(self, P):
        z = self.dec(P)
        return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def predict_field(self, P, S):
        if P.shape[0] == 1 and S.shape[0] > 1:
            P = P.expand(S.shape[0], -1)
        return self.field(torch.cat([P, S], -1))


def make_streams(corpus, tok, B, T, seed, mean_docs=1):
    """B streams of ordinary text; genre switches at document boundaries; returns token ids and per-token genre."""
    rng = random.Random(seed)
    genres = sorted(corpus)
    out_ids, out_g = [], []
    for b in range(B):
        ids, g = [], []
        while len(ids) < T + 2:
            gi = rng.randrange(len(genres))
            doc = rng.choice(corpus[genres[gi]])
            t = tok(doc + "\n\n", add_special_tokens=False)["input_ids"]
            ids += t
            g += [gi] * len(t)
        out_ids.append(ids[:T + 2])
        out_g.append(g[:T + 2])
    return torch.tensor(out_ids), np.array(out_g), genres


def contrastive(pred, targets, b):
    d2 = ((pred[None] - targets) ** 2).sum(-1)
    tau = d2.detach().median().clamp_min(1e-6)
    logits = (-d2 / tau).T
    return F.cross_entropy(logits, torch.full((pred.shape[0],), b, dtype=torch.long, device=pred.device)), float((logits.argmax(1) == b).float().mean())


def run(cfg):
    if os.name == "nt":   # on the laptop the GPU job must win the CPU against background analyses
        try:
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x8000)
        except Exception:
            pass
    torch.manual_seed(cfg["init_seed"])
    out = cfg["out"]
    os.makedirs(out, exist_ok=True)
    R = Runner(cfg["model"], band=tuple(cfg["band"]), read_layer=cfg["read_layer"], window=cfg["window"], revision=cfg.get("model_revision"))
    corpus = json.load(open(cfg["corpus"]))
    B, K, m, T = cfg["B"], cfg["K"], cfg["m"], cfg["T"]
    ids, gen, genres = make_streams(corpus, R.tok, B, T, cfg["stream_seed"])
    ids = ids.to(R.dev)
    f = Interface(R, m=m, p=cfg["p"], R=cfg["R"], bound=cfg["bound"], steer_reader=cfg.get("steer_reader", True)).to(R.dev)
    arm = cfg["arm"]
    twin = None
    if arm in ("another", "carried"):
        z = np.load(os.path.join(cfg["twin"], "twin_record.npz"))
        twin = {k: torch.tensor(z[k], device=R.dev) for k in ("s", "S", "D")}
    w_rec = 0.0 if arm == "nopic" else 1.0
    w_div = 0.0 if (arm == "nopic" or cfg.get("nodiv")) else 1.0
    closure = arm in ("itself", "carried")
    unit = lambda z: z / z.norm(dim=-1, keepdim=True)  # noqa: E731
    pg = torch.Generator().manual_seed(77)
    st = R.new_state(B)
    s = unit(torch.randn(B, m, generator=torch.Generator().manual_seed(cfg["init_seed"]))).to(R.dev)
    lam = np.array(cfg.get("lam", [1.0, 1.0, 1.0]))
    opt = torch.optim.Adam(f.parameters(), lr=cfg["lr"])
    keys = ("L_ext", "L_rec", "L_div", "acc", "world_dep", "move")
    log = {k: [] for k in keys}
    traj_s = np.zeros((T, B, m), np.float32)
    traj_P = np.zeros((T, B, cfg["p"]), np.float32)
    Kp = K + B
    rec_S = np.zeros((T, Kp, m), np.float32) if arm == "itself" else None
    rec_D = np.zeros((T, B, Kp, m), np.float32) if arm == "itself" else None
    chunk_loss, t0 = 0.0, time.time()
    calib, calib_sums = cfg.get("calib_tokens", 150), np.zeros(3)
    for t in range(T):
        if arm == "carried":
            s = twin["s"][t].clone()
        with torch.no_grad():
            h_in = R.lower(st, ids[:, t])
            S = unit(torch.randn(K, m, generator=pg)).to(R.dev)
            S = torch.cat([S, s.detach()], 0) if arm != "another" else twin["S"][t]
            b_idx = torch.arange(B, device=R.dev).repeat_interleave(S.shape[0])
            SS = S.repeat(B, 1)
            hr_probe = R.upper_probe(st, h_in, f.offs, SS, b_idx).float()          # (B*Kp, 1, d)
        hist_p = torch.cat([st["hist"][b_idx], hr_probe], 1)[:, -f.R:]
        Pp = f.read(SS, hist_p)
        Dp = (f.next_stance(Pp) - SS).view(B, -1, m)                               # moves in every world at every probe
        Dt = Dp if closure else Dp.detach()
        if arm == "another":
            Dt = twin["D"][t]
        h_read, logits = R.upper_live(st, h_in, f.offs, s)
        hist = torch.cat([st["hist"], h_read.float()], 1)[:, -f.R:]
        P = f.read(s, hist)
        s_next = f.next_stance(P)
        L_ext = F.cross_entropy(logits[:, 0].float(), ids[:, t + 1])
        if cfg.get("body_contract", False):
            # Amendment A8: the attention setting (writer) is trained only by reading the world; the self-model pressures
            # train the self-description and the moves (reader, decoder, field), never the model's processing directly.
            # Without this the system made itself recognizable by wrecking its reading (next-token loss 3.4 -> 6.8).
            P_self = f.read(s.detach(), hist.detach())
            s_self = f.next_stance(P_self)
        else:
            P_self, s_self = P, s_next
        L_rec, accs, L_div = 0.0, [], 0.0
        for b in range(B):
            l, a = contrastive(f.predict_field(P_self[b:b + 1], S), Dt, b)
            L_rec, accs = L_rec + l / B, accs + [a]
            c_star = S[Dt[b].detach().norm(dim=-1).argmax()]
            L_div = L_div + (s_self[b] - c_star).norm() / B
        if t < calib:
            calib_sums += [float(L_ext), float(L_rec), float(L_div)]
            if t == calib - 1:
                lam = calib / np.maximum(calib_sums, 1e-6)    # fixed rule, once, from the first tokens of the life
                json.dump(dict(cfg=cfg, loss_weights=lam.tolist()), open(os.path.join(out, "config.json"), "w"), indent=1)
        loss = lam[0] * L_ext + w_rec * lam[1] * L_rec + w_div * lam[2] * L_div
        chunk_loss = chunk_loss + (loss if t >= calib else 0.0 * loss)
        st["hist"] = hist.detach() if (t + 1) % cfg["chunk"] == 0 else hist
        traj_s[t] = s.detach().cpu().numpy()
        traj_P[t] = P.detach().cpu().numpy()
        if rec_S is not None:
            rec_S[t], rec_D[t] = S.cpu().numpy(), Dp.detach().cpu().numpy()
        mv = float((s_next.detach() - s.detach()).norm(dim=-1).mean())
        for k_, v_ in zip(keys, (float(L_ext), float(L_rec), float(L_div), float(np.mean(accs)),
                                 float((Dp.detach() - Dp.detach().mean(0, keepdim=True)).norm(dim=-1).mean()), mv)):
            log[k_].append(v_)
        s = s_next if arm != "carried" else s
        if (t + 1) % cfg["chunk"] == 0:
            if t >= calib:
                opt.zero_grad()
                chunk_loss.backward()
                torch.nn.utils.clip_grad_norm_(f.parameters(), 1.0)
                opt.step()
            chunk_loss = 0.0
            s = s.detach()
            st["K"] = [k_.detach() for k_ in st["K"]]
            st["V"] = [v_.detach() for v_ in st["V"]]
        if (t + 1) % cfg.get("print_every", 250) == 0:
            pe = cfg.get("print_every", 250); w = traj_s[t - pe + 1:t + 1]
            apart = float(np.mean([np.linalg.norm(w[:, a] - w[:, c], axis=-1).mean() for a in range(B) for c in range(a + 1, B)]))
            print(f"[{arm}{'+' + cfg['tag'] if cfg.get('tag') else ''} s{cfg['init_seed']}] t={t + 1} {time.time() - t0:.0f}s "
                  f"Lext {np.mean(log['L_ext'][-pe:]):.3f} acc {np.mean(log['acc'][-pe:]):.2f} Ldiv {np.mean(log['L_div'][-pe:]):.2f} "
                  f"move {np.median(log['move'][-pe:]):.4f} worldsApart {apart:.3f} worldDep {np.mean(log['world_dep'][-pe:]):.3f} "
                  f"picSD {traj_P[t - pe + 1:t + 1].std(0).mean():.3f} mem {torch.cuda.max_memory_allocated() / 2**30:.2f}GiB", flush=True)
        if (t + 1) in (T // 2, T):
            torch.save(f.state_dict(), os.path.join(out, f"iface_{t + 1}.pt"))
    np.savez_compressed(os.path.join(out, "life.npz"), s=traj_s, P=traj_P, genre=gen[:, :T].T, **{k: np.array(v, np.float32) for k, v in log.items()})
    if rec_S is not None:
        np.savez(os.path.join(out, "twin_record.npz"), s=traj_s, S=rec_S, D=rec_D)
    open(os.path.join(out, "DONE"), "w").write(f"{time.time() - t0:.1f}\n")


if __name__ == "__main__":
    run(json.load(open(sys.argv[1])))
