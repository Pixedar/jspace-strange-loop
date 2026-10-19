"""Do lives with a common origin come to move in their own ways? (developmental individuality between lives; v22)
Every life's interface is run frozen on the same four texts, from the same initial stances, with the same 12 fixed
probe stances as dynamics_anatomy.py, and its field's state-dependent part (destinations minus their mean over the
probes) is recorded every 10 tokens of the last two thirds, with the stance path.
  record:  python life_fields.py record RUN_DIR IFACE_PT [T_EVAL]  -> fields_<iface>.npz in the run directory
  compare: python life_fields.py compare OUT_JSON NPZ [NPZ ...]
Compare reports, for every pair of records: the correlation of their state-dependent parts on the same text, copy and
time (1 = they would move alike from every probe stance), the same for the whole destinations, and the mean distance
of their stance paths. The pairs of the same life at two checkpoints give the drift within a life: lives are
individual only if they differ from each other more than a life differs from itself 500 tokens later."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def record(run, iface, TE):
    import torch
    from llm_loop import Interface, make_streams
    from loop12 import Runner12, channel, set_picture_geometry, set_picture_map
    cfg = json.load(open(os.path.join(run, "run.json")))
    B, m, NP, EV = 4, cfg["m"], 12, 10
    unit = lambda z: z / z.norm(dim=-1, keepdim=True)  # noqa: E731
    R = Runner12(cfg["model"], band=tuple(cfg["band"]), read_layer=cfg["read_layer"], window=cfg["window"], revision=cfg.get("model_revision"))
    f = Interface(R, m=m, p=cfg["p"], R=cfg["R"], bound=cfg["bound"], steer_reader=cfg.get("steer_reader", True)).to(R.dev)
    set_picture_map(f, cfg.get("picture_map"))
    set_picture_geometry(f, cfg.get("picture", "cube"), cfg.get("reader_route", "operation"))
    sd_ = torch.load(iface, map_location=R.dev)
    sd_.pop("lam_raw", None)
    f.load_state_dict(sd_)
    f.eval()
    offs_fn, spot_live, spot_probe = channel(f, cfg)      # (v24) the shared spotlight, if the life has one
    corpus = json.load(open(os.environ.get("JSPACE_CORPUS", "data/corpus.json")))
    ids, gen, _ = make_streams(corpus, R.tok, B, TE + 5, 99)                  # the texts of dynamics_anatomy.py
    ids = ids.to(R.dev)
    S = unit(torch.randn(NP, m, generator=torch.Generator().manual_seed(4242))).to(R.dev)
    dest_all, path, times = [], [], []
    t0 = time.time()
    with torch.no_grad():
        st = R.new_state(B)
        s = unit(torch.randn(B, m, generator=torch.Generator().manual_seed(0))).to(R.dev)
        for t in range(TE):
            h_in = R.lower(st, ids[:, t])
            if t % EV == 0 and t >= TE // 3:
                b_idx = torch.arange(B, device=R.dev).repeat_interleave(NP)
                SS = S.repeat(B, 1)
                hr = R.upper_probe(st, h_in, offs_fn, SS, b_idx, sb=spot_probe(SS, st["hist"][b_idx], s, st["hist"], b_idx)).float()
                hp = torch.cat([st["hist"][b_idx], hr], 1)[:, -f.R:]
                dest_all.append(f.next_stance(f.read(S.repeat(B, 1), hp)).view(B, NP, m).cpu().numpy())
                times.append(t)
            h_read, _ = R.upper_live(st, h_in, offs_fn, s, sb=spot_live(s, st["hist"])[0])
            st["hist"] = torch.cat([st["hist"], h_read.float()], 1)[:, -f.R:]
            path.append(s.cpu().numpy())
            s = f.next_stance(f.read(s, st["hist"]))
    out = os.path.join(run, f"fields_{os.path.basename(iface).split('.')[0]}.npz")
    np.savez_compressed(out, dest=np.array(dest_all), path=np.array(path), times=np.array(times), genre=gen[:, :TE].T)
    print(f"{out} {time.time() - t0:.0f}s", flush=True)


def corr(x, y):
    x, y = x.ravel() - x.mean(), y.ravel() - y.mean()
    return float((x * y).sum() / max(np.sqrt((x ** 2).sum() * (y ** 2).sum()), 1e-12))


def compare(out_json, paths):
    recs = {}
    for p in paths:
        z = np.load(p)
        name = os.path.basename(os.path.dirname(p)) + "@" + os.path.basename(p).split("_")[-1].split(".")[0]
        dest = z["dest"]
        recs[name] = dict(res=dest - dest.mean(2, keepdims=True), dest=dest, path=z["path"], size=float(np.sqrt(((dest - dest.mean(2, keepdims=True)) ** 2).mean())))
    names = sorted(recs)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, Bv = recs[a], recs[b]
            n = min(len(A["res"]), len(Bv["res"]))
            pa, pb = A["path"], Bv["path"]
            k = min(len(pa), len(pb))
            rows.append(dict(a=a, b=b, same_life=a.split("@")[0] == b.split("@")[0],
                             dependence_corr=corr(A["res"][:n], Bv["res"][:n]), destination_corr=corr(A["dest"][:n], Bv["dest"][:n]),
                             path_distance=float(np.linalg.norm(pa[k // 3:k] - pb[k // 3:k], axis=-1).mean()),
                             sizes=(A["size"], Bv["size"])))
    for r in rows:
        print(f"{r['a']:>30s} vs {r['b']:<30s} {'(same life)' if r['same_life'] else '':12s} dependence corr {r['dependence_corr']:+.2f} "
              f"destination corr {r['destination_corr']:+.2f} paths apart {r['path_distance']:.2f} sizes {r['sizes'][0]:.3f}/{r['sizes'][1]:.3f}")
    json.dump(rows, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "record":
        record(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 900)
    else:
        compare(sys.argv[2], sys.argv[3:])
