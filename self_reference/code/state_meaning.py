"""What do the emerged states mean? (side quest 2 from your message of 18 October: grammar, text indices, attention
state, abstract content; all existing data at both sizes, the new v22 lives and step B.)
Each life can have two kinds of records:
- "ref": the frozen reference pass on four new texts (self_formation_ref.npz: stance, picture, the reader's attention
  entropy and age, the model's loss; 1,100 tokens; the same four texts for every 0.6B life), last 900 tokens;
- "life": the life itself (life.npz: stance and picture at every token of its own texts), second half.
Per token, features of what is being read, in four families:
- grammar: the class of the current token (newline, sentence end, other punctuation, number, function word, content
  word, word continuation) and of the previous token; tokens since the sentence began (log); inside quotes;
- text indices: tokens since the document began and since the line began (log); for "life" also the place in the run;
- attention state ("ref" only): the reader's attention entropy and age at this token, the model's surprise at it;
- abstract content: the genre, and the text's content as the model represents it (read-layer state of Qwen3-0.6B
  without the loop's offsets: top 16 principal components of the mean over the reader's 16-token window, and of the
  current token's own state). For the 4B lives the 0.6B representation stands in for the text (a stated proxy).
Measures (leave one copy out: fit on three copies' texts, test on the fourth, so slow features cannot be decoded by
remembering the same text):
- decoding each feature from the picture P[t], from the next stance s[t+1] = dec(P[t]) and from the stance s[t] that
  was formed before token t was read (a built-in control for features of token t), against the same decoding with
  the state shifted 300 tokens in time (keeps the state's own statistics, breaks its tie to the text);
- whether the stance steers the reader's attention beyond what the text sets: attention at t from the text alone
  against the text plus s[t]; and whether the picture carries the next token's attention beyond the next text;
- what the state is made of: the share of the stance's and the picture's variance explained by each family alone,
  by all together, and each family's unique share;
- time scales: the share of the stance's variance between documents, between sentences within a document, and
  within sentences, against the same with the boundaries shifted 300 tokens;
- the resting states (stance held at least 5 tokens, grouped at distance 0.3, pooled over copies): how much the state
  tells about the genre, the class of the token that started it and the sentence position, against shifted labels;
  how often a jump follows each token class, against the base rate.
Usage: python -I state_meaning.py OUT_JSON RUN_DIR [RUN_DIR ...]"""
import json
import os
import random
import sys
import time
import warnings
from collections import Counter

import numpy as np
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
CORPUS = os.environ.get("JSPACE_CORPUS", "data/corpus.json")
CACHE = os.environ.get("MEANING_CACHE", "artifacts/meaning_cache")
SHIFT, NPC, NWIN = 300, 16, 16
TEXT_MODEL = ("Qwen/Qwen3-0.6B", (12, 18), 19, 256)
FUNC = set("""a an the this that these those my your his her its our their some any no every each either neither i you
he she it we they me him us them myself yourself himself herself itself ourselves themselves who whom whose which what
of in on at by for with from to into onto upon about over under after before between through during without within
and or but nor so yet because although though if while when where whether than as is am are was were be been being
have has had do does did will would shall should can could may might must not there here then also just only very
too""".split())
CLASSES = ["newline", "sent_end", "punct", "num", "func", "word", "cont"]
ALPHAS = np.logspace(-2, 4, 7)


def streams(corpus, tok, B, T, seed):
    """make_streams of llm_loop.py, also returning each token's place in its document."""
    rng = random.Random(seed)
    genres = sorted(corpus)
    I, G, D = [], [], []
    for _ in range(B):
        ids, g, dp = [], [], []
        while len(ids) < T + 2:
            gi = rng.randrange(len(genres))
            doc = rng.choice(corpus[genres[gi]])
            t = tok(doc + "\n\n", add_special_tokens=False)["input_ids"]
            ids += t
            g += [gi] * len(t)
            dp += list(range(len(t)))
        I.append(ids[:T + 2])
        G.append(g[:T + 2])
        D.append(dp[:T + 2])
    return np.array(I), np.array(G), np.array(D)


def token_features(ids, dp, tok):
    B, L = ids.shape
    surf = {}
    cls, sent, line, quote = (np.zeros((B, L), int), np.zeros((B, L)), np.zeros((B, L)), np.zeros((B, L)))
    for b in range(B):
        prev, sp, lp, q = "newline", 0, 0, 0
        for t in range(L):
            if dp[b, t] == 0:
                prev, sp, lp, q = "newline", 0, 0, 0
            i = int(ids[b, t])
            if i not in surf:
                surf[i] = tok.decode([i])
            s_ = surf[i]
            st = s_.strip()
            if "\n" in s_:
                c = "newline"
            elif not st or all(not ch.isalnum() for ch in st):
                c = "sent_end" if any(ch in ".!?" for ch in st) else "punct"
            elif st[0].isdigit():
                c = "num"
            elif s_.startswith(" ") or prev in ("newline", "sent_end", "punct"):
                c = "func" if st.lower() in FUNC else "word"
            else:
                c = "cont"
            cls[b, t], sent[b, t], line[b, t], quote[b, t] = CLASSES.index(c), sp, lp, q % 2
            q += s_.count('"') + s_.count("\u201c") + s_.count("\u201d")
            sp = 0 if c in ("sent_end", "newline") else sp + 1
            lp = 0 if c == "newline" else lp + 1
            prev = c
    return cls, sent, line, quote


def content(seed, T, B, ids):
    """Read-layer states of the plain model (no loop offsets) along the streams: top principal components of the
    reader-window mean and of the current token's state. Cached per stream."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"content_seed{seed}_T{T}_B{B}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["win"], z["now"]
    import torch
    from loop12 import Runner12
    name, band, rl, win = TEXT_MODEL
    R = Runner12(name, band=band, read_layer=rl, window=win)
    L = ids.shape[1] - 2
    H = np.zeros((L, B, R.d), np.float32)
    t0 = time.time()
    with torch.no_grad():
        st = R.new_state(B)
        x = torch.tensor(ids, device=R.dev)
        for t in range(L):
            h, _ = R.upper_live(st, R.lower(st, x[:, t]), lambda l, s: None, None)
            H[t] = h[:, 0].float().cpu().numpy()
            if t % 1000 == 0:
                print(f"  content pass seed {seed} T {T}: {t}/{L} {time.time() - t0:.0f}s", flush=True)
    del R
    H = (H - H[16:].reshape(-1, H.shape[-1]).mean(0)) / (H[16:].reshape(-1, H.shape[-1]).std(0) + 1e-6)  # outlier dims
    cs = np.cumsum(np.concatenate([np.zeros((1, B, H.shape[-1])), H.astype(np.float64)], 0), 0)
    lo = np.maximum(np.arange(L) - NWIN + 1, 0)
    W = (cs[np.arange(L) + 1] - cs[lo]) / (np.arange(L) + 1 - lo)[:, None, None]

    def pcs(X):
        Y = X.reshape(-1, X.shape[-1])[16 * B:]                                  # fit after the first window
        mu = Y.mean(0)
        w, V = np.linalg.eigh(np.cov((Y - mu).T))
        V = V[:, ::-1][:, :NPC]
        Z = (X.reshape(-1, X.shape[-1]) - mu) @ V
        return (Z / Z[16 * B:].std(0)).reshape(L, B, NPC).astype(np.float32)
    win_, now_ = pcs(W), pcs(H)
    np.savez_compressed(path, win=win_, now=now_)
    return win_, now_


def fit_score(X, y, kind, shift=0):
    """Leave one copy out. X: (n, B, k); y: (n, B) labels or (n, B, q) values. Balanced accuracy or pooled R^2."""
    n, B = X.shape[:2]
    Xs = np.roll(X, shift, axis=0) if shift else X
    sc = []
    for hold in range(B):
        tr = [b for b in range(B) if b != hold]
        Xtr, Xte = Xs[:, tr].reshape(n * (B - 1), -1), Xs[:, hold].reshape(n, -1)
        keep = Xtr.std(0) > 1e-6                          # constant inputs (e.g. a saturated picture value) carry nothing
        if not keep.any():
            continue
        Xtr, Xte = Xtr[:, keep], Xte[:, keep]
        Xte = np.clip(Xte, Xtr.min(0), Xtr.max(0))        # no extrapolation beyond the training copies' range
        mu, sd = Xtr.mean(0), Xtr.std(0)
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        if kind == "cat":
            ytr, yte = y[:, tr].reshape(-1), y[:, hold]
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            m = RidgeClassifierCV(alphas=ALPHAS, class_weight="balanced").fit(Xtr, ytr)
            sc.append(balanced_accuracy_score(yte, m.predict(Xte)))
        else:
            ytr, yte = y[:, tr].reshape(n * (B - 1), -1), y[:, hold].reshape(n, -1)
            sst = ((yte - ytr.mean(0)) ** 2).sum()
            if ytr.std(0).max() < 1e-6 or sst < 1e-8 * yte.size:      # a target that does not vary says nothing
                continue
            m = RidgeCV(alphas=ALPHAS).fit(Xtr, ytr)
            sse = ((yte - m.predict(Xte).reshape(n, -1)) ** 2).sum()
            sc.append(1 - sse / sst)
    return float(np.mean(sc)) if sc else None


def onehot(a, k):
    return np.eye(k)[a]


def H_(c):
    p = np.array(list(c.values()), float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def mi(a, b):
    return H_(Counter(a)) + H_(Counter(b)) - H_(Counter(zip(a, b)))


def nested_shares(Y, doc_id, sent_id, genre=None):
    """Y: (n, B, q). Variance shares between documents, between sentences within documents, within sentences; with
    genre labels, the between-documents share is also split into between genres and between documents of a genre
    (what a document is about beyond its genre)."""
    mu = Y.reshape(-1, Y.shape[-1]).mean(0)
    tot, bd, bs, ws = 0.0, 0.0, 0.0, 0.0
    for b in range(Y.shape[1]):
        y = Y[:, b]
        tot += ((y - mu) ** 2).sum()
        for d in np.unique(doc_id[:, b]):
            md = doc_id[:, b] == d
            yd = y[md]
            m_d = yd.mean(0)
            bd += md.sum() * ((m_d - mu) ** 2).sum()
            for s in np.unique(sent_id[md, b]):
                ms = sent_id[md, b] == s
                m_s = yd[ms].mean(0)
                bs += ms.sum() * ((m_s - m_d) ** 2).sum()
                ws += ((yd[ms] - m_s) ** 2).sum()
    if tot < 1e-8 * Y.size:                               # a stance that never moves has no time scales
        return dict(between_documents=None, between_sentences=None, within_sentences=None)
    out = dict(between_documents=bd / tot, between_sentences=bs / tot, within_sentences=ws / tot)
    if genre is not None:
        Yf, gf = Y.reshape(-1, Y.shape[-1]), genre.reshape(-1)
        bg = sum((gf == k).sum() * ((Yf[gf == k].mean(0) - mu) ** 2).sum() for k in np.unique(gf))
        out.update(between_genres=bg / tot, between_documents_within_genre=(bd - bg) / tot)
    return out


def regime(S):
    stp = np.linalg.norm(S[1:] - S[:-1], axis=-1)
    return dict(rests=float((stp < 0.01).mean()), big_steps=float((stp > 0.5).mean()),
                cos_next=float(np.median((S[1:] * S[:-1]).sum(-1))), cos_two_apart=float(np.median((S[2:] * S[:-2]).sum(-1))))


def rest_states(S):
    """Tokens resting (step < 0.01 for >= 5 tokens) get a state id (greedy grouping at 0.3, pooled over copies)."""
    n, B, _ = S.shape
    stp = np.linalg.norm(S[1:] - S[:-1], axis=-1)
    lab = -np.ones((n, B), int)
    centers = []
    for b in range(B):
        t = 0
        while t < n - 1:
            if stp[t, b] < 0.01:
                u = t
                while u < n - 1 and stp[u, b] < 0.01:
                    u += 1
                if u - t >= 5:
                    c = S[t:u + 1, b].mean(0)
                    j = next((i for i, cc in enumerate(centers) if np.linalg.norm(cc - c) < 0.3), None)
                    if j is None:
                        centers.append(c)
                        j = len(centers) - 1
                    lab[t:u + 1, b] = j
                t = u + 1
            else:
                t += 1
    return lab, len(centers)


def analyse(name, S_all, P_all, t_range, feats, att=None):
    """S_all: (Tn, B, m) with s[t] = stance used at token t; P_all[t] the picture after token t."""
    a, z = t_range
    tt = np.arange(a, z)
    s_before, s_next, P = S_all[tt], S_all[tt + 1], np.nan_to_num(P_all[tt])      # "nopic" lives record no picture
    B = S_all.shape[1]
    f = {k: v[:, tt].T if v.ndim == 2 else v[tt] for k, v in feats.items()}    # (n, B[, q])
    out = {"record": name, "tokens": int(len(tt)), "regime": regime(S_all[a:z + 1])}
    targets = {"class_now": ("cat", f["cls"]), "class_prev": ("cat", feats["cls"][:, tt - 1].T),
               "sentence_position": ("val", np.log1p(f["sent"])), "line_position": ("val", np.log1p(f["line"])),
               "document_position": ("val", np.log1p(f["dp"])), "in_quote": ("cat", f["quote"].astype(int)),
               "genre": ("cat", f["genre"]), "content_window_pcs": ("val", f["win"]),
               "content_token_pcs": ("val", f["now"])}
    if "time" in f:
        targets["run_position"] = ("val", f["time"])
    if att is not None:
        ent, age, loss = att
        targets.update({"attention_entropy": ("val", ent[tt]), "attention_age": ("val", age[tt]),
                        "surprise_at_token": ("val", loss[tt - 1]), "next_attention_entropy": ("val", ent[tt + 1]),
                        "next_attention_age": ("val", age[tt + 1])})
    dec = {}
    for tn, (kind, y) in targets.items():
        row = {}
        for xn, X in (("picture", P), ("next_stance", s_next), ("stance_before", s_before)):
            row[xn] = fit_score(X, y, kind)
            row[xn + "_shifted"] = fit_score(X, y, kind, SHIFT)
        row["text_only"] = fit_score(np.concatenate([f["win"], f["now"]], -1), y, kind) if tn not in (
            "content_window_pcs", "content_token_pcs") else None
        dec[tn] = row
    out["decoding"] = dec
    G = np.concatenate([onehot(f["cls"], 7), onehot(feats["cls"][:, tt - 1].T, 7), np.log1p(f["sent"])[..., None],
                        f["quote"][..., None]], -1)
    I_ = np.stack([np.log1p(f["dp"]), np.log1p(f["line"])] + ([f["time"]] if "time" in f else []), -1)
    C = np.concatenate([onehot(f["genre"], 5), f["win"], f["now"]], -1)
    fams = {"grammar": G, "indices": I_, "content": C}
    if att is not None:
        ent, age, loss = att
        fams["attention"] = np.stack([ent[tt], age[tt], loss[tt - 1]], -1)
        text = np.concatenate([G, I_, C], -1)
        out["attention_steering"] = {
            "attention_from_text": fit_score(text, np.stack([ent[tt], age[tt]], -1), "val"),
            "attention_from_text_and_stance": fit_score(np.concatenate([text, s_before], -1), np.stack([ent[tt], age[tt]], -1), "val"),
            "attention_from_stance_alone": fit_score(s_before, np.stack([ent[tt], age[tt]], -1), "val")}
        # the next token's text features with the current picture: is there anything about the next attention in P?
        nxt = np.concatenate([onehot(feats["cls"][:, tt + 1].T, 7), np.log1p(feats["sent"][:, tt + 1].T)[..., None],
                              feats["win"][tt + 1], feats["now"][tt + 1]], -1)
        yN = np.stack([ent[tt + 1], age[tt + 1]], -1)
        out["attention_steering"]["next_attention_from_next_text"] = fit_score(nxt, yN, "val")
        out["attention_steering"]["next_attention_from_next_text_and_picture"] = fit_score(np.concatenate([nxt, P], -1), yN, "val")
    comp = {}
    for yn, Y in (("next_stance", s_next), ("picture", P)):
        allf = np.concatenate(list(fams.values()), -1)
        r_all = fit_score(allf, Y, "val")
        row = {"all_families": r_all}
        for fn, Fm in fams.items():
            row[fn + "_alone"] = fit_score(Fm, Y, "val")
            rest = np.concatenate([v for k, v in fams.items() if k != fn], -1)
            r_rest = fit_score(rest, Y, "val")
            row[fn + "_unique"] = None if r_all is None or r_rest is None else r_all - r_rest
        row["all_families_shifted"] = fit_score(allf, Y, "val", SHIFT)
        comp[yn] = row
    out["made_of"] = comp
    doc_id = np.cumsum(f["dp"] == 0, 0)
    ends = np.isin(f["cls"], [CLASSES.index("sent_end"), CLASSES.index("newline")])
    sent_id = doc_id * 100000 + np.cumsum(np.vstack([np.zeros((1, B), bool), ends[:-1]]), 0)
    out["time_scales"] = {"next_stance": nested_shares(s_next, doc_id, sent_id, f["genre"]),
                          "picture": nested_shares(P, doc_id, sent_id, f["genre"]),
                          "next_stance_boundaries_shifted": nested_shares(s_next, np.roll(doc_id, SHIFT, 0), np.roll(sent_id, SHIFT, 0)),
                          "content_window_pcs": nested_shares(f["win"], doc_id, sent_id, f["genre"])}
    lab, ns = rest_states(S_all[a:z + 1])
    lab = lab[1:]                                       # state of s[t+1]: entered after reading token t
    rs = {"states": ns, "resting_share": float((lab >= 0).mean())}
    if ns >= 2:
        msk = lab >= 0
        ent_t = np.zeros_like(lab) - 1
        for b in range(B):
            for t in range(1, len(lab)):
                if lab[t, b] >= 0 and lab[t - 1, b] != lab[t, b]:
                    ent_t[t, b] = 1
        gen_s = np.roll(f["genre"], SHIFT, 0)
        sp_bin = np.digitize(f["sent"], [1, 4, 11])
        rs.update({"genre_bits": mi(lab[msk], f["genre"][msk]), "genre_bits_shifted": mi(lab[msk], gen_s[msk]),
                   "sentence_position_bits": mi(lab[msk], sp_bin[msk]),
                   "sentence_position_bits_shifted": mi(lab[msk], np.roll(sp_bin, SHIFT, 0)[msk]),
                   "state_entropy_bits": H_(Counter(lab[msk]))})
        e = ent_t == 1
        if e.sum() >= 10:
            rs["entry_token_class_bits"] = mi(lab[e], f["cls"][e])
            rs["entry_token_class_bits_shifted"] = mi(lab[e], np.roll(f["cls"], SHIFT, 0)[e])
    stp = np.linalg.norm(s_next - s_before, axis=-1)
    jump = stp > 0.5
    base = float(jump.mean())
    rs["jump_rate"] = base
    rs["jump_lift_by_token_class"] = {c: (float(jump[f["cls"] == i].mean()) / base if base > 0 and (f["cls"] == i).sum() > 20 else None)
                                      for i, c in enumerate(CLASSES)}
    out["rest_states"] = rs
    return out


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL[0])
    corpus = json.load(open(CORPUS))
    results = []
    feat_cache = {}

    def get_feats(seed, T, B):
        key = (seed, T, B)
        if key not in feat_cache:
            ids, gen, dp = streams(corpus, tok, B, T, seed)
            cls, sent, line, quote = token_features(ids, dp, tok)
            win, now = content(seed, T, B, ids)
            L = win.shape[0]
            feat_cache[key] = dict(cls=cls[:, :L], sent=sent[:, :L], line=line[:, :L], quote=quote[:, :L],
                                   genre=gen[:, :L], dp=dp[:, :L], win=win, now=now)
        return feat_cache[key]

    for d in sys.argv[2:]:
        d = d.rstrip("/\\")
        cfg = json.load(open(os.path.join(d, "run.json")))
        B = cfg.get("B", 4)
        life = {"life": os.path.basename(d), "dir": d, "model": cfg["model"], "arm": cfg.get("arm"),
                "picture": cfg.get("picture", "cube"), "route": cfg.get("reader_route", "operation"),
                "identity": cfg.get("identity", "position"), "div_scale": cfg.get("div_scale", 1.0), "records": []}
        t0 = time.time()
        try:
            add_records(life, d, cfg, B, get_feats)
        except Exception as e:                                       # a life whose records do not fit is listed, not hidden
            life["error"] = repr(e)
        print(f"{life['life']}: {time.time() - t0:.0f}s {life.get('error', '')}", flush=True)
        results.append(life)
        json.dump(results, open(sys.argv[1], "w"), indent=1, default=float)


def add_records(life, d, cfg, B, get_feats):
    rp = os.path.join(d, "self_formation_ref.npz")
    if os.path.exists(rp):
        z = np.load(rp)
        zz = {k: z[k] for k in ("s", "P", "genre", "att_entropy", "att_age", "loss")}
        TE = zz["s"].shape[0]
        fe = get_feats(99, TE + 5, B)
        assert (fe["genre"][:, :TE].T == zz["genre"]).all(), "reference stream mismatch"
        life["records"].append(analyse("ref", zz["s"], zz["P"], (TE - 900, TE - 2), fe,
                                       (zz["att_entropy"], zz["att_age"], zz["loss"])))
    lp = os.path.join(d, "life.npz")
    if os.path.exists(lp):
        z = np.load(lp)
        S, P, g = z["s"], z["P"], z["genre"]
        T = S.shape[0]
        fe = dict(get_feats(cfg["stream_seed"], cfg.get("stream_T", cfg["T"]), B))
        assert (fe["genre"][:, :T].T == g).all(), "life stream mismatch"
        fe["time"] = np.broadcast_to((np.arange(fe["win"].shape[0]) / T)[None], (B, fe["win"].shape[0])).copy()
        life["records"].append(analyse("life", S, P, (T // 2, T - 2), fe))


if __name__ == "__main__":
    main()
