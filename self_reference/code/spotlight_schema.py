"""Does the system represent its own act of attending? (PLAN section 18, prediction 5; the tested criterion: "the model must
represent the act of attending, not just what gets attended to".)
Targets, per token and copy: how sharp the reader's attention is (entropy) and how far back it looks (mean age, 0 =
the newest of the 16 positions); with the spotlight this is also what focuses the model.
Three nested predictors, each fitted leave-one-copy-out (state_meaning.fit_score):
  text            the content of what is read (state_meaning's text features: model representation, grammar, genre);
  text + stance   adds the stance that set the reader's query (attention *control*);
  text + stance + picture   adds the picture formed by that reading.
Reported: R² of each, the gain of the stance (control) and the further gain of the picture (representation of the
attending beyond what was intended and what was there).
Records: "ref" (frozen reference pass, self_formation_ref.npz: reader attention entropy/age, 1,100 tokens, same texts
for every life) and, for spotlight lives, "own_record" (life.npz "spot": the spotlight itself, second half).
Usage: python -I spotlight_schema.py OUT_JSON RUN_DIR [RUN_DIR ...]"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import state_meaning as sm  # noqa: E402


def text_features(fe, tt):
    cls = fe["cls"][:, tt].T
    G = np.concatenate([sm.onehot(cls, 7), np.log1p(fe["sent"][:, tt].T)[..., None]], -1)
    C = np.concatenate([sm.onehot(fe["genre"][:, tt].T, 5), fe["win"][tt], fe["now"][tt]], -1)
    I_ = np.log1p(fe["dp"][:, tt].T)[..., None]
    return np.concatenate([G, C, I_], -1)


def gains(X_text, s, P, targets):
    out = {}
    for name, y in targets.items():
        r1 = sm.fit_score(X_text, y, "val")
        r2 = sm.fit_score(np.concatenate([X_text, s], -1), y, "val")
        r3 = sm.fit_score(np.concatenate([X_text, s, P], -1), y, "val")
        out[name] = dict(text=r1, text_stance=r2, text_stance_picture=r3,
                         control_gain=None if r1 is None or r2 is None else r2 - r1,
                         representation_gain=None if r2 is None or r3 is None else r3 - r2)
    return out


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(sm.TEXT_MODEL[0])
    corpus = json.load(open(sm.CORPUS))
    cache = {}

    def feats(seed, T, B):
        if (seed, T, B) not in cache:
            ids, gen, dp = sm.streams(corpus, tok, B, T, seed)
            cls, sent, line, quote = sm.token_features(ids, dp, tok)
            win, now = sm.content(seed, T, B, ids)
            L = win.shape[0]
            cache[(seed, T, B)] = dict(cls=cls[:, :L], sent=sent[:, :L], genre=gen[:, :L], dp=dp[:, :L], win=win, now=now)
        return cache[(seed, T, B)]
    res = []
    for d in sys.argv[2:]:
        d = d.rstrip("/\\")
        cfg = json.load(open(os.path.join(d, "run.json")))
        row = {"life": os.path.basename(d), "spotlight": cfg.get("spotlight"), "spotlight_from": cfg.get("spotlight_from", "own")}
        rp = os.path.join(d, "self_formation_ref.npz")
        if os.path.exists(rp):
            z = np.load(rp)
            TE = z["s"].shape[0]
            fe = feats(99, TE + 5, 4)
            tt = np.arange(TE - 900, TE - 2)
            row["ref"] = gains(text_features(fe, tt), z["s"][tt], np.nan_to_num(z["P"][tt]),
                               {"attention_sharpness": z["att_entropy"][tt], "attention_age": z["att_age"][tt]})
        lp = os.path.join(d, "life.npz")
        if os.path.exists(lp):
            z = np.load(lp)
            if "spot" in z.files:
                T = z["s"].shape[0]
                fe = feats(cfg["stream_seed"], cfg.get("stream_T", cfg["T"]), 4)
                tt = np.arange(T // 2, T - 2)
                p = z["spot"][tt].astype(np.float64)                       # (n, B, 16), newest position last
                p = p / np.maximum(p.sum(-1, keepdims=True), 1e-12)
                ent = -(p * np.log(np.maximum(p, 1e-12))).sum(-1)
                age = (p * np.arange(p.shape[-1] - 1, -1, -1)).sum(-1)
                row["own_record"] = gains(text_features(fe, tt), z["s"][tt], z["P"][tt], {"spotlight_sharpness": ent, "spotlight_age": age})
        res.append(row)
        print(json.dumps(row, default=float), flush=True)
        json.dump(res, open(sys.argv[1], "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
