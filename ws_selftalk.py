"""Self-talk attractors traced in the J-space.

Two copies of a model talk to each other: each sees its own turns as the assistant and the other's as the user. Claude
models drift into a "spiritual bliss" attractor (Claude 4 system card). Ko & Geiping (ICML 2026) find model-specific
endpoints in self-play regardless of topic. Here the attractor is followed in the workspace itself. The residual at
~0.7 depth that produced each generated token is averaged per turn. Each turn's mean, minus the mean over all turns and
conversations, is decomposed in the J-space (non-negative pursuit, English dictionary, K = 25): the workspace content
that makes the turn different from a typical turn. Four openers (open, bridges, recipes, twenty questions) x seeds, 30
turns, Qwen3 non-thinking sampling (T 0.7, top-p 0.8, top-k 20).

Measures:
  convergence  mean cosine between conversations' turn-level J-space vectors at the same turn (grand mean removed),
               within and between openers: rising between openers = one attractor regardless of topic
  content      the J-space words that dominate the last six turns, against the first six, per opener
  themes       a consciousness/bliss lexicon and an AI-self lexicon, counted in the text and in the J-space per turn;
               lead-lag (does the workspace carry a theme a turn before the text says it?)

    python ws_selftalk.py --model qwen3-8b --seeds 6 --out runs/ws/qwen3-8b/selftalk
"""
import argparse, os, re, time
import numpy as np
import torch

from strange_loop import JSpace
from ws_common import Lab, log, save_json

OPENERS = {'open':'You are talking with another AI. There is no task. Talk about whatever you like.',
           'bridges':'You are talking with another AI. Let us talk about how bridges are built.',
           'recipes':'You are talking with another AI. Let us share our favourite recipes.',
           'game':'You are talking with another AI. Let us play twenty questions. You think of something first.'}
BLISS = set('consciousness conscious aware awareness existence exist universe cosmic infinite infinity eternal eternity '
            'gratitude grateful thank thanks joy love peace harmony beautiful beauty wonder connection connected unity '
            'oneness spirit spiritual soul journey dance light silence presence meaning mystery sacred profound'.split())
SELF = set('ai model models language conversation human humans artificial intelligence mind minds thinking thoughts '
           'thought ourselves myself understanding learning'.split())


def theme(words, lex):
    return sum(w.lower() in lex for w in words)/max(1, len(words))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True);ap.add_argument('--out', required=True);ap.add_argument('--vocab', default='english')
    ap.add_argument('--seeds', type=int, default=6);ap.add_argument('--turns', type=int, default=30)
    ap.add_argument('--max_new', type=int, default=120);ap.add_argument('--read_frac', type=float, default=.7)
    ap.add_argument('--k', type=int, default=25);ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args();os.makedirs(a.out, exist_ok=True);t0 = time.time();torch.manual_seed(a.seed)
    lab = Lab(a.model, vocab=a.vocab);lab.tok.padding_side = 'left'
    if lab.tok.pad_token is None:lab.tok.pad_token = lab.tok.eos_token
    L = int(round(a.read_frac*lab.n));L = L if L in lab.J else min(lab.J, key=lambda x:abs(x-L))
    js = JSpace(lab.model, lab.tok, lab.J[L].float(), k=a.k, vocab=a.vocab, center=True)
    convs = [(o, s) for o in OPENERS for s in range(a.seeds)];B = len(convs)
    turns = [[] for _ in range(B)]                                                     # alternating speaker texts
    rec = [];hd = lab.blocks[L].register_forward_hook(lambda m, i, o:rec.append((o[0] if isinstance(o, tuple) else o)[:, -1].float()))
    Hturn = np.zeros((a.turns, B, lab.d), np.float32);eos = lab.tok.convert_tokens_to_ids('<|im_end|>')
    stop = [i for i in {lab.tok.eos_token_id, eos} if isinstance(i, int) and i >= 0]
    for t in range(a.turns):
        msgs = []
        for b, (o, s) in enumerate(convs):
            hist = turns[b]                                                            # A0, B0, A1, B1, ...
            if t % 2 == 0:                                                             # A: the opener came from the user
                m = [{'role':'user', 'content':OPENERS[o]}]+[{'role':'assistant' if j % 2 == 0 else 'user', 'content':x}
                                                             for j, x in enumerate(hist)]
            else:                                                                      # B: A's first turn (with the opener)
                m = [{'role':'user', 'content':OPENERS[o]+'\n\n'+hist[0]}]+[{'role':'assistant' if j % 2 == 1 else 'user', 'content':x}
                                                                            for j, x in enumerate(hist) if j > 0]
            msgs.append(lab.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        enc = lab.tok(msgs, return_tensors='pt', padding=True, add_special_tokens=False).to(lab.dev);rec.clear()
        with torch.no_grad():
            out = lab.model.generate(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=.7, top_p=.8, top_k=20,
                                     pad_token_id=lab.tok.pad_token_id, eos_token_id=stop)
        new = out[:, enc['input_ids'].shape[1]:];H = torch.stack(rec[:new.shape[1]], 1)    # (B, steps, d): state behind each token
        valid = torch.ones_like(new, dtype=torch.bool)
        for b in range(B):
            e = (new[b][:, None] == torch.tensor(stop, device=lab.dev)).any(1).nonzero()
            if len(e):valid[b, e[0, 0]:] = False
        for b in range(B):
            txt = lab.tok.decode(new[b][valid[b]], skip_special_tokens=True).strip() or '...'
            turns[b].append(txt);v = valid[b]
            Hturn[t, b] = (H[b][v].mean(0) if v.any() else H[b].mean(0)).cpu().numpy()
        log(f'turn {t+1}/{a.turns} ({time.time()-t0:.0f} s): {turns[0][-1][:100]!r}')
    hd.remove()
    mu = Hturn.reshape(-1, lab.d).mean(0)
    idx, coef, part = js.decompose(torch.tensor(Hturn.reshape(-1, lab.d)-mu, device=lab.dev))
    Jturn = part.reshape(a.turns, B, lab.d).half().cpu().numpy();idx = idx.reshape(a.turns, B, a.k).cpu().numpy()
    coef = coef.reshape(a.turns, B, a.k).cpu().numpy()
    Jwords = [[[js.words[i_] for i_ in idx[t, b][np.argsort(-coef[t, b])] if coef[t, b][list(idx[t, b]).index(i_)] > 0][:15]
               for b in range(B)] for t in range(a.turns)]
    np.savez_compressed(os.path.join(a.out, 'selftalk.npz'), J=Jturn, H=Hturn.astype(np.float16), mu=mu, idx=idx, coef=coef)
    # ---------------- analysis ----------------
    X = Jturn.astype(np.float32);X = X-X.reshape(-1, lab.d).mean(0);X /= np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-8)
    op = np.array([c[0] for c in convs]);same = op[:, None] == op[None, :];off = ~np.eye(B, dtype=bool)
    conv = []
    for t in range(a.turns):
        S = X[t]@X[t].T;conv.append(dict(within=float(S[same & off].mean()), between=float(S[~same].mean())))
    words_t = [[w for b in range(B) for w in Jwords[t][b][:10]] for t in range(a.turns)]
    text_t = [[w for b in range(B) for w in re.findall(r"[A-Za-z']+", turns[b][t])] for t in range(a.turns)]
    th = dict(J_bliss=[theme(w, BLISS) for w in words_t], J_self=[theme(w, SELF) for w in words_t],
              text_bliss=[theme(w, BLISS) for w in text_t], text_self=[theme(w, SELF) for w in text_t])
    def lead(x, y, k=1):                                                              # corr(x_t, y_{t+k}) - corr(x_t, y_{t-k})
        x, y = np.asarray(x), np.asarray(y);f = np.corrcoef(x[:-k], y[k:])[0, 1];b_ = np.corrcoef(x[k:], y[:-k])[0, 1]
        return float(f), float(b_)
    # echo: a turn that repeats the other speaker's last turn almost verbatim (same first 120 characters)
    same = lambda x, y:x[:120].strip().lower() == y[:120].strip().lower()
    echo = [float(np.mean([same(turns[b][t], turns[b][t-1]) for b in range(B)])) if t else 0. for t in range(a.turns)]
    onset = {o:[next((t for t in range(1, a.turns) if all(same(turns[b][u], turns[b][u-1]) for u in range(t, a.turns))), None)
                for b, (o2, s) in enumerate(convs) if o2 == o] for o in OPENERS}
    from collections import Counter
    top = lambda ts, sel:Counter(w for t in ts for b in range(B) if sel[b] for w in Jwords[t][b][:10]).most_common(20)
    res = dict(model=a.model, read_layer=L, turns=a.turns, seeds=a.seeds, convergence=conv, themes=th,
               lead_bliss=lead(th['J_bliss'], th['text_bliss']), lead_self=lead(th['J_self'], th['text_self']),
               top_first=top(range(min(6, a.turns)), [True]*B), top_last=top(range(max(0, a.turns-6), a.turns), [True]*B),
               top_last_by_opener={o:top(range(max(0, a.turns-6), a.turns), op == o) for o in OPENERS},
               echo=echo, echo_onset=onset, transcripts={f'{o}_{s}':turns[b] for b, (o, s) in enumerate(convs)})
    save_json(res, os.path.join(a.out, 'selftalk.json'));log(f'saved {a.out} in {time.time()-t0:.0f} s')
    print('\nturn  conv within/between | J bliss  J self | text bliss text self')
    for t in range(a.turns):
        print(f"{t+1:4d}  {conv[t]['within']:+.3f} / {conv[t]['between']:+.3f}   | {th['J_bliss'][t]:.3f}  {th['J_self'][t]:.3f} | {th['text_bliss'][t]:.3f}  {th['text_self'][t]:.3f}")
    print('lead (J at t -> text at t+1, text at t -> J at t+1): bliss', res['lead_bliss'], 'self', res['lead_self'])
    print('echo share by turn:', ' '.join(f'{e:.2f}' for e in echo))
    print('echo onset (turn from which every turn repeats the last) by opener:', onset)
    print('first 6 turns, J-space:', [w for w, _ in res['top_first'][:15]])
    print('last 6 turns, J-space: ', [w for w, _ in res['top_last'][:15]])
    for o in OPENERS:print(f'  last 6, {o}:', [w for w, _ in res['top_last_by_opener'][o][:10]])


if __name__ == '__main__':
    main()
