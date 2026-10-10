"""Criticality in the J-space: does the workspace show the temperature phase transition of language models?

Nakaishi et al. (2024) found a real phase transition in GPT-2 text near T = 1. Below it, generation falls into
repetition and correlations do not die out. Above it, correlations decay fast. At the transition they decay as a power
law, as in natural language. Here the same is asked of the J-space: the model writes 1024 tokens of free story at 11
temperatures (pure temperature sampling, end-of-text banned), and the J-space content at ~0.7 depth is read at every
token: non-negative pursuit on the English dictionary (K = 25, as in strange_loop.JSpace) of the residual minus the
mean residual over all tokens and temperatures, so a constant "assistant baseline" does not take workspace slots.

Measures per temperature, over seeds:
  text      repetition (share of tokens that close a 4-gram already seen), distinct 2-grams, NLL at T = 1
  C(tau)    autocorrelation of the J-space content (cosine, ensemble mean removed) at lags 1..512; power-law vs
            exponential fit over lags 2..256 (R^2 in log-log vs log-linear), and the summed correlation
  beta      spectral exponent of the J-space trajectory (top 10 components)
  richness  distinct top-1 workspace words and their entropy; workspace occupancy |part| / |h|

Predictions, fixed before the run: power-law decay of C(tau) close to T = 1 (and nowhere else), long correlations
from repetition below it, short ones above it, and the richest workspace near T = 1.

    python ws_temperature.py --model qwen3-4b --seeds 12 --out runs/ws/qwen3-4b/temperature
"""
import argparse, math, os, time
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):os.environ.setdefault(_v, '8')   # 64 BLAS threads on a
import numpy as np                                                                                      # shared host thrash
import torch

from strange_loop import JSpace
from ws_common import Lab, log, save_json

TEMPS = (.3, .5, .7, .8, .9, 1., 1.1, 1.2, 1.3, 1.5, 1.8)
PROMPT = 'Tell me a long story. Just keep writing, without stopping.'


def autocorr(U, maxlag=512):
    """U (T, d) unit vectors -> mean cosine at every lag 1..maxlag (index 0 = lag 1)."""
    G = U@U.T;T = U.shape[0]
    return np.array([np.diagonal(G, k).mean() if k < T else np.nan for k in range(1, maxlag+1)])


def fits(lags, C):
    lags = np.asarray(lags, float);m = (lags >= 2) & (lags <= 256) & np.isfinite(C) & (C > 1e-3)
    if m.sum() < 5:return dict(r2_pow=None, r2_exp=None, exponent=None, xi=None)
    x, y = lags[m], np.log(C[m])
    def r2(X):
        A = np.vstack([X, np.ones_like(X)]).T;coef, *_ = np.linalg.lstsq(A, y, rcond=None);res = y-A@coef
        return float(1-res.var()/max(y.var(), 1e-12)), coef
    rp, cp = r2(np.log(x));re, ce = r2(x)
    return dict(r2_pow=rp, r2_exp=re, exponent=float(-cp[0]), xi=float(-1/ce[0]) if ce[0] < 0 else None)


def beta(X, k=10):
    X = X-X.mean(0);ev, ec = np.linalg.eigh(X@X.T);Y = ec[:, -k:]*np.sqrt(np.clip(ev[-k:], 0, None))
    F = np.abs(np.fft.rfft(Y*np.hanning(len(Y))[:, None], axis=0))**2;f = np.fft.rfftfreq(len(Y));m = (f > 2/len(Y)) & (f < .25)
    return float(-np.polyfit(np.log(f[m]), np.log(F[m].sum(1)+1e-20), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True);ap.add_argument('--out', required=True);ap.add_argument('--vocab', default='english')
    ap.add_argument('--seeds', type=int, default=12);ap.add_argument('--T', type=int, default=1024)
    ap.add_argument('--read_frac', type=float, default=.7);ap.add_argument('--k', type=int, default=25);ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--analyze_only', action='store_true', help='re-run the analysis from temperature.npz (CPU)')
    a = ap.parse_args();os.makedirs(a.out, exist_ok=True);t0 = time.time();torch.set_num_threads(8)
    if a.analyze_only:
        z = np.load(os.path.join(a.out, 'temperature.npz'))
        P, IDX, CF, TOK, NLL, OCC = (z[k] for k in ('part', 'idx', 'coef', 'tok', 'nll', 'occ'))
        from transformers import AutoTokenizer
        from ws_common import MODELS
        analyse(a, P, IDX, CF, TOK, NLL, OCC, AutoTokenizer.from_pretrained(MODELS[a.model][0]), None, t0);return
    lab = Lab(a.model, vocab=a.vocab)
    L = int(round(a.read_frac*lab.n));L = L if L in lab.J else min(lab.J, key=lambda x:abs(x-L))
    js = JSpace(lab.model, lab.tok, lab.J[L].float(), k=a.k, vocab=a.vocab, center=True)
    temps = torch.tensor([t for t in TEMPS for _ in range(a.seeds)], device=lab.dev);B = len(temps)
    msgs = [{'role':'user', 'content':PROMPT}]
    text = lab.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = torch.tensor([lab.tok(text, add_special_tokens=False)['input_ids']], device=lab.dev).expand(B, -1)
    ban = sorted({i for i in [lab.tok.eos_token_id, lab.tok.pad_token_id]+lab.tok.convert_tokens_to_ids(['<|im_end|>', '<|endoftext|>'])
                  if isinstance(i, int) and i >= 0})
    rec = {};h_read = {}
    hd = lab.blocks[L].register_forward_hook(lambda m, i, o:h_read.__setitem__('h', (o[0] if isinstance(o, tuple) else o)[:, -1].float()))
    g = torch.Generator(device=lab.dev);g.manual_seed(a.seed)
    HR = np.zeros((a.T, B, lab.d), np.float16)                                       # raw residuals at the read layer
    TOK = np.zeros((a.T, B), np.int32);NLL = np.zeros((a.T, B), np.float32)
    with torch.no_grad():
        out = lab.model(input_ids=ids, use_cache=True, logits_to_keep=1);cache = out.past_key_values
        for t in range(a.T):
            lg = out.logits[:, -1].float();lg[:, ban] = -math.inf
            lp1 = torch.log_softmax(lg, -1)
            nt = torch.multinomial(torch.softmax(lg/temps[:, None], -1), 1, generator=g)
            NLL[t] = (-lp1.gather(1, nt)[:, 0]).cpu().numpy();TOK[t] = nt[:, 0].cpu().numpy()
            out = lab.model(input_ids=nt, past_key_values=cache, use_cache=True, logits_to_keep=1);cache = out.past_key_values
            HR[t] = h_read['h'].half().cpu().numpy()
            if (t+1) % 128 == 0:log(f'token {t+1}/{a.T} ({time.time()-t0:.0f} s)')
    hd.remove()
    # the workspace content relative to the typical residual, decomposed in large batches after generation
    mu = torch.tensor(HR.reshape(-1, lab.d).astype(np.float32).mean(0), device=lab.dev)
    P = np.zeros((a.T, B, lab.d), np.float16);IDX = np.zeros((a.T, B, a.k), np.int32);CF = np.zeros((a.T, B, a.k), np.float16)
    OCC = np.zeros((a.T, B), np.float32);flat = HR.reshape(-1, lab.d)
    for s0 in range(0, len(flat), 4096):
        x = torch.tensor(flat[s0:s0+4096], device=lab.dev).float()-mu;idx, coef, part = js.decompose(x)
        tt, bb = np.unravel_index(np.arange(s0, s0+len(x)), (a.T, B))
        P[tt, bb] = part.half().cpu().numpy();IDX[tt, bb] = idx.cpu().numpy();CF[tt, bb] = coef.half().cpu().numpy()
        OCC[tt, bb] = (part.norm(dim=-1)/x.norm(dim=-1).clamp_min(1e-6)).cpu().numpy()
    log(f'decomposed {len(flat)} residuals ({time.time()-t0:.0f} s)')
    np.savez_compressed(os.path.join(a.out, 'temperature.npz'), part=P, idx=IDX, coef=CF, tok=TOK, nll=NLL, occ=OCC,
                        temps=temps.cpu().numpy(), mu=mu.cpu().numpy())
    log(f'saved raw ({time.time()-t0:.0f} s)')
    analyse(a, P, IDX, CF, TOK, NLL, OCC, lab.tok, L, t0)


def analyse(a, P, IDX, CF, TOK, NLL, OCC, tokzr, L, t0):
    T_, B_, d = P.shape;a.T = T_
    lags = np.unique(np.round(np.logspace(0, math.log10(512), 40)).astype(int))
    X = P.astype(np.float32);mu_all = X.reshape(-1, d).mean(0);res = dict(model=a.model, read_layer=L, T=a.T, seeds=a.seeds, temps={})
    for ti, tv in enumerate(TEMPS):
        bs = list(range(ti*a.seeds, (ti+1)*a.seeds));mu_t = X[:, bs].reshape(-1, d).mean(0);C, Ct, bt, rep, d2, words, ent = [], [], [], [], [], [], []
        for b in bs:
            tok = TOK[:, b].tolist();seen = set();r = 0
            for i in range(3, len(tok)):
                q = tuple(tok[i-3:i+1]);r += q in seen;seen.add(q)
            rep.append(r/max(1, len(tok)-3));bg = list(zip(tok[:-1], tok[1:]));d2.append(len(set(bg))/len(bg))
            for mu, acc in ((mu_all, C), (mu_t, Ct)):
                U = X[:, b]-mu;U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-8);acc.append(autocorr(U))
            bt.append(beta(X[64:, b]))
            top = IDX[:, b][np.arange(a.T), np.argmax(CF[:, b].astype(np.float32), 1)];words.append(len(set(top.tolist())))
            pr = np.bincount(top)/len(top);pr = pr[pr > 0];ent.append(float(-(pr*np.log(pr)).sum()))
        Cf, Ctf = np.nanmean(C, 0), np.nanmean(Ct, 0);Cm, Ctm = Cf[lags-1], Ctf[lags-1]
        res['temps'][tv] = dict(repetition=float(np.mean(rep)), distinct2=float(np.mean(d2)), nll=float(NLL[:, bs].mean()),
                                C=Cm.tolist(), C_within=Ctm.tolist(), lags=lags.tolist(), fit=fits(lags, Cm), fit_within=fits(lags, Ctm),
                                C_sum=float(np.nansum(Cf)), C_sum_within=float(np.nansum(Ctf)), beta=float(np.mean(bt)),
                                words=float(np.mean(words)), entropy=float(np.mean(ent)),
                                occupancy=float(OCC[:, bs].mean()),
                                sample=tokzr.decode(TOK[:200, bs[0]].tolist())[:600])
        log(f'analysed T = {tv} ({time.time()-t0:.0f} s)')
    save_json(res, os.path.join(a.out, 'temperature.json'));log(f'saved {a.out} in {time.time()-t0:.0f} s')
    print(f"\n{'T':>4} | rep   dist2  nll  | C(1)  C(16) C(128) Csum | pow R2 exp R2 expo  | beta | words entropy occ")
    for tv in TEMPS:
        r = res['temps'][tv];C = lambda k:float(np.interp(k, r['lags'], r['C']));f = r['fit'];q = lambda v, p='.2f':format(v, p) if isinstance(v, float) else '  - '
        print(f"{tv:4.1f} | {r['repetition']:.2f} {r['distinct2']:.2f} {r['nll']:5.2f} | {C(1):.2f} {C(16):5.2f} {C(128):5.2f} {r['C_sum']:6.1f} |"
              f" {q(f['r2_pow'])}  {q(f['r2_exp'])}  {q(f['exponent'])} | {r['beta']:.2f} | {r['words']:5.0f} {r['entropy']:.2f} {r['occupancy']:.3f}")


if __name__ == '__main__':
    main()
