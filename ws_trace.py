"""Trace conditioning and the local-global test under J-space ablation. These are two tests that Dehaene & Naccache
(2026) proposed for the J-space, not yet run on open models.

Trace conditioning. Every line of a memory game starts with a cue word that decides the line's last word, with g
random distractors in between (6 cue -> target pairs, each shown twice). At test, a cue and g new distractors: does
the model produce the target? In animals and humans, bridging a gap needs conscious access, and adjacent pairs do not.
Lindsey's preliminary Claude result: J-space ablation impairs long gaps and spares g = 0.

Local-global (Bekinschtein et al. 2009). Blocks of five words, "x x x x x" or "x x x x y", eight blocks of one kind,
then a test block "x x x x ?". Surprisal of x and y as the fifth word gives a 2 x 2 design: local standard/deviant
(repeat or change) by global standard/deviant (matches the blocks or not). The local effect is automatic. The global
effect, P3b-like, needs the workspace.

Ablation over the workspace band (ws_common.band), at every position: the top-10 J-lens atoms (jspace), 10 random
atoms (rand), or 10 random atoms scaled to remove the same norm as the J-space ablation (matched). The clean run's
top-10 next-token predictions are protected. Capability check: top-1 next-token agreement with the clean run over all
positions.

Predictions, fixed before the run:
  T1  J-space ablation hurts long gaps more than g = 0 (gap x ablation interaction), more than rand and matched do
  T2  J-space ablation shrinks the global effect more than the local effect; rand and matched do not
  T3  J-space ablation keeps most next-token predictions (agreement well above the controls' damage)

    python ws_trace.py --model qwen3-4b --out runs/ws/qwen3-4b/trace
"""
import argparse, os, time
import numpy as np
import torch

from ws_common import Lab, Ablation, band, log, save_json

GAPS = (0, 1, 2, 4, 8)
CONDS = ('clean', 'jspace', 'rand', 'matched')
HEADER = 'Memory game. In every line, the first word decides the last word.\n'


def pools(lab, seed=0):
    """Common single-token lowercase words, split into cues, targets, distractors."""
    rng = np.random.default_rng(seed);ws = [w for w in lab.words if w.islower() and 4 <= len(w) <= 8]
    try:
        from wordfreq import zipf_frequency
        ws = [w for w in ws if zipf_frequency(w, 'en') >= 4.]
    except ImportError:
        pass
    ws = sorted(set(ws));rng.shuffle(ws);ids = lab.word_ids(ws);ws = [w for w, i in zip(ws, ids) if i is not None]
    return ws[:40], ws[40:80], ws[80:]


def trace_prompts(lab, cues, targets, distract, gap, n, rng, M=6, R=2):
    out = []
    for _ in range(n):
        ci = rng.choice(len(cues), M, replace=False);ti = rng.choice(len(targets), M, replace=False)
        pairs = [(cues[c], targets[t]) for c, t in zip(ci, ti)];lines = []
        for _ in range(R):
            for p in rng.permutation(M):
                ds = rng.choice(len(distract), gap, replace=False)
                lines.append(' '.join([pairs[p][0]]+[distract[i] for i in ds]+[pairs[p][1]]))
        rng.shuffle(lines);q = rng.integers(M);ds = rng.choice(len(distract), gap, replace=False)
        test = ' '.join([pairs[q][0]]+[distract[i] for i in ds])
        text = HEADER+'\n'.join(lines)+'\n'+test
        out.append(dict(text=text, target=lab.word_ids([pairs[q][1]])[0], cands=lab.word_ids([p[1] for p in pairs])))
    return out


def lg_prompts(lab, words, n, rng, blocks=8):
    out = []
    for _ in range(n):
        x, y = rng.choice(len(words), 2, replace=False);x, y = words[x], words[y]
        for ctx in ('A', 'B'):
            blk = ' '.join([x]*5) if ctx == 'A' else ' '.join([x]*4+[y])
            text = ' | '.join([blk]*blocks)+' | '+' '.join([x]*4)
            out.append(dict(text=text, ctx=ctx, x=lab.word_ids([x])[0], y=lab.word_ids([y])[0]))
    return out


@torch.no_grad()
def run(lab, abl, prompts, cond, bs, clean_top=None):
    """Last-position log-probs (B, V) for every prompt; for 'clean' also each position's top-10 prediction ids,
    for the ablations the top-1 agreement with the clean run over all positions."""
    lp, tops, agree = [], [], []
    for s in range(0, len(prompts), bs):
        P = prompts[s:s+bs];enc = lab.tok([p['text'] for p in P], return_tensors='pt', padding=True)
        ids = enc['input_ids'].to(lab.dev);am = enc['attention_mask'].to(lab.dev)
        if cond == 'clean':
            abl.mode = None;abl.protect = None
        else:
            abl.mode = cond;abl.protect = lab.dict_index[clean_top[s:s+bs]]               # clean top-10, kept
        out = lab.model(input_ids=ids, attention_mask=am, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        lp.append(torch.log_softmax(out.logits[:, -1].float(), -1).cpu())
        hN = out.hidden_states[-1]                                                     # final-normed last hidden state
        t1 = torch.cat([(hN[:, i:i+256]@lab.model.get_output_embeddings().weight.T).topk(10, dim=-1).indices
                        for i in range(0, hN.shape[1], 256)], 1)                      # (B, T, 10)
        if cond == 'clean':tops.append(t1)
        else:
            agree.append(((t1[..., 0] == clean_top[s:s+bs][..., 0]).float()*am).sum().item()/am.sum().item())
    abl.mode = None
    return torch.cat(lp), (tops if cond == 'clean' else None), (float(np.mean(agree)) if agree else 1.)


def paired(x, y):
    d = np.asarray(x, float)-np.asarray(y, float);n = len(d);se = d.std(ddof=1)/np.sqrt(n) if n > 1 else 0.
    return dict(diff=float(d.mean()), se=float(se), t=float(d.mean()/se) if se > 0 else 0., n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True);ap.add_argument('--out', required=True);ap.add_argument('--vocab', default='english')
    ap.add_argument('--n', type=int, default=128);ap.add_argument('--bs', type=int, default=32);ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args();os.makedirs(a.out, exist_ok=True);t0 = time.time()
    lab = Lab(a.model, vocab=a.vocab);lab.tok.padding_side = 'left'
    if lab.tok.pad_token is None:lab.tok.pad_token = lab.tok.eos_token
    B = band(lab.n);abl = Ablation(lab, B, k=a.k, seed=a.seed)
    cues, targets, distract = pools(lab, a.seed);rng = np.random.default_rng(a.seed)
    log(f'band {B[0]}..{B[-1]} of {lab.n}; pools {len(cues)}/{len(targets)}/{len(distract)}')
    res = dict(model=a.model, band=B, k=a.k, n=a.n, trace={}, local_global={}, agreement={})
    # ---------------- trace conditioning ----------------
    for g in GAPS:
        P = trace_prompts(lab, cues, targets, distract, g, a.n, rng);tgt = torch.tensor([p['target'] for p in P])
        cand = torch.tensor([p['cands'] for p in P]);per = {}
        lp, tops, _ = run(lab, abl, P, 'clean', a.bs)
        tops_dev = tops                                                                # per batch: (B, T, 10) on the GPU
        for cond in CONDS:
            if cond == 'clean':L = lp;ag = 1.
            else:
                Ls, ags = [], []
                for bi, s in enumerate(range(0, len(P), a.bs)):
                    l_, _, ag_ = run(lab, abl, P[s:s+a.bs], cond, a.bs, clean_top=tops_dev[bi]);Ls.append(l_);ags.append(ag_)
                L = torch.cat(Ls);ag = float(np.mean(ags))
            ltgt = L.gather(1, tgt[:, None])[:, 0].numpy();lc = L.gather(1, cand)
            acc_c = (lc.argmax(1) == (cand == tgt[:, None]).float().argmax(1)).float().numpy();top1 = (L.argmax(1) == tgt).float().numpy()
            per[cond] = dict(logp=ltgt.tolist(), acc_cand=acc_c.tolist(), top1=top1.tolist(), agree=ag)
        res['trace'][g] = {c:dict(logp=float(np.mean(v['logp'])), acc_cand=float(np.mean(v['acc_cand'])), top1=float(np.mean(v['top1'])),
                                  agree=v['agree'], drop_logp=paired(per['clean']['logp'], v['logp']) if c != 'clean' else None,
                                  drop_acc=paired(per['clean']['acc_cand'], v['acc_cand']) if c != 'clean' else None)
                           for c, v in per.items()}
        res['trace'][g]['_per'] = per
        log(f'trace gap {g}: ' + ' | '.join(f"{c} acc {res['trace'][g][c]['acc_cand']:.2f} logp {res['trace'][g][c]['logp']:.2f}" for c in CONDS)
            + f" ({time.time()-t0:.0f} s)")
    # interaction: drop at the longest gaps minus drop at gap 0, paired over prompts is not possible (different prompts)
    # accuracy drop as a share of the clean model's margin over chance (1/6): log-prob drops shrink at the floor
    for g in GAPS:
        ca = res['trace'][g]['clean']['acc_cand']
        for c in CONDS[1:]:
            res['trace'][g][c]['rel_acc_drop'] = (ca-res['trace'][g][c]['acc_cand'])/(ca-1/6) if ca > 1/6+.1 else None
    for c in CONDS[1:]:
        d0 = res['trace'][0][c]['drop_logp'];dl = [res['trace'][g][c]['drop_logp'] for g in GAPS if g >= 4]
        diff = np.mean([x['diff'] for x in dl])-d0['diff'];se = np.sqrt(np.mean([x['se']**2 for x in dl])/len(dl)+d0['se']**2)
        res['trace'][f'interaction_{c}'] = dict(diff=float(diff), se=float(se), z=float(diff/se) if se > 0 else 0.)
    # ---------------- local-global ----------------
    words = cues+targets+distract[:200];P = lg_prompts(lab, words, a.n, rng);per = {}
    lp, tops, _ = run(lab, abl, P, 'clean', a.bs)
    for cond in CONDS:
        if cond == 'clean':L = lp;ag = 1.
        else:
            Ls, ags = [], []
            for bi, s in enumerate(range(0, len(P), a.bs)):
                l_, _, ag_ = run(lab, abl, P[s:s+a.bs], cond, a.bs, clean_top=tops[bi]);Ls.append(l_);ags.append(ag_)
            L = torch.cat(Ls);ag = float(np.mean(ags))
        sx = -L.gather(1, torch.tensor([p['x'] for p in P])[:, None])[:, 0].numpy()
        sy = -L.gather(1, torch.tensor([p['y'] for p in P])[:, None])[:, 0].numpy()
        A = np.array([p['ctx'] == 'A' for p in P]);ls_gs, ld_gd = sx[A], sy[A];ls_gd, ld_gs = sx[~A], sy[~A]
        local = ((ld_gs+ld_gd)-(ls_gs+ls_gd))/2;glob = ((ls_gd+ld_gd)-(ls_gs+ld_gs))/2
        per[cond] = dict(local=local, glob=glob)
        res['local_global'][cond] = dict(local=float(local.mean()), local_se=float(local.std(ddof=1)/np.sqrt(len(local))),
                                         glob=float(glob.mean()), glob_se=float(glob.std(ddof=1)/np.sqrt(len(glob))),
                                         cells=dict(LS_GS=float(ls_gs.mean()), LD_GD=float(ld_gd.mean()), LS_GD=float(ls_gd.mean()), LD_GS=float(ld_gs.mean())),
                                         agree=ag)
    for cond in CONDS[1:]:
        res['local_global'][cond]['glob_drop'] = paired(per['clean']['glob'], per[cond]['glob'])
        res['local_global'][cond]['local_drop'] = paired(per['clean']['local'], per[cond]['local'])
        # the GNW question: does ablation hurt the global effect more than the local one?
        res['local_global'][cond]['glob_minus_local_drop'] = paired(per['clean']['glob']-per[cond]['glob'],
                                                                    per['clean']['local']-per[cond]['local'])
    res['removed_norm_share'] = float(np.mean(abl.removed)) if abl.removed else None
    for g in GAPS:res['trace'][g].pop('_per')
    save_json(res, os.path.join(a.out, 'trace.json'));log(f'saved {a.out} in {time.time()-t0:.0f} s')
    print('\ntrace conditioning: accuracy among the 6 targets (drop in log p of the target vs clean)')
    print('gap   ' + '  '.join(f'{c:>22s}' for c in CONDS))
    for g in GAPS:
        r = res['trace'][g]
        rd = lambda c:'' if c == 'clean' or r[c]['rel_acc_drop'] is None else f" {r[c]['rel_acc_drop']:+.0%}"
        print(f'{g:3d}   ' + '  '.join(f"{r[c]['acc_cand']:.2f} ({(r[c]['drop_logp'] or dict(diff=0))['diff']:+.2f}){rd(c)}".rjust(22) for c in CONDS))
    for c in CONDS[1:]:
        it = res['trace'][f'interaction_{c}'];print(f'  {c:8s} drop at gaps 4-8 minus drop at gap 0: {it["diff"]:+.3f} ± {it["se"]:.3f} (z {it["z"]:+.1f})')
    print('\nlocal-global (nats): local effect, global effect; drops vs clean')
    for c in CONDS:
        r = res['local_global'][c];extra = ''
        if c != 'clean':extra = f" | drop local {r['local_drop']['diff']:+.2f}, global {r['glob_drop']['diff']:+.2f}, global-minus-local {r['glob_minus_local_drop']['diff']:+.2f} (t {r['glob_minus_local_drop']['t']:+.1f}) | agree {r['agree']:.3f}"
        print(f"  {c:8s} local {r['local']:+.2f}±{r['local_se']:.2f}  global {r['glob']:+.2f}±{r['glob_se']:.2f}{extra}")


if __name__ == '__main__':
    main()
