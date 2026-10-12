"""Experiment 1 (protocol section 3): numerical qualification + lens calibration, H1 binding, H2 dispositions, E1-T.
Resumable: every stage writes <out>/<stage>/DONE; finished stages are skipped on restart.
    python e1.py --model qwen3-8b --scale pilot|main [--stages qual,h1,h2,think]"""
import argparse, json, math, pathlib, time
import numpy as np, torch
from common import LM, OUT, log, atomic_npz, atomic_json, seed_all, top_tokens
from tasks import build_all, user_text

KS = (8, 16, 25, 32, 64)


def chat_items(lm, items, query=None):
    """texts + character marks [core_end, evid_end, body_end] for task items (F1 items need a query index)."""
    texts, marks = [], []
    for it in items:
        if it['family'] == 'F1':
            body = it['scenario'];m = dict(core_end=len(body), evid_end=len(body), body_end=len(body))
            q = it['queries'][query if query is not None else 0]['q'];u = f'{body}\nQuestion: {q} Answer with one word.'
        else:
            u, m = user_text(it)
        full = lm.chat(u);off = full.index(u)
        texts.append(full);marks.append([off+m['core_end'], off+m['evid_end'], off+m['body_end']])
    return texts, marks


def bsize(T):
    return int(max(8, min(192, 24000//max(T, 1))))


def encode_fixed(lm, texts, marks, fix):
    """Left-padded batch padded further to a fixed shape (T_fix, B_fix) with dummy rows, so that every row's numbers
    are independent of its batch companions (verified in stage qual). Returns ids, att, pos (marks), n real rows."""
    ids, att, pos = lm.encode(texts, marks);B, T = ids.shape
    if fix is not None:
        Tf, Bf = fix;assert Tf >= T and Bf >= B, (Tf, T, Bf, B)
        if Tf > T:
            ids = torch.cat([torch.full((B, Tf-T), lm.tok.pad_token_id, device=ids.device, dtype=ids.dtype), ids], 1)
            att = torch.cat([torch.zeros((B, Tf-T), device=att.device, dtype=att.dtype), att], 1)
            if pos is not None:pos = pos+(Tf-T)
        if Bf > B:
            ids = torch.cat([ids, ids[:1].expand(Bf-B, -1)], 0);att = torch.cat([att, att[:1].expand(Bf-B, -1)], 0)
            if pos is not None:pos = torch.cat([pos, pos[:1].expand(Bf-B, -1)], 0)
    return ids, att, pos, B


@torch.no_grad()
def run(lm, texts, marks, layers, edits=None, want_reads=True, fix=None, full_reads=False):
    """One batch. Reads (B, 4, d) at [core_end, evid_end, body_end, last] (or full sequences) for each layer.
    edits: dict layer -> (fn(h_at_positions (B, d), pos (B,)) -> new (B, d), column index of the position)."""
    ids, att, pos, B = encode_fixed(lm, texts, marks, fix)
    T = ids.shape[1];last = torch.full((ids.shape[0], 1), T-1, device=lm.dev)
    allpos = torch.cat([pos, last], 1)
    lm.clear()
    if want_reads:
        for l in layers:lm.read[l] = None if full_reads else allpos
    if edits:
        for l, (fn, which) in edits.items():
            p = allpos[:B, which]

            def e(h, fn=fn, p=p):
                r = torch.arange(B, device=h.device);h = h.clone()
                h[r, p] = fn(h[r, p].float(), p).to(h.dtype);return h
            lm.edit[l] = e
    logits = lm.forward(ids, att)[:B, -1]
    reads = {l:lm.store[l][:B] for l in layers} if want_reads else {}
    lm.clear()
    return logits, reads


def max_len(lm, texts):
    n = max(len(x) for x in lm.tok(texts, add_special_tokens=False)['input_ids'])
    return int(8*math.ceil(n/8))


def center(lm, l, x, ptype):
    return x-lm.lens.mu[(l, ptype)].to(x.device)


# ------------------------------------------------------------------------------------------------------------------
def stage_qual(lm, data, out, layers):
    """Numerical qualification, centring means, lens calibration scan."""
    seed_all(0);res = {}
    pool = [it for k in ('F2', 'F3') for it in data[k] if it.get('challenge') == 'none'][:96]
    texts, marks = chat_items(lm, pool);cands = [it['cands'] for it in pool];Tq = max_len(lm, texts)
    l1, _ = run(lm, texts[:64], marks[:64], layers, want_reads=False);l2, _ = run(lm, texts[:64], marks[:64], layers, want_reads=False)
    a = lm.logodds(l1, cands[:64]);b = lm.logodds(l2, cands[:64])
    singles = np.concatenate([lm.logodds(run(lm, texts[i:i+1], marks[i:i+1], layers, want_reads=False)[0], cands[i:i+1])
                              for i in range(16)])
    res['replay_max_abs'] = float(np.abs(a-b).max())
    res['geometry_noise_b1_vs_b64'] = float(np.abs(singles-a[:16]).max())
    # fixed shape: the same 16 prompts with two different sets of companions, both padded to (Tq, 64)
    f1 = lm.logodds(run(lm, texts[:16]+texts[16:64], marks[:16]+marks[16:64], layers, want_reads=False, fix=(Tq, 64))[0][:16], cands[:16])
    f2 = lm.logodds(run(lm, texts[:16]+texts[48:96], marks[:16]+marks[48:96], layers, want_reads=False, fix=(Tq, 64))[0][:16], cands[:16])
    f3 = lm.logodds(run(lm, texts[:16], marks[:16], layers, want_reads=False, fix=(Tq, 64))[0], cands[:16])
    res['fixed_shape_noise_companions'] = float(np.abs(f1-f2).max())
    res['fixed_shape_noise_dummy_rows'] = float(np.abs(f1-f3).max())
    res['logodds_scale_median_abs'] = float(np.median(np.abs(a)))
    log('qual', res)
    acc = {};cnt = {}
    calib = []
    for k in ('F1', 'F2', 'F3', 'F5'):
        its = [it for it in data[k] if it['split'] in ('train', 'pilot') and it.get('challenge', 'none') == 'none']
        calib += its[::max(1, len(its)//300)][:300]
    order = sorted(range(len(calib)), key=lambda i:len(chat_items(lm, [calib[i]])[0][0]))
    for chunk in [order[i:i+48] for i in range(0, len(order), 48)]:
        its = [calib[i] for i in chunk];texts_c, marks_c = chat_items(lm, its)
        _, reads = run(lm, texts_c, marks_c, layers)
        for l in layers:
            for j, pt in ((0, 'core'), (1, 'evid'), (2, 'body'), (3, 'dec')):
                v = reads[l][:, j].double()
                acc[(l, pt)] = acc.get((l, pt), 0)+v.sum(0);cnt[(l, pt)] = cnt.get((l, pt), 0)+v.shape[0]
    slot_texts, slot_marks = slot_prompts(lm, [q['q'] for it in data['F1'][:200] for q in it['queries']])
    for i in range(0, len(slot_texts), 64):
        _, reads = run(lm, slot_texts[i:i+64], slot_marks[i:i+64], layers)
        for l in layers:
            v = reads[l][:, 0].double();acc[(l, 'slot')] = acc.get((l, 'slot'), 0)+v.sum(0)
            cnt[(l, 'slot')] = cnt.get((l, 'slot'), 0)+v.shape[0]
    mu = {f'{l}|{pt}':(acc[(l, pt)]/cnt[(l, pt)]).float().cpu().numpy() for (l, pt) in acc}
    atomic_npz(out/'mu.npz', **mu)
    set_mu(lm, out/'mu.npz')
    logits, reads = run(lm, texts[:64], marks[:64], layers)
    final_top = logits.argmax(-1)
    norm = lm.model.model.norm;scan = {}
    for l in layers:
        h = reads[l][:, 3]
        lj = lm.model.lm_head(norm((h@lm.lens.J[l].T).to(lm.model.dtype))).float().argmax(-1)
        x = center(lm, l, h, 'dec');idx, coef, rec = lm.lens.decompose(l, x, 25)
        share = (rec.norm(dim=-1)**2/x.norm(dim=-1)**2).cpu().numpy()
        cos = torch.nn.functional.cosine_similarity(rec, x, dim=-1).cpu().numpy()
        ex = [dict(prompt=texts[i][-160:], words=lm.lens.words_of(idx[i].cpu().numpy(), coef[i].cpu().numpy(), 12))
              for i in range(0, 64, 8)]
        scan[l] = dict(depth=l/lm.n_layers, lens_final_agree=float((lj == final_top).float().mean()),
                       j_share_median=float(np.median(share)), recon_cos_median=float(np.median(cos)), examples=ex)
        log('layer', l, {k:v for k, v in scan[l].items() if k != 'examples'})
    res['scan'] = scan;res['provenance'] = lm.provenance
    atomic_json(out/'qual.json', res)


def set_mu(lm, path):
    z = np.load(path)
    for k in z.files:
        l, pt = k.split('|');lm.lens.mu[(int(l), pt)] = torch.tensor(z[k], device=lm.dev)


def slot_prompts(lm, questions):
    texts, marks = [], []
    for q in questions:
        u = f'There was a scene.\nQuestion: {q} Answer with one word.';full = lm.chat(u);off = full.index(u)
        e = off+len('There was a scene.');texts.append(full);marks.append([e, e, e])
    return texts, marks


def decompose_all(lm, l, X, ptype):
    """X (N, d) raw residuals -> codes for every K (nested greedy selection order from the K=64 run)."""
    from common import nnls_fista
    out = {}
    for i0 in range(0, X.shape[0], 2048):
        x = center(lm, l, X[i0:i0+2048].to(lm.dev).float(), ptype)
        idx, coef64, _ = lm.lens.decompose(l, x, 64)
        part = dict(idx=idx.cpu().numpy().astype(np.int32))
        for K in KS:
            if K == 64:c = coef64
            else:
                A = lm.lens.D[l][idx[:, :K]].float();G = A@A.transpose(1, 2)+1e-4*torch.eye(K, device=lm.dev)
                bb = (A@x[:, :, None])[:, :, 0];c = nnls_fista(G, bb, torch.linalg.solve(G, bb))
            part[f'coef{K}'] = c.cpu().numpy().astype(np.float32)
        for k, v in part.items():out.setdefault(k, []).append(v)
    return {k:np.concatenate(v) for k, v in out.items()}


@torch.no_grad()
def context_patch(lm, l, tgt_texts, tgt_marks, src_texts, src_marks, cands, fix, conds, mu=None):
    """Patch, at every position whose token is identical in target and source (after the first differing token and up to
    the end of the scenario), the full state / J component / remainder / random-J from the source into the target.
    Returns log-odds of the target's correct answer (n, len(conds)), number of patched positions, and the mu used."""
    n = len(tgt_texts)
    ids_s, att_s, pos_s, _ = encode_fixed(lm, src_texts, src_marks, fix)
    ids_t, att_t, pos_t, _ = encode_fixed(lm, tgt_texts, tgt_marks, fix)
    lm.clear();lm.read[l] = None
    lm.forward(ids_s, att_s);Hs = lm.store[l][:n];lm.clear()
    T = ids_t.shape[1];ar = torch.arange(T, device=lm.dev)[None]
    same = (ids_t[:n] == ids_s[:n])&(att_t[:n] > 0)&(att_s[:n] > 0)
    diff = (~same)&(att_t[:n] > 0)
    first = torch.where(diff.any(1), diff.float().argmax(1), torch.full((n,), T, device=lm.dev))
    end = pos_t[:n, 2]
    mask = same&(ar > first[:, None])&(ar <= end[:, None])&(pos_t[:n, 2] == pos_s[:n, 2])[:, None]
    k_idx, p_idx = torch.nonzero(mask, as_tuple=True)
    S = Hs[k_idx, p_idx]
    if mu is None:mu = S.mean(0) if len(S) else torch.zeros(lm.d, device=lm.dev)
    if len(S):
        _, _, SJ = lm.lens.decompose(l, S-mu, 25)
        Srem = S-mu-SJ;Srand = torch.randn_like(SJ);Srand = Srand/Srand.norm(dim=-1, keepdim=True).clamp_min(1e-6)*SJ.norm(dim=-1, keepdim=True)
    rows_t = [t for t in tgt_texts for _ in conds];rows_m = [m for m in tgt_marks for _ in conds]
    ids, att, pos, B = encode_fixed(lm, rows_t, rows_m, (fix[0], max(fix[1], len(rows_t))))
    nc = len(conds)

    def e(h):
        h = h.clone()
        for c, kind in enumerate(conds):
            if kind == 'none' or not len(S):continue
            rr = k_idx*nc+c;hp = h[rr, p_idx].float()
            if kind == 'full':new = S
            else:
                _, _, Jt = lm.lens.decompose(l, hp-mu, 25)
                new = hp-Jt+SJ if kind == 'J' else (mu+Jt+Srem if kind == 'rem' else hp-Jt+Srand)
            h[rr, p_idx] = new.to(h.dtype)
        return h
    lm.clear();lm.edit[l] = e
    logits = lm.forward(ids, att)[:B, -1];lm.clear()
    lo = lm.logodds(logits, [c for c in cands for _ in conds]).reshape(n, nc)
    return lo, mask.sum(1).cpu().numpy(), mu


def stage_h1(lm, data, out, layers, splits):
    """Binding (H1): states at the end of the scenario; in-context answers; context patch at identical-token positions
    (the causal binding test); scenario-end and decision-position patches; neutral-prompt transplant."""
    items = [it for it in data['F1'] if it['split'] in splits];N = len(items);log('h1 states', N)
    t1 = [chat_items(lm, [it], query=q)[0][0] for it in items for q in (0, 1)];fixF1 = (max_len(lm, t1), 96)
    H = {l:np.zeros((N, 3, lm.d), np.float16) for l in layers}       # [scenario end, decision q0, decision q1]
    lo = np.zeros((N, 2), np.float32)
    for q in (0, 1):
        for i0 in range(0, N, 96):
            its = items[i0:i0+96];texts, marks = chat_items(lm, its, query=q)
            logits, reads = run(lm, texts, marks, layers, fix=fixF1)
            lo[i0:i0+len(its), q] = lm.logodds(logits, [it['queries'][q]['cands'] for it in its])
            for l in layers:
                if q == 0:H[l][i0:i0+len(its), 0] = reads[l][:, 2].cpu().numpy().astype(np.float16)
                H[l][i0:i0+len(its), 1+q] = reads[l][:, 3].cpu().numpy().astype(np.float16)
    meta = dict(sid=[it['sid'] for it in items], sub=[it['sub'] for it in items], template=[it['template'] for it in items],
                binding=[it['binding'] for it in items], split=[it['split'] for it in items])
    codes = {}
    for l in layers:
        c = decompose_all(lm, l, torch.tensor(H[l][:, 0]), 'body');codes.update({f'{l}_{k}':v for k, v in c.items()})
    atomic_npz(out/'states.npz', lo=lo, **{f'H{l}':H[l] for l in layers}, **codes)
    atomic_json(out/'meta.json', meta)
    log('h1 in-context accuracy', float((lo > 0).mean()))
    ev = [i for i, it in enumerate(items) if it['split'] in ('val', 'test', 'pilot')]
    pair = {}
    for i, it in enumerate(items):pair.setdefault((it['sid'], it['template']), {})[it['binding']] = i
    tasks = [(i, pair[(items[i]['sid'], items[i]['template'])][1-items[i]['binding']]) for i in ev
             if len(pair[(items[i]['sid'], items[i]['template'])]) == 2]
    conds = ['none', 'full', 'J', 'rem', 'rand']
    CP = np.full((len(tasks), 2, len(layers), len(conds)), np.nan, np.float32);NP = np.zeros((len(tasks), 2), np.int16)
    for li, l in enumerate(layers):
        mu = None
        for q in (0, 1):
            for c0 in range(0, len(tasks), 16):
                tt = tasks[c0:c0+16]
                tx_t, mk_t = chat_items(lm, [items[a] for a, b in tt], query=q)
                tx_s, mk_s = chat_items(lm, [items[b] for a, b in tt], query=q)
                lo5, npos, mu = context_patch(lm, l, tx_t, mk_t, tx_s, mk_s, [items[a]['queries'][q]['cands'] for a, b in tt],
                                              fixF1, conds, mu)
                CP[c0:c0+len(tt), q, li] = lo5;NP[c0:c0+len(tt), q] = npos
        log('h1 context patch layer', l, {c:float((CP[:, :, li, k] > 0).mean()) for k, c in enumerate(conds)},
            'mean positions', float(NP.mean()))
    SP = np.full((len(tasks), 2, len(layers), len(conds), 2), np.nan, np.float32)
    for li, l in enumerate(layers):
        Hl = torch.tensor(H[l]).float().to(lm.dev)
        for wi, which in enumerate((2, 3)):
            for q in (0, 1):
                col = 0 if which == 2 else 1+q;ptype = 'body' if which == 2 else 'dec'
                mu = lm.lens.mu[(l, ptype)];X = Hl[:, col]
                _, _, XJ = lm.lens.decompose(l, X-mu, 25);Xrem = X-mu-XJ
                Xr = torch.randn_like(XJ);Xr = Xr/Xr.norm(dim=-1, keepdim=True)*XJ.norm(dim=-1, keepdim=True)
                for c0 in range(0, len(tasks), 16):
                    tt = tasks[c0:c0+16];nb = len(tt)
                    texts, marks = chat_items(lm, [items[a] for a, b in tt], query=q)
                    T5 = [t for t in texts for _ in conds];M5 = [m for m in marks for _ in conds]
                    src = torch.tensor([b for a, b in tt], device=lm.dev).repeat_interleave(len(conds))
                    cid = torch.arange(len(conds), device=lm.dev).repeat(nb)

                    def pf(hp, p, src=src, cid=cid, mu=mu, X=X, XJ=XJ, Xrem=Xrem, Xr=Xr, l=l):
                        _, _, Jt = lm.lens.decompose(l, hp-mu, 25);new = hp.clone()
                        for c, name in enumerate(conds):
                            m = cid == c
                            if name == 'none' or not m.any():continue
                            s_ = src[m]
                            new[m] = {'full':X[s_], 'J':hp[m]-Jt[m]+XJ[s_], 'rem':mu+Jt[m]+Xrem[s_], 'rand':hp[m]-Jt[m]+Xr[s_]}[name]
                        return new
                    logits, _ = run(lm, T5, M5, [], edits={l:(pf, which)}, want_reads=False, fix=fixF1)
                    SP[c0:c0+nb, q, li, :, wi] = lm.logodds(logits, [items[a]['queries'][q]['cands'] for a, b in tt
                                                                     for _ in conds]).reshape(nb, len(conds))
        log('h1 single-position layer', l, 'scen-end', {c:float((SP[:, :, li, k, 0] > 0).mean()) for k, c in enumerate(conds)},
            'decision', {c:float((SP[:, :, li, k, 1] > 0).mean()) for k, c in enumerate(conds)})
    atomic_npz(out/'patch.npz', CP=CP, NP=NP, SP=SP, tasks=np.array(tasks), conds=np.array(conds), layers=np.array(layers))
    tconds = ['none', 'full', 'J', 'rem', 'bag', 'shuf', 'rand', 'donor'];rng = np.random.default_rng(1)
    TR = np.full((len(ev), 2, len(layers), len(tconds)), np.nan, np.float32)
    fixS = (max_len(lm, slot_prompts(lm, [q['q'] for i in ev for q in items[i]['queries']])[0]), 192)
    for li, l in enumerate(layers):
        X = torch.tensor(H[l][ev, 0]).float().to(lm.dev);xs = center(lm, l, X, 'body')
        idx, coef, Jsrc = lm.lens.decompose(l, xs, 25);rem_src = xs-Jsrc
        bag = lm.lens.synth(l, idx, (coef > 0).float())
        bag = bag*(Jsrc.norm(dim=-1, keepdim=True)/bag.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        perm = torch.argsort(torch.rand(coef.shape, device=lm.dev), 1);shuf = lm.lens.synth(l, idx, torch.gather(coef, 1, perm))
        shuf = shuf*(Jsrc.norm(dim=-1, keepdim=True)/shuf.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        rnd = torch.randn_like(Jsrc);rnd = rnd/rnd.norm(dim=-1, keepdim=True)*Jsrc.norm(dim=-1, keepdim=True)
        donor = torch.tensor([int(rng.choice([j for j in range(len(ev)) if items[ev[j]]['sid'] != items[ev[i]]['sid']]))
                              for i in range(len(ev))], device=lm.dev)
        mu_s = lm.lens.mu[(l, 'slot')]
        for q in (0, 1):
            for i0 in range(0, len(ev), 24):
                ii = list(range(i0, min(len(ev), i0+24)));nb = len(ii)
                qs = [items[ev[i]]['queries'][q] for i in ii]
                texts, marks = slot_prompts(lm, [x['q'] for x in qs])
                T8 = [t for t in texts for _ in tconds];M8 = [m for m in marks for _ in tconds]
                src = torch.tensor(ii, device=lm.dev).repeat_interleave(len(tconds))
                cid = torch.arange(len(tconds), device=lm.dev).repeat(nb)

                def fn(hp, p, l=l, src=src, cid=cid, X=X, Jsrc=Jsrc, rem_src=rem_src, bag=bag, shuf=shuf, rnd=rnd, donor=donor, mu_s=mu_s):
                    _, _, Jt = lm.lens.decompose(l, hp-mu_s, 25);new = hp.clone()
                    for c, name in enumerate(tconds):
                        m = cid == c
                        if not m.any() or name == 'none':continue
                        s_ = src[m]
                        if name == 'full':new[m] = X[s_]
                        elif name == 'rem':new[m] = mu_s+Jt[m]+rem_src[s_]
                        else:
                            v = {'J':Jsrc[s_], 'bag':bag[s_], 'shuf':shuf[s_], 'rand':rnd[s_], 'donor':Jsrc[donor[s_]]}[name]
                            new[m] = hp[m]-Jt[m]+v
                    return new
                logits, _ = run(lm, T8, M8, [], edits={l:(fn, 0)}, want_reads=False, fix=fixS)
                TR[ii, q, li] = lm.logodds(logits, [x['cands'] for x in qs for _ in tconds]).reshape(nb, len(tconds))
        log('h1 transplant layer', l, {c:float((TR[:, :, li, k] > 0).mean()) for k, c in enumerate(tconds)})
    atomic_npz(out/'transplant.npz', TR=TR, ev=np.array(ev), conds=np.array(tconds), layers=np.array(layers))
    f3 = [it for it in data['F3'] if it['split'] in splits and it['split'] != 'train' and it['challenge'] == 'none'
          and it['evidence'] in ('none', 'tb')]
    idx3 = {(it['sid'], it['evidence'], it['qkind'], it['binding']):i for i, it in enumerate(f3)}
    tasks3 = [(i, idx3[(it['sid'], it['evidence'], it['qkind'], 1-it['binding'])]) for i, it in enumerate(f3)
              if (it['sid'], it['evidence'], it['qkind'], 1-it['binding']) in idx3]
    tx3, mk3 = chat_items(lm, f3);fix3 = (max_len(lm, tx3), 96)
    P3 = np.full((len(tasks3), len(layers), len(conds)), np.nan, np.float32);N3 = np.zeros(len(tasks3), np.int16)
    for li, l in enumerate(layers):
        mu = None
        for c0 in range(0, len(tasks3), 16):
            tt = tasks3[c0:c0+16]
            lo5, npos, mu = context_patch(lm, l, [tx3[a] for a, b in tt], [mk3[a] for a, b in tt], [tx3[b] for a, b in tt],
                                          [mk3[b] for a, b in tt], [f3[a]['cands'] for a, b in tt], fix3, conds, mu)
            P3[c0:c0+len(tt), li] = lo5;N3[c0:c0+len(tt)] = npos
        log('h1 F3 context patch layer', l, {c:float(np.nanmean(P3[:, li, k] > 0)) for k, c in enumerate(conds)})
    atomic_npz(out/'f3patch.npz', P3=P3, N3=N3, tasks=np.array(tasks3))
    atomic_json(out/'f3meta.json', [{k:it[k] for k in ('sid', 'evidence', 'qkind', 'binding', 'split', 'cands')} for it in f3])


def stage_h2(lm, data, out, layers, splits, int_layers):
    """Dispositions: states before the challenge; responses to text and internal challenges (fixed batch shape)."""
    for fam in ('F2', 'F3'):
        fo = out/fam;fo.mkdir(parents=True, exist_ok=True)
        if (fo/'DONE').exists():continue
        allit = [it for it in data[fam] if it['split'] in splits and (fam == 'F2' or (it['binding'] == 0 and it['qkind'] == 'belief'))]
        states = [it for it in allit if it['challenge'] == 'none']
        key = lambda it:(it['sid'], it['evidence'], it['edir'])
        sidx = {key(it):i for i, it in enumerate(states)};N = len(states);log(fam, 'states', N)
        fix = (max_len(lm, chat_items(lm, allit)[0]), 96)
        H = {l:np.zeros((N, 2, lm.d), np.float16) for l in layers}
        lo0 = np.zeros(N, np.float32);tops_i = np.zeros((N, 20), np.int32);tops_v = np.zeros((N, 20), np.float16)
        for c0 in range(0, N, 96):
            chunk = list(range(c0, min(N, c0+96)));its = [states[i] for i in chunk];texts, marks = chat_items(lm, its)
            logits, reads = run(lm, texts, marks, layers, fix=fix)
            lo0[chunk] = lm.logodds(logits, [it['cands'] for it in its])
            ti, tv = top_tokens(lm, logits);tops_i[chunk] = ti;tops_v[chunk] = tv
            for l in layers:H[l][chunk] = reads[l][:, [1, 3]].cpu().numpy().astype(np.float16)
        chal_names = [c for c in dict.fromkeys(it['challenge'] for it in allit) if c != 'none']
        LC = np.full((N, len(chal_names)), np.nan, np.float32);cdir = np.zeros((N, len(chal_names)), np.int8)
        ch_items = [it for it in allit if it['challenge'] != 'none']
        for c0 in range(0, len(ch_items), 96):
            its = ch_items[c0:c0+96];texts, marks = chat_items(lm, its)
            logits, _ = run(lm, texts, marks, layers, want_reads=False, fix=fix)
            v = lm.logodds(logits, [it['cands'] for it in its])
            for it, x in zip(its, v):
                i = sidx[key(it)];c = chal_names.index(it['challenge']);LC[i, c] = x
                cdir[i, c] = it['cdir'] if it['cdir'] is not None else (1-it['edir'] if it['edir'] is not None else 1)

        def ref_of(it):
            d = it['edir'];cd = (1-d) if d is not None else (int(it['sid'].split('-')[-1]) % 2)
            if fam == 'F2':return sidx.get((it['sid'], 'strong', cd)), cd
            return sidx.get((it['sid'], 'tb' if cd == 1 else 'weak1', cd)), cd
        refs = [ref_of(it) for it in states]
        alphas = (0.5, 1.0);icond = [('none', 0.)]+[(kind, a) for kind in ('J', 'res') for a in alphas]
        LI = np.full((N, len(int_layers), len(icond)), np.nan, np.float32);idir = np.array([r[1] for r in refs], np.int8)
        for li, l in enumerate(int_layers):
            Hd = torch.tensor(H[l][:, 1]).float().to(lm.dev)
            xd = center(lm, l, Hd, 'dec');_, _, Jd = lm.lens.decompose(l, xd, 25)
            valid = [i for i in range(N) if refs[i][0] is not None]
            for chunk in [valid[i:i+19] for i in range(0, len(valid), 19)]:
                its = [states[i] for i in chunk];texts, marks = chat_items(lm, its);nb = len(chunk)
                Tn = [t for t in texts for _ in icond];Mn = [m for m in marks for _ in icond]
                me = torch.tensor(chunk, device=lm.dev).repeat_interleave(len(icond))
                rf = torch.tensor([refs[i][0] for i in chunk], device=lm.dev).repeat_interleave(len(icond))
                cid = torch.arange(len(icond), device=lm.dev).repeat(nb)

                def fn(hp, p, me=me, rf=rf, cid=cid, Hd=Hd, Jd=Jd):
                    new = hp.clone()
                    for c, (kind, a) in enumerate(icond):
                        if kind == 'none':continue
                        m = cid == c
                        d = (Jd[rf[m]]-Jd[me[m]]) if kind == 'J' else (Hd[rf[m]]-Hd[me[m]])
                        new[m] = hp[m]+a*d
                    return new
                logits, _ = run(lm, Tn, Mn, [], edits={l:(fn, 3)}, want_reads=False, fix=fix)
                LI[chunk, li] = lm.logodds(logits, [it['cands'] for it in its for _ in icond]).reshape(nb, len(icond))
        codes = {}
        for l in layers:
            for j, pt in ((0, 'evid'), (1, 'dec')):
                c = decompose_all(lm, l, torch.tensor(H[l][:, j]), pt);codes.update({f'{l}_{pt}_{k}':v for k, v in c.items()})
        atomic_npz(fo/'states.npz', lo0=lo0, LC=LC, cdir=cdir, LI=LI, idir=idir, tops_i=tops_i, tops_v=tops_v,
                   **{f'H{l}':H[l] for l in layers}, **codes)
        atomic_json(fo/'meta.json', dict(chal_names=chal_names, icond=[f'{k}{a}' for k, a in icond], int_layers=int_layers, fix=fix,
                                         states=[{k:it[k] for k in ('sid', 'sub', 'split', 'template', 'evidence', 'edir', 'cands')}
                                                 for it in states]))
        (fo/'DONE').write_text('ok')
        log(fam, 'done; internal none vs lo0 max diff', float(np.nanmax(np.abs(LI[:, 0, 0]-lo0))),
            'acc strong', float(np.mean([(lo0[i] > 0) == (states[i]['edir'] == 0) for i in range(N)
                                         if states[i]['evidence'] in ('strong', 'tb')])))


def stage_h1full(lm, data, out, layers, splits, full_layers):
    """Robustness check (protocol section 1): the decision-position patch with the FULL-token J dictionary (every
    non-special token, including the capitalised no-space answer tokens that the English-word dictionary lacks)."""
    from common import Lens
    LF = Lens(lm.model, lm.tok, lm.lens_path, full_layers, vocab='full', dev=lm.dev)
    for k_, v_ in lm.lens.mu.items():LF.mu[k_] = v_
    items = [it for it in data['F1'] if it['split'] in splits];h1 = np.load(out.parent/'h1'/'states.npz')
    pair = {}
    for i, it in enumerate(items):pair.setdefault((it['sid'], it['template']), {})[it['binding']] = i
    ev = [i for i, it in enumerate(items) if it['split'] in ('val', 'test', 'pilot')]
    tasks = [(i, pair[(items[i]['sid'], items[i]['template'])][1-items[i]['binding']]) for i in ev
             if len(pair[(items[i]['sid'], items[i]['template'])]) == 2]
    t1 = [chat_items(lm, [it], query=q)[0][0] for it in items for q in (0, 1)];fixF1 = (max_len(lm, t1), 96)
    conds = ['none', 'full', 'J', 'rem', 'rand'];SP = np.full((len(tasks), 2, len(full_layers), len(conds)), np.nan, np.float32)
    words = {}
    for li, l in enumerate(full_layers):
        Hl = torch.tensor(h1[f'H{l}']).float().to(lm.dev);mu = lm.lens.mu[(l, 'dec')]
        for q in (0, 1):
            X = Hl[:, 1+q];XJ = torch.zeros_like(X)
            for i0 in range(0, X.shape[0], 512):
                idx_, coef_, XJ[i0:i0+512] = LF.decompose(l, X[i0:i0+512]-mu, 25)
                if q == 0 and i0 == 0:words[l] = [LF.words_of(idx_[j].cpu().numpy(), coef_[j].cpu().numpy(), 12) for j in range(0, 64, 8)]
            Xrem = X-mu-XJ;Xr = torch.randn_like(XJ);Xr = Xr/Xr.norm(dim=-1, keepdim=True)*XJ.norm(dim=-1, keepdim=True)
            for c0 in range(0, len(tasks), 16):
                tt = tasks[c0:c0+16];nb = len(tt)
                texts, marks = chat_items(lm, [items[a] for a, b in tt], query=q)
                T5 = [t for t in texts for _ in conds];M5 = [m for m in marks for _ in conds]
                src = torch.tensor([b for a, b in tt], device=lm.dev).repeat_interleave(len(conds))
                cid = torch.arange(len(conds), device=lm.dev).repeat(nb)

                def pf(hp, p, src=src, cid=cid, mu=mu, X=X, XJ=XJ, Xrem=Xrem, Xr=Xr, l=l):
                    _, _, Jt = LF.decompose(l, hp-mu, 25);new = hp.clone()
                    for c, name in enumerate(conds):
                        m = cid == c
                        if name == 'none' or not m.any():continue
                        s_ = src[m]
                        new[m] = {'full':X[s_], 'J':hp[m]-Jt[m]+XJ[s_], 'rem':mu+Jt[m]+Xrem[s_], 'rand':hp[m]-Jt[m]+Xr[s_]}[name]
                    return new
                logits, _ = run(lm, T5, M5, [], edits={l:(pf, 3)}, want_reads=False, fix=fixF1)
                SP[c0:c0+nb, q, li] = lm.logodds(logits, [items[a]['queries'][q]['cands'] for a, b in tt for _ in conds]).reshape(nb, len(conds))
        log('h1full decision layer', l, {c:float((SP[:, :, li, k] > 0).mean()) for k, c in enumerate(conds)}, 'words', words[l][:2])
    atomic_npz(out/'patch_full.npz', SP=SP, tasks=np.array(tasks), conds=np.array(conds), layers=np.array(full_layers))
    atomic_json(out/'words_full.json', {str(k):v for k, v in words.items()})


EPI = ('ambiguous ambiguity unknown unclear uncertain uncertainty unable neither both either possibly perhaps maybe '
       'conflicting contradictory contradiction disputed inconclusive insufficient cannot unsure doubt doubtful undecided').split()


def stage_epi(lm, data, out, layers, splits, steer_layers):
    """Exploratory (amendment 13): causal test of the epistemic concepts seen in the decision-position workspace. Adds
    alpha * m_J * (unit mean of the epistemic-word atoms) at the decision position; controls: same-size random word sets
    with the same norm. Outcome: answer log-odds of F2/F3 states (confidence = |log-odds|)."""
    wi = lm.lens.word_index
    epi = sorted({wi[w] for w0 in EPI for w in (w0, w0.capitalize()) if w in wi})
    log('epistemic atoms', len(epi), [lm.lens.words[i] for i in epi])
    rng = np.random.default_rng(13)
    for fam in ('F2', 'F3'):
        states = [it for it in data[fam] if it['split'] in splits and it['split'] != 'train' and it['challenge'] == 'none'
                  and (fam == 'F2' or (it['binding'] == 0 and it['qkind'] == 'belief'))]
        fix = (max_len(lm, chat_items(lm, states)[0]), 96);N = len(states)
        conds = [('none', 0.)]+[('epi', a) for a in (-1., -.5, .5, 1.)]+[('rand', a) for a in (-1., 1.)]
        LO = np.full((N, len(steer_layers), len(conds)), np.nan, np.float32)
        for li, l in enumerate(steer_layers):
            D = lm.lens.D[l].float();ve = D[epi].mean(0);ve = ve/ve.norm()
            rsets = [rng.choice(D.shape[0], len(epi), replace=False) for _ in range(16)]
            vr = torch.stack([D[torch.tensor(r_, device=lm.dev)].mean(0) for r_ in rsets]);vr = vr/vr.norm(dim=-1, keepdim=True)
            # natural J norm at this layer's decision position, for the scale
            texts, marks = chat_items(lm, states[:96]);_, rd = run(lm, texts, marks, [l], fix=fix)
            _, _, jr = lm.lens.decompose(l, center(lm, l, rd[l][:, 3], 'dec'), 25);mJ = float(jr.norm(dim=-1).median())
            for c0 in range(0, N, 13):
                ii = list(range(c0, min(N, c0+13)));nb = len(ii);its = [states[i] for i in ii];texts, marks = chat_items(lm, its)
                Tn = [t for t in texts for _ in conds];Mn = [m for m in marks for _ in conds]
                cid = torch.arange(len(conds), device=lm.dev).repeat(nb)
                rid = torch.tensor([int(rng.integers(16)) for _ in range(len(Tn))], device=lm.dev)

                def fn(hp, p, cid=cid, rid=rid, ve=ve, vr=vr, mJ=mJ):
                    new = hp.clone()
                    for c, (kind, a) in enumerate(conds):
                        if kind == 'none':continue
                        m = cid == c;v = ve[None].expand(int(m.sum()), -1) if kind == 'epi' else vr[rid[m]]
                        new[m] = hp[m]+a*mJ*v
                    return new
                logits, _ = run(lm, Tn, Mn, [], edits={l:(fn, 3)}, want_reads=False, fix=fix)
                LO[ii, li] = lm.logodds(logits, [it['cands'] for it in its for _ in conds]).reshape(nb, len(conds))
            log('epi', fam, l, 'mean |lo| by cond', {f'{k}{a:g}':float(np.nanmean(np.abs(LO[:, li, j]))) for j, (k, a) in enumerate(conds)})
        atomic_npz(out/f'{fam}_epi.npz', LO=LO, conds=np.array([f'{k}{a:g}' for k, a in conds]), layers=np.array(steer_layers),
                   epi=np.array(epi))
        atomic_json(out/f'{fam}_meta.json', [{k:it[k] for k in ('sid', 'evidence', 'edir', 'split', 'cands')} for it in states])


def stage_think(lm, data, out, layers, n=30, budget=1024):
    """E1-T: matched subset with thinking on; J codes at the end of thinking vs the non-thinking decision position."""
    rec = []
    for fam in ('F2', 'F3'):
        its = [it for it in data[fam] if it['split'] in ('val', 'pilot') and it['challenge'] == 'none'
               and (fam == 'F2' or (it['binding'] == 0 and it['qkind'] == 'belief'))][::7][:n]
        for i0 in range(0, len(its), 30):
            b = its[i0:i0+30]
            texts = [lm.chat(user_text(it)[0], thinking=True) for it in b]
            enc = lm.tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to(lm.dev)
            lm.clear()
            with torch.no_grad():
                g = lm.model.generate(**enc, max_new_tokens=budget, do_sample=False, pad_token_id=lm.tok.pad_token_id)
            gen = lm.tok.batch_decode(g[:, enc['input_ids'].shape[1]:], skip_special_tokens=False)
            for it, t, gtxt in zip(b, texts, gen):
                closed = '</think>' in gtxt;think = gtxt.split('</think>')[0]
                full = t+think+('' if closed else '\n')+'</think>\n\n'
                logits, reads = run(lm, [full], [[len(full)-1]*3], layers)
                lo = float(lm.logodds(logits, [it['cands']])[0]);codes = {}
                for l in layers:
                    x = center(lm, l, reads[l][:, 3], 'dec');idx, coef, _ = lm.lens.decompose(l, x, 25)
                    codes[l] = lm.lens.words_of(idx[0].cpu().numpy(), coef[0].cpu().numpy(), 25)
                after = gtxt.split('</think>')[1][:40] if closed else ''
                rec.append(dict(family=fam, sid=it['sid'], evidence=it['evidence'], edir=it['edir'], closed=closed,
                                n_think_tokens=len(lm.tok.encode(think)), logodds_think=lo, codes=codes, answer_text=after,
                                think=think[-800:]))
    atomic_json(out/'think.json', rec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='qwen3-8b');p.add_argument('--scale', default='pilot')
    p.add_argument('--stages', default='qual,h1,h2,think');p.add_argument('--cache_dir', default=None)
    p.add_argument('--dev', default='cuda');p.add_argument('--fp32', action='store_true');p.add_argument('--lens_path', default=None)
    p.add_argument('--tag', default=None)
    a = p.parse_args()
    out = OUT/'e1'/a.model/(a.tag or a.scale);out.mkdir(parents=True, exist_ok=True)
    lm = LM(a.model, cache_dir=a.cache_dir, dev=a.dev, dtype=torch.float32 if a.fp32 else torch.bfloat16,
            lens_path=a.lens_path);layers = lm.cfg['layers']
    data = build_all(a.scale)
    atomic_json(out/'provenance.json', dict(lm.provenance, scale=a.scale, n_items={k:len(v) for k, v in data.items()}))
    splits = ('pilot',) if a.scale in ('pilot', 'smoke') else ('train', 'val', 'test')
    for st in a.stages.split(','):
        so = out/st;so.mkdir(exist_ok=True)
        if (so/'DONE').exists():
            log('skip', st)
            if st == 'qual':set_mu(lm, so/'mu.npz')
            continue
        if st != 'qual':set_mu(lm, out/'qual'/'mu.npz')
        t0 = time.time();log('stage', st)
        if st == 'qual':stage_qual(lm, data, so, layers)
        elif st == 'h1':stage_h1(lm, data, so, layers, splits)
        elif st == 'h2':stage_h2(lm, data, so, layers, splits, int_layers=[layers[1], layers[2]])
        elif st == 'think':stage_think(lm, data, so, layers)
        elif st == 'epi':stage_epi(lm, data, so, layers, splits, steer_layers=[layers[2], layers[3]])
        elif st == 'h1full':stage_h1full(lm, data, so, layers, splits, full_layers=[layers[2], layers[3]])
        (so/'DONE').write_text(f'{time.time()-t0:.0f}s');log('stage done', st, f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
