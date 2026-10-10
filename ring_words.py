"""Do the J-space words of the ring mean anything? An analysis of the recorded trajectories, before any new run.

Every step of every mind stores its 25 active J-space words and their coefficients (non-negative pursuit on the
English dictionary, of the read-layer state minus the typical state). The J-lens labels cannot vouch for themselves,
so meaning is checked in an independent space: each word is embedded with a sentence-embedding model
(all-MiniLM-L6-v2) that never saw these trajectories. Per condition, over the last 300 steps:

  coherence     are the words active together related? Coefficient-weighted mean pairwise cosine of their
                embeddings, minus the same for random 25-word sets drawn from the words the run uses (by usage)
  persistence   weighted overlap of the active words at lags 1, 10, 100; distinct words a mind uses
  semantic PR   the participation ratio of the coefficient-weighted embedding centroid in 100-step windows. If the
                extra J-space dimensions that error feedback explores carry meaning, this rises too; if they are
                meaningless, it does not
  step size     semantic distance between successive centroids
  transitions   the dominant semantic cluster per step (k-means on the embeddings): entropy of the transitions, and
                whether minds of one condition visit the same clusters (reproducibility across seeds)
  shared        how many of a mind's words another mind of the same condition also holds at the same step
  individual    the words a mind holds that fewer than a quarter of the other minds hold at that step: their share of
                the content, their coherence against the null, and whether they persist 10 steps later
  abstractness  coefficient-weighted position of the words on an abstract-concrete axis (idea, theory, meaning...
                against table, apple, stone...), and on a mind/self axis (think, aware, reflect, self... against the
                run's own vocabulary): is the loop's content more abstract, or more about the mind?

    python ring_words.py <words.npz with idx, coef> <meta.json> [out.json]
"""
import json, os, sys
import numpy as np


def embed(words, cache):
    """Word embeddings from all-MiniLM-L6-v2, or from OpenAI's text-embedding-3-large when EMBEDDER=openai (key in
    OPENAI_API_KEY)."""
    use_openai = os.environ.get('EMBEDDER') == 'openai'
    cache = cache.replace('.npz', '_openai.npz') if use_openai else cache
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z['words']) == list(words):return z['E']
    if use_openai:
        import httpx
        out = []
        for i in range(0, len(words), 1000):
            r = httpx.post('https://api.openai.com/v1/embeddings', timeout=120,
                           headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY']},
                           json={'model':'text-embedding-3-large', 'input':list(words[i:i+1000])})
            r.raise_for_status();out += [d['embedding'] for d in sorted(r.json()['data'], key=lambda d:d['index'])]
        E = np.array(out, np.float32);E /= np.linalg.norm(E, axis=1, keepdims=True)
        np.savez(cache, words=np.array(words, dtype=object), E=E)
        return E
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    m = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').eval();out = []
    with torch.no_grad():
        for i in range(0, len(words), 512):
            enc = tok(list(words[i:i+512]), padding=True, return_tensors='pt');h = m(**enc).last_hidden_state
            e = (h*enc['attention_mask'][..., None]).sum(1)/enc['attention_mask'].sum(1, keepdim=True)
            out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
    E = np.concatenate(out);np.savez(cache, words=np.array(words, dtype=object), E=E)
    return E


def pr(V):
    V = V-V.mean(0);ev = np.clip(np.linalg.eigvalsh(V@V.T), 0, None)
    return float(ev.sum()**2/max((ev**2).sum(), 1e-12))


def main():
    wz, mf = sys.argv[1], sys.argv[2];out = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(wz)[0]+'_words.json'
    z = np.load(wz);idx, coef = z['idx'], z['coef'].astype(np.float32);M = json.load(open(mf, encoding='utf-8'))
    words = M['words'];rows = M['rows'];T, B, K = idx.shape;late = slice(T-300, T)
    used = np.unique(idx);E_used = embed([words[i] for i in used], os.path.join(os.path.dirname(wz) or '.', 'minilm_cache.npz'))
    pos = {w:i for i, w in enumerate(used)};E = np.zeros((len(words), E_used.shape[1]), np.float32);E[used] = E_used
    conds = list(dict.fromkeys(r['cond'] for r in rows));seeds = sorted({r['seed'] for r in rows})
    row = {(r['cond'], r['seed']):i for i, r in enumerate(rows)}
    W = np.clip(coef, 0, None);W = W/np.maximum(W.sum(-1, keepdims=True), 1e-9)              # weights over the 25 words
    cent = np.zeros((T, B, E.shape[1]), np.float32)                                       # in time chunks: E[idx] is big
    for t0 in range(0, T, 10):cent[t0:t0+10] = np.einsum('tbk,tbkd->tbd', W[t0:t0+10], E[idx[t0:t0+10]])
    cent /= np.maximum(np.linalg.norm(cent, axis=-1, keepdims=True), 1e-9)
    # usage over the whole run, for the null sets and the clusters
    usage = np.bincount(idx[late].ravel(), weights=W[late].ravel(), minlength=len(words));p_use = usage/usage.sum()
    rng = np.random.default_rng(0)

    def coherence(ix, w):
        V = E[ix];S = V@V.T;ww = np.outer(w, w);np.fill_diagonal(ww, 0)
        return float((S*ww).sum()/max(ww.sum(), 1e-9))
    null = [coherence(rng.choice(len(words), K, replace=False, p=p_use), np.full(K, 1/K)) for _ in range(2000)]
    null_mu = float(np.mean(null))
    # semantic clusters of the words in use
    from numpy.linalg import norm
    top_used = np.array([i for i in np.argsort(-usage)[:3000] if usage[i] > 0]);Xc = E[top_used];k = 24
    C0 = [Xc[rng.integers(len(Xc))]]
    for _ in range(k-1):                                                                     # k-means++ seeding
        d = np.clip(1-np.max(Xc@np.stack(C0).T, 1), 0, None);C0.append(Xc[rng.choice(len(Xc), p=d/d.sum())])
    C0 = np.stack(C0)
    for _ in range(30):
        lab = np.argmax(Xc@C0.T, 1);C0 = np.stack([Xc[lab == j].mean(0) if (lab == j).any() else C0[j] for j in range(k)])
        C0 /= np.maximum(norm(C0, axis=1, keepdims=True), 1e-9)
    cl = np.argmax(E@C0.T, 1)                                                              # cluster of every word
    dom = np.zeros((T, B), int)
    for t in range(T):
        for b in range(B):dom[t, b] = np.bincount(cl[idx[t, b]], weights=W[t, b], minlength=k).argmax()
    # two axes in the same embedding space: abstract vs concrete, and mind/self words against the run's vocabulary
    ABS = 'idea concept theory notion meaning principle belief reason abstraction essence possibility truth logic value quality'.split()
    CON = 'table apple dog car stone chair bottle hammer tree shoe bread door cup rock horse'.split()
    META = 'think thinking thought mind aware awareness reflect reflection self consciousness cognition introspection attention know perceive'.split()
    TEST_A, TEST_C = 'justice freedom knowledge purpose concept'.split(), 'spoon carrot window pillow bicycle'.split()
    A = embed(ABS+CON+META+TEST_A+TEST_C, os.path.join(os.path.dirname(wz) or '.', 'anchors_cache.npz'))
    nA, nC, nM = len(ABS), len(CON), len(META)
    ax_abs = A[:nA].mean(0)-A[nA:nA+nC].mean(0);ax_abs /= np.linalg.norm(ax_abs)
    vocab_mean = (E[used]*p_use[used, None]).sum(0)/p_use[used].sum()
    ax_meta = A[nA+nC:nA+nC+nM].mean(0)-vocab_mean;ax_meta /= np.linalg.norm(ax_meta)
    s_abs, s_meta = E@ax_abs, E@ax_meta
    check = dict(abstract_test=float((A[-10:-5]@ax_abs).mean()), concrete_test=float((A[-5:]@ax_abs).mean()))
    res = dict(null_coherence=null_mu, axis_check=check, conds={})
    for c in conds:
        bs = [row[c, s] for s in seeds];m = {}
        m['coherence'] = [np.mean([coherence(idx[t, b], W[t, b]) for t in range(T-300, T, 5)])-null_mu for b in bs]
        def overlap(lag):
            return [np.mean([float(np.minimum(np.bincount(idx[t, b], W[t, b], len(words))[np.union1d(idx[t, b], idx[t+lag, b])],
                                              np.bincount(idx[t+lag, b], W[t+lag, b], len(words))[np.union1d(idx[t, b], idx[t+lag, b])]).sum())
                             for t in range(T-300, T-lag, 10)]) for b in bs]
        m['overlap1'], m['overlap10'], m['overlap100'] = overlap(1), overlap(10), overlap(100)
        m['distinct'] = [len(np.unique(idx[late, b])) for b in bs]
        m['sem_pr'] = [np.mean([pr(cent[w:w+100, b]) for w in range(T-300, T-99, 50)]) for b in bs]
        m['sem_step'] = [float(1-(cent[T-299:T, b]*cent[T-300:T-1, b]).sum(-1).mean()) for b in bs]
        H = []
        for b in bs:
            tr = np.zeros((k, k));d = dom[late, b]
            for a_, b_ in zip(d[:-1], d[1:]):tr[a_, b_] += 1
            pr_ = tr/np.maximum(tr.sum(1, keepdims=True), 1);occ = tr.sum(1)/tr.sum()
            H.append(float(-(occ[:, None]*pr_*np.log(np.where(pr_ > 0, pr_, 1))).sum()))
        m['transition_entropy'] = H
        m['clusters_visited'] = [len(np.unique(dom[late, b])) for b in bs]
        prof = np.stack([np.bincount(dom[late, b], minlength=k)/300 for b in bs])
        sims = prof@prof.T/np.outer(norm(prof, axis=1), norm(prof, axis=1));m['profile_agreement'] = float(sims[~np.eye(len(bs), dtype=bool)].mean())
        others = [b2 for b2 in range(B) if rows[b2]['cond'] == c]
        m['shared_pairwise'] = [np.mean([np.mean([len(np.intersect1d(idx[t, b], idx[t, o]))/K for o in others if o != b])
                                         for t in range(T-300, T, 20)]) for b in bs]
        # individual words: active in this mind but in fewer than a quarter of the other minds at the same step
        indiv_share, indiv_coh, indiv_pers = [], [], []
        for b in bs:
            sh, co, pe = [], [], []
            for t in range(T-300, T-10, 10):
                cnt = np.bincount(np.concatenate([idx[t, o] for o in others if o != b]), minlength=len(words))
                ind = [j for j in range(K) if cnt[idx[t, b, j]] < .25*(len(others)-1) and W[t, b, j] > 0]
                sh.append(W[t, b, ind].sum())
                if len(ind) >= 3:co.append(coherence(idx[t, b, ind], W[t, b, ind]/W[t, b, ind].sum())-null_mu)
                if ind:pe.append(len(np.intersect1d(idx[t, b, ind], idx[t+10, b]))/len(ind))
            indiv_share.append(np.mean(sh));indiv_coh.append(np.mean(co) if co else np.nan);indiv_pers.append(np.mean(pe) if pe else np.nan)
        m['indiv_share'], m['indiv_coherence'], m['indiv_persist10'] = indiv_share, indiv_coh, indiv_pers
        u = np.bincount(idx[late][:, bs].ravel(), weights=W[late][:, bs].ravel(), minlength=len(words))
        m['abstractness'] = [float((W[late, b]*s_abs[idx[late, b]]).sum(-1).mean()) for b in bs]
        m['mind_words'] = [float((W[late, b]*s_meta[idx[late, b]]).sum(-1).mean()) for b in bs]
        res['conds'][c] = dict(mean={kk:float(np.mean(v)) for kk, v in m.items() if isinstance(v, list)},
                               se={kk:float(np.std(v, ddof=1)/np.sqrt(len(v))) for kk, v in m.items() if isinstance(v, list)},
                               profile_agreement=m['profile_agreement'], per_seed={kk:[float(x) for x in v] for kk, v in m.items() if isinstance(v, list)},
                               top_words=[words[i] for i in np.argsort(-u)[:20]],
                               example=[[words[i] for i in idx[t, bs[0]][np.argsort(-coef[t, bs[0]])][:5]] for t in (100, 200, 300, 400, 500, 599)])
    # words over-represented in a condition against free (log-odds with a prior)
    base = np.bincount(idx[late][:, [row['free', s] for s in seeds]].ravel(), minlength=len(words))+1.
    for c in conds:
        u = np.bincount(idx[late][:, [row[c, s] for s in seeds]].ravel(), minlength=len(words))+1.
        lo = np.log(u/u.sum())-np.log(base/base.sum());lo[(u+base) < 30] = 0
        res['conds'][c]['over_vs_free'] = [words[i] for i in np.argsort(-lo)[:15]]
    res['clusters'] = [[words[i] for i in top_used[np.argsort(-(Xc@C0[j]))][:8]] for j in range(k)]
    json.dump(res, open(out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'model {M["args"]["model"]}, {len(seeds)} seeds, last 300 of {T} steps; null coherence (random 25-word sets by usage) {null_mu:.3f}')
    print(f"abstract axis check: held-out abstract words {check['abstract_test']:+.3f}, concrete words {check['concrete_test']:+.3f}\n")
    keys = ('coherence', 'overlap10', 'overlap100', 'distinct', 'sem_pr', 'sem_step', 'transition_entropy', 'clusters_visited',
            'shared_pairwise', 'indiv_share', 'indiv_coherence', 'indiv_persist10', 'abstractness', 'mind_words')
    print(f"{'condition':11s} " + ' '.join(f'{k_[:9]:>9s}' for k_ in keys) + '  agree')
    for c in conds:
        v = res['conds'][c]['mean'];print(f'{c:11s} ' + ' '.join(f'{v[k_]:9.3f}' for k_ in keys) + f"  {res['conds'][c]['profile_agreement']:.2f}")
    print('\npaired differences over seeds, ring_err minus control (mean ± se, seeds in favour)')
    for ctrl in ('vel', 'yoked_err', 'shuf_err', 'rand_err', 'free', 'content'):
        if ctrl not in res['conds']:continue
        line = []
        for k_ in ('sem_pr', 'overlap10', 'distinct', 'indiv_share', 'abstractness', 'mind_words'):
            d = np.array(res['conds']['ring_err']['per_seed'][k_])-np.array(res['conds'][ctrl]['per_seed'][k_]);d = d[np.isfinite(d)]
            line.append(f'{k_} {d.mean():+.3f}±{d.std(ddof=1)/np.sqrt(len(d)):.3f} ({(d > 0).sum()}/{len(d)})')
        print(f'  vs {ctrl:9s} ' + '  '.join(line))
    print('\nwords over-represented against free (late steps):')
    for c in conds:print(f'  {c:11s}', ', '.join(res['conds'][c]['over_vs_free'][:12]))
    print('\none mind per condition, top-5 words at steps 100..600:')
    for c in ('free', 'content', 'vel', 'ring_err', 'rand_err'):
        if c in res['conds']:print(f'  {c:9s}', ' | '.join(', '.join(x) for x in res['conds'][c]['example']))
    print('\nsemantic clusters (k-means on the embeddings of the 3000 most used words):')
    for j, cw in enumerate(res['clusters']):print(f'  {j:2d}', ', '.join(cw))


if __name__ == '__main__':
    main()
