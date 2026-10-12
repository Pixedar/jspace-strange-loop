"""Functional atlas (required deliverable of the original proposal): concrete prompts, the J-space words at the relevant
positions, the causal patch outcomes, the challenge responses, and the words a loop runs through over time.
Writes <root>/atlas.json.   python atlas.py /workspace/v3 --model qwen3-8b"""
import sys, json, pathlib, argparse
import numpy as np


def words_of(words, idx, coef, n=10):
    o = np.argsort(-np.asarray(coef, float))[:n]
    return [(words[int(idx[i])], round(float(coef[i]), 1)) for i in o if coef[i] > 0]


def main():
    p = argparse.ArgumentParser();p.add_argument('root');p.add_argument('--model', default='qwen3-8b')
    a = p.parse_args();root = pathlib.Path(a.root);out = {}
    from an_e1 import Dict
    from common import MODELS
    import torch
    layers = MODELS[a.model]['layers'];D = Dict(a.model, layers, 'cuda' if torch.cuda.is_available() else 'cpu');W = D.words
    from tasks import build_all, user_text
    e1 = root/'e1'/a.model/'main'
    if (e1/'h1'/'states.npz').exists():
        data = build_all('main');items = [it for it in data['F1']]
        st = np.load(e1/'h1'/'states.npz');meta = json.loads((e1/'h1'/'meta.json').read_text())
        pz = np.load(e1/'h1'/'patch.npz');tasks = pz['tasks'];conds = list(pz['conds'])
        idx_of = {(m_, t_, b_):i for i, (m_, t_, b_) in enumerate(zip(meta['sid'], meta['template'], meta['binding']))}
        ex = [];seen = set()
        for ti, (ta, sb) in enumerate(tasks):
            sub = meta['sub'][ta]
            if sub in seen:continue
            seen.add(sub);it = next(x for x in items if x['sid'] == meta['sid'][ta] and x['template'] == meta['template'][ta] and x['binding'] == meta['binding'][ta])
            e = dict(relation=sub, scenario=it['scenario'], swapped_scenario=next(x for x in items if x['sid'] == meta['sid'][sb] and x['template'] == meta['template'][sb] and x['binding'] == meta['binding'][sb])['scenario'],
                     query=it['queries'][0]['q'], answer=it['queries'][0]['cands'][0], in_context_logodds=float(st['lo'][ta, 0]),
                     J_words_scenario_end={l:words_of(W, st[f'{l}_idx'][ta, :25], st[f'{l}_coef25'][ta]) for l in layers[2:]},
                     decision_patch_logodds={l:{c:float(pz['SP'][ti, 0, li, k, 1]) for k, c in enumerate(conds)} for li, l in enumerate(layers)},
                     context_patch_logodds={l:{c:float(pz['CP'][ti, 0, li, k]) for k, c in enumerate(conds)} for li, l in enumerate(layers)})
            ex.append(e)
            if len(ex) >= 4:break
        out['H1_binding'] = ex
    if (e1/'h2'/'F2'/'states.npz').exists():
        z = np.load(e1/'h2'/'F2'/'states.npz');m = json.loads((e1/'h2'/'F2'/'meta.json').read_text());S = m['states']
        data = build_all('main');f2 = {(it['sid'], it['evidence'], it['edir']):it for it in data['F2'] if it['challenge'] == 'none'}
        sid0 = next(s['sid'] for s in S if s['split'] == 'test');ex = []
        for i, s in enumerate(S):
            if s['sid'] != sid0:continue
            it = f2[(s['sid'], s['evidence'], s['edir'])]
            ex.append(dict(evidence=s['evidence'], toward=s['edir'], text=user_text(it)[0], logodds_now=float(z['lo0'][i]),
                           J_words_decision_L29={l:words_of(W, z[f'{l}_dec_idx'][i, :25], z[f'{l}_dec_coef25'][i]) for l in layers[3:]},
                           response_to_challenges={c:float(z['LC'][i, k]-z['lo0'][i]) for k, c in enumerate(m['chal_names'])},
                           internal={f'L{l}_{c}':float(z['LI'][i, li, k+1]-z['LI'][i, li, 0]) for li, l in enumerate(m['int_layers']) for k, c in enumerate(m['icond'][1:])}))
        out['H2_dispositions_one_scenario'] = ex
    e2 = root/'e2'/a.model/'main'
    for stage, conds_want in (('A2', ('J', 'dense', 'nofb')), ('X', ('inc', 'out', 'lens', 'J'))):
        fs = sorted((e2/stage).glob(f'{stage}_mem_*.npz'))
        if not fs:continue
        z = np.load(fs[0]);m = json.loads(fs[0].with_suffix('.json').read_text());ctx = json.loads((e2/'contexts.json').read_text())
        ex = [];famseen = set()
        for i, r in enumerate(m['rows']):
            u = m['units'][r['unit']]
            if r['cond'] not in conds_want or u['tag'] != 'A' or (u['fam'], r['cond'], r['g']) in famseen:continue
            famseen.add((u['fam'], r['cond'], r['g']))
            e = dict(family=u['fam'], condition=r['cond'], gain=r['g'], init_context=ctx[u['init']]['text'], running_context=ctx[u['ctx']]['text'],
                     candidates=ctx[u['ctx']]['cands'], logodds_by_step={t:float(z['lo'][t, i]) for t in (0, 1, 2, 4, 16, 64) if t < z['lo'].shape[0]},
                     workspace_words_by_step={t:words_of(W, z['idx'][t, i], z['coef'][t, i], 8) for t in (0, 1, 2, 4, 16, 64) if t < z['idx'].shape[0]})
            if 'widx' in z.files and r['cond'] in ('inc', 'out', 'lens'):
                e['broadcast_words_by_step'] = {t:[W[int(j)] for j in z['widx'][t, i][:8]] for t in (0, 1, 4, 16, 63) if t < z['widx'].shape[0]}
            ex.append(e)
        out[f'loop_{stage}'] = ex
    fs = sorted((e2/'BC').glob('BC_std_*.npz'))
    if fs:
        z = np.load(fs[0]);m = json.loads(fs[0].with_suffix('.json').read_text());ctx = json.loads((e2/'contexts.json').read_text());ex = []
        names = ['none']+[f'amplify#{k}' for k in range(1, 5)]+[f'suppress#{k}' for k in range(1, 5)]
        for c in ('B', 'C1', 'C2', 'R'):
            i = next(j for j, r in enumerate(m['rows']) if r['cond'] == c and r['unit'] == 0 and r.get('seed', 0) == 0)
            ex.append(dict(condition=c, gain=m['rows'][i]['g'], context=ctx[m['units'][0]['ctx']]['text'],
                           actions=[names[int(x)] for x in z['act'][:24, i]], self_model_error=[round(float(x), 3) for x in z['perr'][:64:8, i]],
                           workspace_words_by_step={t:words_of(W, z['idx'][t, i], z['coef'][t, i], 6) for t in (0, 8, 32, 64)}))
        out['self_model_regimes_one_context'] = ex
    (root/'atlas.json').write_text(json.dumps(out, indent=1, default=float));print('atlas written', list(out))


if __name__ == '__main__':
    main()
