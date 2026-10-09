"""Run the strange-loop experiment up a ladder of model sizes, each with its own Neuronpedia J-lens.

Per model: validate the lens (validate_lens.py); sweep the recurrent gain on observers and pick the one whose
observers keep about as much memory as Qwen3-4B-Base did in run C (lag-1 similarity ~0.2, fluent); at that gain, test
the reflection layer and gain on 16 observers and keep the one whose self-model best picks out its own window; then the
main run (free / own individual self / foreign individual self / clamp, with the forward question) and its analysis.
Each step is skipped when its output exists, so the ladder resumes after a crash.

Dictionary: the lens is Neuronpedia's (Anthropic's reference code), the decomposition the paper's (non-negative
pursuit, K = 25), the J-lens vectors those of common English words. The paper's every-token dictionary was measured
first (fidelity_check.py, 16 observers): on Qwen3-1.7B it read the trajectory half as well (25 % vs 49 %, raw
residual 46 %), on Qwen3-4B 57 % vs 68 % (raw 47 %).

    python run_ladder.py --models qwen3-1.7b,qwen3-4b,qwen3-8b,qwen3-14b,qwen3-32b
"""
import argparse, json, os, re, subprocess, sys
import numpy as np

LENSES = {'qwen3-1.7b':('Qwen/Qwen3-1.7B', 'Qwen3-1.7B'), 'qwen3-4b':('Qwen/Qwen3-4B', 'Qwen3-4B'),
          'qwen3-8b':('Qwen/Qwen3-8B', 'Qwen3-8B'), 'qwen3-14b':('Qwen/Qwen3-14B', 'Qwen3-14B'),
          'qwen3-32b':('Qwen/Qwen3-32B', 'Qwen3-32B')}
N_LAYERS = {'qwen3-1.7b':28, 'qwen3-4b':36, 'qwen3-8b':36, 'qwen3-14b':40, 'qwen3-32b':64}
PY = sys.executable


def sh(cmd, log):
    print('>>', ' '.join(cmd), flush=True)
    with open(log, 'a') as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:raise SystemExit(f'failed ({r.returncode}): see {log}')


def lens_path(tag):
    from huggingface_hub import hf_hub_download
    name = LENSES[tag][1]
    return hf_hub_download('neuronpedia/jacobian-lens', f'{tag}/jlens/Salesforce-wikitext/{name}_jacobian_lens.pt')


def pick_gain(sweep_dir, target=.2):
    """Gain: the observers' lag-1 similarity closest to `target` among fluent settings (run C of Qwen3-4B-Base sat at
    0.19-0.23). Reflection: the layer and gain whose self-model best picks out its own window among all rows."""
    z = np.load(f'{sweep_dir}/traj.npz');m = json.load(open(f'{sweep_dir}/meta.json', encoding='utf-8'))
    rows, a = m['rows'], m['args'];K = a['K'];j = z['j'].astype(np.float32);N, B = j.shape[:2]
    unit = lambda x:x/np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
    ju = unit(j);zz = np.zeros_like(j);s_ = np.zeros_like(j[0])
    for t in range(N):s_ = .6*s_+.4*j[t];zz[t] = s_
    wordy = lambda s:np.mean([bool(re.fullmatch(r"[A-Za-z',.!?;:-]+", w)) for w in s.split()] or [0])
    gains = {}
    for g in sorted({r['g'] for r in rows}):
        bs = [b for b, r in enumerate(rows) if r['g'] == g]
        lag1 = float(np.mean([(ju[10:-1, b]*ju[11:, b]).sum(-1).mean() for b in bs]))
        wd = float(np.mean([wordy(m['texts'][t][b]) for t in range(N) for b in bs]))
        d2 = float(np.mean([len(set(zip(w[:-1], w[1:])))/max(1, len(w)-1) for t in range(N) for b in bs for w in [m['texts'][t][b].split()]]))
        gains[g] = dict(lag1=lag1, wordy=wd, distinct2=d2)
    ok = [g for g, v in gains.items() if v['wordy'] >= .9 and v['distinct2'] >= .85] or list(gains)
    return dict(g=min(ok, key=lambda g:abs(gains[g]['lag1']-target)), gains=gains)


def pick_reflection(sweep_dir):
    """All rows share one gain here, so a self-model's own window competes with 15 others like it."""
    z = np.load(f'{sweep_dir}/traj.npz');m = json.load(open(f'{sweep_dir}/meta.json', encoding='utf-8'))
    K = m['args']['K'];j = z['j'].astype(np.float32);N, B = j.shape[:2]
    unit = lambda x:x/np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
    zz = np.zeros_like(j);s_ = np.zeros_like(j[0])
    for t in range(N):s_ = .6*s_+.4*j[t];zz[t] = s_
    NR = z['m'].shape[0];win = unit(np.stack([zz[r*K:(r+1)*K].mean(0) for r in range(NR)]))
    fid = {}
    for name in [f for f in z.files if f.startswith('fid_')]:
        M = unit(z[name].astype(np.float32));hit = []
        for r in range(1, NR):
            S = M[r]@win[r].T;hit += list(S.argmax(1) == np.arange(B))
        fid[name] = float(np.mean(hit))
    best = max(fid, key=fid.get);L_in, g_r = best[4:].split('_')
    return dict(L_in=int(L_in), g_r=float(g_r), fidelity=fid, chance=1/B)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', default='qwen3-1.7b,qwen3-4b,qwen3-8b,qwen3-14b,qwen3-32b')
    p.add_argument('--seeds', type=int, default=12);p.add_argument('--n_iter', type=int, default=300)
    p.add_argument('--t_int', type=int, default=120);p.add_argument('--k', type=int, default=25)
    p.add_argument('--root', default='runs/ladder2');p.add_argument('--vocab', default='english')
    p.add_argument('--atoms', default='centred')
    p.add_argument('--depth', type=float, default=.5, help='read layer as a fraction of depth (Qwen3 lens check: concepts from ~0.6)')
    a = p.parse_args()
    for tag in a.models.split(','):
        model, _ = LENSES[tag];out = f'{a.root}/{tag}';os.makedirs(out, exist_ok=True);log = f'{out}/log.txt'
        lens = lens_path(tag);n = N_LAYERS[tag];L = int(round(a.depth*n))
        common = ['--model', model, '--jlens', lens, '--vocab', a.vocab, '--atoms', a.atoms, '--k', str(a.k), '--layer', str(L)]
        if not os.path.exists(f'{out}/val.json'):
            sh([PY, 'validate_lens.py', '--model', model, '--jlens', lens, '--out', f'{out}/val.json'], log)
        if not os.path.exists(f'{out}/gsweep/meta.json'):
            sh([PY, 'strange_loop.py', *common, '--grid', 'gsweep', '--gs', '.03,.05,.08,.12,.2', '--seeds', '6', '--n_iter', '60',
                '--L_in', str(L-1), '--out', f'{out}/gsweep'], log)
        if not os.path.exists(f'{out}/gain.json'):json.dump(pick_gain(f'{out}/gsweep'), open(f'{out}/gain.json', 'w'), indent=1)
        g = json.load(open(f'{out}/gain.json'))['g']
        if not os.path.exists(f'{out}/fsweep/meta.json'):
            offs = sorted({1, 2, 3, max(1, int(round(.1*n))), max(1, int(round(.15*n)))})
            fid = ','.join(f'{L-o}:{gr}' for o in offs for gr in (.5, 1.))
            sh([PY, 'strange_loop.py', *common, '--grid', 'gsweep', '--gs', str(g), '--seeds', '16', '--n_iter', '60',
                '--L_in', str(L-1), '--fid', fid, '--out', f'{out}/fsweep'], log)
        if not os.path.exists(f'{out}/params.json'):
            json.dump(dict(g=g, **pick_reflection(f'{out}/fsweep')), open(f'{out}/params.json', 'w'), indent=1)
        P = json.load(open(f'{out}/params.json'));print(tag, 'picked', {k:P[k] for k in ('g', 'L_in', 'g_r')}, flush=True)
        if not os.path.exists(f'{out}/main/meta.json'):
            sh([PY, 'strange_loop.py', *common, '--grid', 'main', '--conds', 'free,loopc,yokedc,clampc', '--seeds', str(a.seeds),
                '--n_iter', str(a.n_iter), '--t_int', str(a.t_int), '--clamp_len', '50', '--g', str(P['g']),
                '--L_in', str(P['L_in']), '--g_r', str(P['g_r']), '--out', f'{out}/main'], log)
        if not os.path.exists(f'{out}/main/results.json'):
            sh([PY, 'analyze.py', f'{out}/main'], f'{out}/main/summary.txt')
        print(tag, 'done', flush=True)


if __name__ == '__main__':
    main()
