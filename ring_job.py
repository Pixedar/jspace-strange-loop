"""Run the ring unattended on one rented GPU, model after model.

Per model: a pilot sweep of the free substrate (gain x noise, free minds only), a fixed rule that picks the regime
from it, the main run (every condition, paired seeds, split into chunks of seeds that fit the card), its analysis, and
an upload of the model's folder to a private Hugging Face dataset while the next model runs. Each step is skipped when
its output exists, so a restart resumes. Last, an exploratory long run on Qwen3-4B (T = 2000).

Pick rule, fixed before any real run: among the (g, sigma) whose free minds are neither collapsed (late share of steps
with step cosine > 0.98 at most 0.25) nor frozen in place (late lag-1 cosine at most 0.9), the one with the longest
memory (lag-100 cosine); if none qualifies, the least collapsed. The ring channel then gets s = g.

    python ring_job.py --models qwen3-4b,qwen3-8b,qwen3-14b,qwen3-1.7b --root /workspace/runs/ring --hf Pixedar/jspace-ring
    python ring_job.py ... --upload_only          # push everything under --root (and the job log) and exit
    python ring_job.py ... --gains .5,2,4         # stage 2, exploratory (added after the first results): every
                                                  # condition again at multiples of the picked gain (ring_gains.py)
"""
import argparse, json, math, os, queue, shutil, subprocess, sys, threading, time
import numpy as np

MODELS = {'qwen3-1.7b':('Qwen/Qwen3-1.7B', 'Qwen3-1.7B', 1.72e9), 'qwen3-4b':('Qwen/Qwen3-4B', 'Qwen3-4B', 4.02e9),
          'qwen3-8b':('Qwen/Qwen3-8B', 'Qwen3-8B', 8.19e9), 'qwen3-14b':('Qwen/Qwen3-14B', 'Qwen3-14B', 14.77e9),
          'tiny':('Qwen/Qwen2.5-0.5B-Instruct', None, .49e9)}             # tiny: identity lens, for testing the job
PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
N_CONDS = 10


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def sh(cmd, logf):
    log('>>', ' '.join(cmd))
    with open(logf, 'a') as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
    if r.returncode:raise RuntimeError(f'failed ({r.returncode}): see {logf}')


def fetch(tag):
    """Model weights and the Neuronpedia lens (the prefetch thread usually has them already)."""
    from huggingface_hub import hf_hub_download, snapshot_download
    repo, name, _ = MODELS[tag]
    snapshot_download(repo, allow_patterns=['*.json', '*.safetensors', '*.txt'])
    if name is None:return 'identity'
    return hf_hub_download('neuronpedia/jacobian-lens', f'{tag}/jlens/Salesforce-wikitext/{name}_jacobian_lens.pt')


def seeds_per_chunk(tag, T, rows_per_seed, mem_gb, cap):
    """How many seeds fit at once: the KV cache of every mind over the whole run dominates, then the stored surprise
    history and the self-model with its Adam state; 14 % of the card and 3 GB more are left as headroom."""
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(MODELS[tag][0])
    hd = getattr(c, 'head_dim', None) or c.hidden_size//c.num_attention_heads
    kv = 2*c.num_key_value_heads*hd*2*c.num_hidden_layers
    per_mind = (100+24+T+40)*kv+T*c.hidden_size*2+8*c.hidden_size*32*4+2*32*c.hidden_size*4
    free = .86*mem_gb*2**30-2*MODELS[tag][2]-3*2**30
    return max(1, min(cap, int(free//(per_mind*rows_per_seed))))


def run_seeds(base, S, per, cmd, lg, least=1):
    """Seeds 0..S-1 in chunks of at most `per` (folders s<first>_<count>, skipped when finished). A failed chunk, in
    practice an out-of-memory, is retried at half the size; the yoked conditions need at least two seeds per chunk."""
    os.makedirs(base, exist_ok=True);per = max(least, min(per, S))
    while True:
        done = set()
        for x in os.listdir(base):
            if x.startswith('s') and os.path.exists(f'{base}/{x}/meta.json'):
                s0, n = map(int, x[1:].split('_'));done |= set(range(s0, s0+n))
        todo = [s for s in range(S) if s not in done]
        if not todo:return
        want = math.ceil(len(todo)/math.ceil(len(todo)/per))                  # balanced: 24 at 11 -> 8, 8, 8
        s0, n = todo[0], 1
        while n < want and s0+n in todo:n += 1
        if 0 < len(todo)-n < least and todo == list(range(s0, s0+len(todo))):n = len(todo)    # no lone seed left over
        d = f'{base}/s{s0}_{n}'
        try:sh(cmd(s0, n, d), lg)
        except RuntimeError:
            shutil.rmtree(d, ignore_errors=True)
            if n <= least:raise
            per = max(least, n//2);log(f'chunk of {n} seeds failed; retrying with {per}')


def pr(V):
    V = V-V.mean(0);ev = np.clip(np.linalg.eigvalsh(V@V.T), 0, None)
    return float(ev.sum()**2/max((ev**2).sum(), 1e-12))


def pick(sweep_dirs):
    rows, sc, U = [], [], []
    for d in sweep_dirs:
        m = json.load(open(f'{d}/meta.json', encoding='utf-8'))
        with np.load(f'{d}/ring.npz') as z:
            C = z['C'].astype(np.float32);sc.append(z['step_cos'])
        U.append(C/np.maximum(np.linalg.norm(C, axis=-1, keepdims=True), 1e-8));rows += m['rows']
    sc = np.concatenate(sc, 1);U = np.concatenate(U, 1);T = sc.shape[0]
    LT, LG, W = min(200, T//3), min(100, T//4), min(100, max(10, T//6))
    grid = {}
    for g, sg in sorted({(r['g'], r['sigma']) for r in rows}):
        bs = [i for i, r in enumerate(rows) if (r['g'], r['sigma']) == (g, sg)]
        grid[f'g{g}_sigma{sg}'] = dict(
            g=g, sigma=sg, stuck=float((sc[T-LT:, bs] > .98).mean()), lag1=float(sc[T-LT:, bs].mean()),
            lag100=float(np.mean([(U[T-LG:, b]*U[T-2*LG:T-LG, b]).sum(-1).mean() for b in bs])),
            explore=float(np.mean([pr(U[w:w+W, b]) for b in bs for w in range(T-LT, T-W+1, max(1, W//2))])))
    ok = [k for k, v in grid.items() if v['stuck'] <= .25 and v['lag1'] <= .9]
    best = max(ok, key=lambda k:grid[k]['lag100']) if ok else min(grid, key=lambda k:(grid[k]['stuck'], grid[k]['lag1']))
    return dict(g=grid[best]['g'], sigma=grid[best]['sigma'], picked=best, qualified=bool(ok), grid=grid)


class Uploader(threading.Thread):
    """Pushes finished folders to the dataset in the background; failures are remembered, not fatal."""

    def __init__(self, repo):
        super().__init__(daemon=True);self.repo = repo;self.q = queue.Queue();self.failed = []
        if repo:
            from huggingface_hub import create_repo
            create_repo(repo, repo_type='dataset', private=True, exist_ok=True)

    def run(self):
        while True:
            item = self.q.get()
            if item is None:break
            folder, path = item
            try:
                from huggingface_hub import HfApi
                HfApi().upload_folder(repo_id=self.repo, repo_type='dataset', folder_path=folder, path_in_repo=path,
                                      commit_message=f'ring: {path}');log('uploaded', path)
            except Exception as e:
                self.failed.append(path);log('UPLOAD FAILED', path, repr(e)[:300])

    def put(self, folder, path):
        if self.repo:self.q.put((folder, path))


def run_model(tag, a, up, mem_gb, long=False):
    model = MODELS[tag][0];name = f'{tag}-long' if long else tag
    out = f'{a.root}/{name}';os.makedirs(out, exist_ok=True);lg = f'{out}/log.txt'
    lens = fetch(tag)
    common = ['--model', model, '--jlens', lens, '--vocab', a.vocab, '--k', str(a.k)]
    # 1. pilot: the free substrate over gain x noise
    if long:
        P = json.load(open(f'{a.root}/{tag}/params.json'))
    else:
        gs, sgs = a.gs.split(','), a.sigmas.split(',')
        per = seeds_per_chunk(tag, a.T, len(gs)*len(sgs), mem_gb, a.max_chunk)
        run_seeds(f'{out}/sweep', a.sweep_seeds, per, lambda s0, n, d:[
            PY, 'ring.py', *common, '--grid', 'sweep', '--gs', a.gs, '--sigmas', a.sigmas, '--seeds', str(n), '--seed0', str(s0),
            '--rng', str(4321+s0), '--T', str(a.T), '--t_freeze', str(a.T+1), '--glimpse', '0', '--out', d], lg)
        if not os.path.exists(f'{out}/params.json'):
            sw = [f'{out}/sweep/{x}' for x in sorted(os.listdir(f'{out}/sweep')) if os.path.exists(f'{out}/sweep/{x}/meta.json')]
            json.dump(pick(sw), open(f'{out}/params.json', 'w'), indent=1)
        P = json.load(open(f'{out}/params.json'))
    log(tag, 'regime', P['picked'], 'qualified' if P['qualified'] else 'NONE QUALIFIED (least collapsed)',
        {k:round(v, 3) for k, v in P['grid'][P['picked']].items()})
    # 2. main run: every condition, paired seeds, chunks that fit
    T, TF, S = (a.long_T, a.long_T//2, a.long_seeds) if long else (a.T, a.t_freeze, a.seeds)
    per = seeds_per_chunk(tag, T, N_CONDS, mem_gb, a.max_chunk)
    run_seeds(f'{out}/main', S, per, lambda s0, n, d:[
        PY, 'ring.py', *common, '--grid', 'main', '--seeds', str(n), '--seed0', str(s0), '--rng', str(1234+s0),
        '--T', str(T), '--t_freeze', str(TF), '--g', str(P['g']), '--sigma', str(P['sigma']), '--out', d], lg, least=2)
    # 3. analysis, then hand the folder to the uploader
    if not os.path.exists(f'{out}/main/results.json'):
        sh([PY, 'ring_analyze.py', f'{out}/main'], f'{out}/main/summary.txt')
    log(tag, 'done' + (' (long)' if long else ''));up.put(out, name)


def run_gains(tag, a, up, mem_gb):
    """Stage 2: the main grid at multiples of the picked gain (s = g as in the main run), then ring_gains.py."""
    out = f'{a.root}/{tag}';P = json.load(open(f'{out}/params.json'));lg = f'{out}/log.txt'
    common = ['--model', MODELS[tag][0], '--jlens', fetch(tag), '--vocab', a.vocab, '--k', str(a.k)]
    per = seeds_per_chunk(tag, a.T, N_CONDS, mem_gb, a.max_chunk)
    for mult in [float(x) for x in a.gains.split(',')]:
        g = round(P['g']*mult, 4)
        run_seeds(f'{out}/gains/g{g}', a.gain_seeds, per, lambda s0, n, d:[
            PY, 'ring.py', *common, '--grid', 'main', '--seeds', str(n), '--seed0', str(s0), '--rng', str(2234+s0),
            '--T', str(a.T), '--t_freeze', str(a.t_freeze), '--g', str(g), '--sigma', str(P['sigma']), '--out', d], lg, least=2)
    if not os.path.exists(f'{out}/gains/gains.json'):
        sh([PY, 'ring_gains.py', out], f'{out}/gains/summary.txt')
    log(tag, 'gain ladder done');up.put(f'{out}/gains', f'{tag}/gains')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', default='qwen3-4b,qwen3-8b,qwen3-14b,qwen3-1.7b');p.add_argument('--root', default='runs/ring')
    p.add_argument('--hf', default='');p.add_argument('--vocab', default='english');p.add_argument('--k', type=int, default=25)
    p.add_argument('--T', type=int, default=600);p.add_argument('--t_freeze', type=int, default=300)
    p.add_argument('--seeds', type=int, default=24);p.add_argument('--sweep_seeds', type=int, default=6)
    p.add_argument('--gs', default='.05,.1,.2,.3,.5,.8');p.add_argument('--sigmas', default='.05,.15')
    p.add_argument('--long', default='qwen3-4b');p.add_argument('--long_T', type=int, default=2000)
    p.add_argument('--long_seeds', type=int, default=8);p.add_argument('--max_chunk', type=int, default=64)
    p.add_argument('--mem_gb', type=float, default=0, help='card memory (default: measured)')
    p.add_argument('--upload_only', action='store_true')
    p.add_argument('--gains', default='', help='stage 2: multiples of the picked gain');p.add_argument('--gain_seeds', type=int, default=16)
    a = p.parse_args();a.root = os.path.abspath(a.root);os.makedirs(a.root, exist_ok=True)
    if a.upload_only:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(repo_id=a.hf, repo_type='dataset', folder_path=a.root, path_in_repo='.', commit_message='ring: all')
        for f in ('/workspace/job.log', '/workspace/job2.log', '/workspace/setup.log'):
            if os.path.exists(f):api.upload_file(path_or_fileobj=f, path_in_repo=f'logs/{os.path.basename(f)}', repo_id=a.hf,
                                                 repo_type='dataset')
        log('final upload done');return
    mem_gb = a.mem_gb or float(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                                                        text=True).split()[0])/1024     # no CUDA context in this process
    tags = a.models.split(',')
    log(f'ring job: {tags} | card {mem_gb:.1f} GB | root {a.root} | upload to {a.hf or "(none)"}')
    threading.Thread(target=lambda:[fetch(t) for t in tags], daemon=True).start()     # downloads overlap the compute
    up = Uploader(a.hf);up.start();errors = []
    if a.gains:
        for tag in tags:
            try:run_gains(tag, a, up, mem_gb)
            except Exception as e:errors.append(tag);log('GAINS FAILED', tag, repr(e)[:400])
        up.q.put(None);up.join()
        json.dump(dict(errors=errors, upload_failed=up.failed, finished=time.time()), open(f'{a.root}/DONE2.json', 'w'))
        log('stage 2 done', 'errors', errors, 'upload failures', up.failed);return
    for tag in tags:
        try:run_model(tag, a, up, mem_gb)
        except Exception as e:errors.append(tag);log('MODEL FAILED', tag, repr(e)[:400])
    if a.long and a.long in tags and a.long not in errors and a.long_T:
        try:run_model(a.long, a, up, mem_gb, long=True)
        except Exception as e:errors.append(a.long+'-long');log('MODEL FAILED', a.long, 'long', repr(e)[:400])
    up.q.put(None);up.join()
    json.dump(dict(errors=errors, upload_failed=up.failed, finished=time.time()), open(f'{a.root}/DONE.json', 'w'))
    log('all done', 'errors', errors, 'upload failures', up.failed)


if __name__ == '__main__':
    main()
