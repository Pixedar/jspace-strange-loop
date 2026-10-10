"""Run the workspace tests unattended on one rented GPU, model after model.

Per model: ws_ignition.py, ws_trace.py, ws_temperature.py and ws_selftalk.py. Each writes <out>/<test>/<test>.json and
is skipped when that file exists, so a restart resumes. Each finished folder goes to a private Hugging Face dataset
in the background while the next test runs. Weights and lenses are prefetched in a thread. Batch sizes for 14B are
halved to fit a 48 GB card.

    python ws_job.py --models qwen3-4b,qwen3-8b,qwen3-14b,qwen3-1.7b --root /workspace/runs/ws --hf Pixedar/jspace-workspace
    python ws_job.py ... --upload_only
"""
import argparse, json, os, queue, subprocess, sys, threading, time

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = ('ignition', 'trace', 'temperature', 'selftalk', 'selfref')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def cmd(test, tag, out, vocab):
    big = tag == 'qwen3-14b'
    base = [PY, f'ws_{test}.py', '--model', tag, '--out', out, '--vocab', vocab]
    return base+{'ignition':['--batch', '256' if big else '512'], 'trace':['--bs', '16' if big else '32'],
                 'temperature':['--seeds', '6' if big else '12'], 'selftalk':['--seeds', '3' if big else '6'],
                 'selfref':['--bs', '8' if big else '16']}[test]


def fetch(tag):
    from huggingface_hub import hf_hub_download, snapshot_download
    from ws_common import MODELS
    repo, name = MODELS[tag];snapshot_download(repo, allow_patterns=['*.json', '*.safetensors', '*.txt'])
    if name:hf_hub_download('neuronpedia/jacobian-lens', f'{tag}/jlens/Salesforce-wikitext/{name}_jacobian_lens.pt')


class Uploader(threading.Thread):
    def __init__(self, repo):
        super().__init__(daemon=True);self.repo = repo;self.q = queue.Queue();self.failed = []
        if repo:
            from huggingface_hub import create_repo
            create_repo(repo, repo_type='dataset', private=True, exist_ok=True)

    def run(self):
        while True:
            item = self.q.get()
            if item is None:break
            try:
                from huggingface_hub import HfApi
                HfApi().upload_folder(repo_id=self.repo, repo_type='dataset', folder_path=item[0], path_in_repo=item[1],
                                      commit_message=f'workspace tests: {item[1]}');log('uploaded', item[1])
            except Exception as e:
                self.failed.append(item[1]);log('UPLOAD FAILED', item[1], repr(e)[:300])

    def put(self, folder, path):
        if self.repo:self.q.put((folder, path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', default='qwen3-4b,qwen3-8b,qwen3-14b,qwen3-1.7b');p.add_argument('--tests', default=','.join(TESTS[:4]))
    p.add_argument('--root', default='runs/ws');p.add_argument('--hf', default='');p.add_argument('--vocab', default='english')
    p.add_argument('--upload_only', action='store_true')
    a = p.parse_args();a.root = os.path.abspath(a.root);os.makedirs(a.root, exist_ok=True)
    if a.upload_only:
        from huggingface_hub import HfApi
        api = HfApi();api.upload_folder(repo_id=a.hf, repo_type='dataset', folder_path=a.root, path_in_repo='.', commit_message='workspace tests: all')
        for f in ('/workspace/job.log', '/workspace/setup.log'):
            if os.path.exists(f):api.upload_file(path_or_fileobj=f, path_in_repo=f'logs/{os.path.basename(f)}', repo_id=a.hf, repo_type='dataset')
        log('final upload done');return
    tags = a.models.split(',');tests = a.tests.split(',')
    log(f'workspace tests: {tags} x {tests} | root {a.root} | upload to {a.hf or "(none)"}')
    threading.Thread(target=lambda:[fetch(t) for t in tags], daemon=True).start()
    up = Uploader(a.hf);up.start();errors = []
    for tag in tags:
        for test in tests:
            out = f'{a.root}/{tag}/{test}'
            if os.path.exists(f'{out}/{test}.json'):log(tag, test, 'already done');continue
            os.makedirs(out, exist_ok=True)
            try:
                fetch(tag);log('>>', tag, test)
                with open(f'{out}/log.txt', 'a') as f:
                    r = subprocess.run(cmd(test, tag, out, a.vocab), stdout=f, stderr=subprocess.STDOUT, cwd=HERE)
                if r.returncode:raise RuntimeError(f'exit {r.returncode}, see {out}/log.txt')
                log(tag, test, 'done');up.put(out, f'{tag}/{test}')
            except Exception as e:
                errors.append(f'{tag}/{test}');log('FAILED', tag, test, repr(e)[:300])
    up.q.put(None);up.join()
    json.dump(dict(errors=errors, upload_failed=up.failed, finished=time.time()), open(f'{a.root}/DONE.json', 'w'))
    log('all done', 'errors', errors, 'upload failures', up.failed)


if __name__ == '__main__':
    main()
