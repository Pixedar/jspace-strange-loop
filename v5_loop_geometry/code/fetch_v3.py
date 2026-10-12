"""Download the v3 E2 arrays needed by geom_v3.py from the private HF dataset onto the analysis instance.
    python fetch_v3.py /workspace/data"""
import sys, time
from huggingface_hub import snapshot_download

d = sys.argv[1]
pats = ['v3/e2/qwen3-8b/main/*.json', 'v3/e2/qwen3-8b/main/calib/*', 'v3/e2/qwen3-8b/main/A1/*', 'v3/e2/qwen3-8b/main/A2/*',
        'v3/e2/qwen3-8b/main/BC/*', 'v3/e2/qwen3-8b/main/BCs/*', 'v3/e2/qwen3-8b/main/pert/*',
        'v3/e2/qwen3-14b/rep/*.json', 'v3/e2/qwen3-14b/rep/calib/*', 'v3/e2/qwen3-14b/rep/A1/*', 'v3/e2/qwen3-14b/rep/A2/*',
        'v3/e2/qwen3-14b/rep/BC1/*', 'v3/e2/qwen3-14b/rep/pert/*']
for attempt in range(5):
    try:
        t0 = time.time();snapshot_download('Pixedar/jspace-v3', repo_type='dataset', local_dir=d, allow_patterns=pats, max_workers=16)
        print(f'downloaded in {time.time()-t0:.0f}s', flush=True);break
    except Exception as e:
        print('retry', attempt, repr(e)[:300], flush=True);time.sleep(30)
else:
    sys.exit(1)
