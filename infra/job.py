"""Queue runner for an unattended Vast.ai instance.
Runs the commands listed in /workspace/queue.txt one after another (new lines may be appended while it runs), uploads
/workspace/v3 to the private HF dataset after every command, and exits when the queue has been empty for the grace
period (/workspace/grace_min, default 25 minutes). run_v3.sh then uploads once more and destroys the instance.
    python job.py            run the queue
    python job.py --upload   upload only"""
import os, sys, time, subprocess, pathlib, traceback

W = pathlib.Path('/workspace');Q = W/'queue.txt';DONE = W/'queue_done.txt';OUTD = pathlib.Path(os.environ.get('V5_OUT', '/workspace/v5'))
REPO = os.environ.get('V3_REPO', 'Pixedar/jspace-v3');PREFIX = os.environ.get('V3_PREFIX', 'v5')


def say(*a):
    print(time.strftime('%F %T'), '[job]', *a, flush=True)


def upload():
    from huggingface_hub import HfApi
    api = HfApi()
    for attempt in range(4):
        try:
            api.create_repo(REPO, repo_type='dataset', private=True, exist_ok=True)
            api.upload_folder(folder_path=str(OUTD), path_in_repo=PREFIX, repo_type='dataset', repo_id=REPO,
                              ignore_patterns=['*.tmp', '*.tmp.npz', '*.partial'],
                              commit_message=f'v5 sync {time.strftime("%F %T")}')
            say('uploaded', OUTD, '->', REPO, PREFIX);return True
        except Exception as e:
            say('upload failed', attempt, repr(e)[:300]);time.sleep(30*(attempt+1))
    return False


def pending():
    lines = [l.strip() for l in Q.read_text().splitlines()] if Q.exists() else []
    done = set(DONE.read_text().splitlines()) if DONE.exists() else set()
    return [l for l in lines if l and not l.startswith('#') and l not in done and 'FAILED '+l not in done]


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    while True:
        todo = pending()
        if not todo:
            grace = float((W/'grace_min').read_text()) if (W/'grace_min').exists() else 25.
            say(f'queue empty, waiting up to {grace:.0f} min for new work')
            t0 = time.time()
            while time.time()-t0 < grace*60 and not pending():time.sleep(20)
            if not pending():say('no new work; exiting');return
            continue
        cmd = todo[0];say('run:', cmd);t0 = time.time()
        rc = subprocess.call(cmd, shell=True, cwd=str(W/'code'))
        with open(DONE, 'a') as f:f.write((cmd if rc == 0 else 'FAILED '+cmd)+'\n')
        say(f'rc={rc} after {time.time()-t0:.0f}s');upload()


if __name__ == '__main__':
    if '--upload' in sys.argv:sys.exit(0 if upload() else 1)
    try:main()
    except Exception:traceback.print_exc();sys.exit(1)
