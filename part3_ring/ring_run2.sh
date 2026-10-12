#!/bin/bash
# Stage 2 (the exploratory gain ladder), chained after ring_run.sh, which must have been told to leave the instance up
# (/workspace/HOLD). Waits until stage 1 has finished and uploaded, runs ring_job.py --gains ..., pushes everything to
# the dataset, then destroys the instance. It stops the instance instead if the upload failed, and leaves it
# running if /workspace/HOLD2 exists.
#   bash ring_run2.sh <ring_job.py args...>
cd "$(dirname "$0")"
PY=${PY:-/opt/conda/bin/python}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
say() { echo "$(date '+%F %T') [run2] $*" >> /workspace/job2.log; }
say "waiting for stage 1"
until grep -q "\[run\] HOLD set" /workspace/job.log 2>/dev/null; do
  # stage 1's wrapper died before its end: go on once its DONE.json is 45 min old
  [ -f /workspace/runs/ring/DONE.json ] && [ $(( $(date +%s) - $(stat -c %Y /workspace/runs/ring/DONE.json) )) -gt 2700 ] && break
  sleep 20
done
say "stage 2 start"
setsid $PY ring_job.py "$@" >> /workspace/job2.log 2>&1 &
JOB=$!
while kill -0 $JOB 2>/dev/null; do
  if [ $(date +%s) -gt $(cat /workspace/deadline) ]; then
    say "deadline passed, stopping the job"; kill -TERM -- -$JOB 2>/dev/null; sleep 30; kill -KILL -- -$JOB 2>/dev/null
  fi
  sleep 20
done
wait $JOB; say "stage 2 ended (rc $?)"
UP=1
for i in 1 2 3; do timeout 2400 $PY ring_job.py "$@" --upload_only >> /workspace/job2.log 2>&1 && { UP=0; break; }; sleep 60; done
if [ -f /workspace/HOLD2 ]; then say "HOLD2 set: leaving the instance running (watchdog stops it an hour after the deadline)"
elif [ $UP = 0 ]; then say "uploaded, destroying the instance"; bash vast_self.sh destroy >> /workspace/job2.log 2>&1
else say "upload failed, stopping the instance (disk kept)"; bash vast_self.sh stop >> /workspace/job2.log 2>&1; fi
