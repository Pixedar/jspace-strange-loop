#!/bin/bash
# Run a job script unattended on a Vast.ai instance (the generic form of ring_run.sh). Runs the job until it ends or
# the deadline passes, pushes everything to the Hugging Face dataset (job.py ... --upload_only), then destroys the
# instance. It stops the instance instead if the upload failed, and leaves it running if /workspace/HOLD exists; the
# watchdog then stops it an hour after the deadline.
#   bash vast_run.sh <hours> <job.py> <job args...>
#   extend:  echo $(( $(date +%s) + 3600 )) > /workspace/deadline
cd "$(dirname "$0")"
PY=${PY:-/opt/conda/bin/python}
H=$1; JOBPY=$2; shift 2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
[ -f /workspace/deadline ] || echo $(( $(date +%s) + H*3600 )) > /workspace/deadline
say() { echo "$(date '+%F %T') [run] $*" >> /workspace/job.log; }
# last resort, independent of this script: stop (never destroy) the instance an hour after the deadline
nohup setsid bash -c "while sleep 120; do [ \$(date +%s) -gt \$(( \$(cat /workspace/deadline) + 3600 )) ] && { bash $PWD/vast_self.sh stop >> /workspace/job.log 2>&1; break; }; done" > /dev/null 2>&1 &
say "start $JOBPY, deadline $(date -d @$(cat /workspace/deadline))"
setsid $PY $JOBPY "$@" >> /workspace/job.log 2>&1 &
JOB=$!
while kill -0 $JOB 2>/dev/null; do
  if [ $(date +%s) -gt $(cat /workspace/deadline) ]; then
    say "deadline passed, stopping the job"; kill -TERM -- -$JOB 2>/dev/null; sleep 30; kill -KILL -- -$JOB 2>/dev/null
  fi
  sleep 20
done
wait $JOB; say "job ended (rc $?)"
UP=1
for i in 1 2 3; do timeout 2400 $PY $JOBPY "$@" --upload_only >> /workspace/job.log 2>&1 && { UP=0; break; }; sleep 60; done
if [ -f /workspace/HOLD ]; then say "HOLD set: leaving the instance running (watchdog stops it an hour after the deadline)"
elif [ $UP = 0 ]; then say "uploaded, destroying the instance"; bash vast_self.sh destroy >> /workspace/job.log 2>&1
else say "upload failed, stopping the instance (disk kept)"; bash vast_self.sh stop >> /workspace/job.log 2>&1; fi
