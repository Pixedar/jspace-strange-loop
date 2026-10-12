#!/bin/bash
# Unattended research-v5 instance. Runs job.py (the queue in /workspace/queue.txt) until it exits or the deadline passes,
# uploads /workspace/v5 to the private HF dataset (prefix v5), then destroys the instance; if the upload failed it stops
# the instance instead (disk kept). With /workspace/HOLD it leaves the instance running. An independent watchdog stops
# the instance an hour after the deadline.
#   bash run_v5.sh <hours>        extend: echo $(( $(date +%s) + 3600 )) > /workspace/deadline
cd /workspace/code
PY=${PY:-/opt/conda/bin/python}
H=${1:-3}
NC=$(nproc); NT=$(( NC < 16 ? NC : 16 ))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT V5_OUT=/workspace/v5 PYTHONWARNINGS=ignore V3_CODE=/workspace/code/v3
[ -s /workspace/deadline ] || echo $(( $(date +%s) + $(awk "BEGIN{print int($H*3600)}") )) > /workspace/deadline
say() { echo "$(date '+%F %T') [run] $*" >> /workspace/job.log; }
nohup setsid bash -c "while sleep 120; do [ \$(date +%s) -gt \$(( \$(cat /workspace/deadline) + 3600 )) ] && { bash /workspace/code/vast_self.sh stop >> /workspace/job.log 2>&1; break; }; done" > /dev/null 2>&1 &
say "start, deadline $(date -d @$(cat /workspace/deadline)), threads $NT of $NC"
setsid $PY job.py >> /workspace/job.log 2>&1 &
JOB=$!
while kill -0 $JOB 2>/dev/null; do
  if [ $(date +%s) -gt $(cat /workspace/deadline) ]; then
    say "deadline passed, stopping the job"; kill -TERM -- -$JOB 2>/dev/null; sleep 30; kill -KILL -- -$JOB 2>/dev/null
  fi
  sleep 20
done
wait $JOB; say "job ended (rc $?)"
UP=1
for i in 1 2 3; do timeout 2400 $PY job.py --upload >> /workspace/job.log 2>&1 && { UP=0; break; }; sleep 60; done
if [ -f /workspace/HOLD ]; then say "HOLD set: leaving the instance running (watchdog stops it an hour after the deadline)"
elif [ $UP = 0 ]; then say "uploaded, destroying the instance"; bash vast_self.sh destroy >> /workspace/job.log 2>&1
else say "upload failed, stopping the instance (disk kept)"; bash vast_self.sh stop >> /workspace/job.log 2>&1; fi
