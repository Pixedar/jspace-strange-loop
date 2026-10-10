#!/bin/bash
# Stage 2: wait until stage 1 (vast_run.sh, told to keep the instance with /workspace/HOLD) has finished and uploaded,
# then run the next job the same unattended way. That run destroys the instance at its end.
#   bash vast_run2.sh <hours> <job.py> <job args...>
cd "$(dirname "$0")"
until grep -q "\[run\] HOLD set" /workspace/job.log 2>/dev/null; do sleep 20; done
echo "$(date '+%F %T') [run2] stage 1 finished, starting stage 2: $*" >> /workspace/job.log
rm -f /workspace/HOLD
exec bash vast_run.sh "$@"
