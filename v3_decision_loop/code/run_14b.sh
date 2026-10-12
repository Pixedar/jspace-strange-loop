#!/bin/bash
# Qwen3-14B replication: E1 (fresh 'rep' scenarios) and E2 regime A at the same relative depths as the chosen 8B loop layers.
cd /workspace/code
PY=/opt/conda/bin/python
export V3_OUT=/workspace/v3 OMP_NUM_THREADS=8
read LR8 LW8 < /workspace/v3/e2/qwen3-8b/chosen_layers.txt
LR=$(( (LR8 * 40 + 18) / 36 )); LW=$(( (LW8 * 40 + 18) / 36 ))
echo "$(date '+%F %T') [14b] loop layers Lr=$LR Lw=$LW (8B $LR8/$LW8)" >> /workspace/job.log
# an early 14B run may have been started by hand; wait for it so the two never write the same files
while pgrep -f "e1.py --model qwen3-14b" > /dev/null || pgrep -f "e2.py --model qwen3-14b" > /dev/null; do sleep 30; done
$PY e1.py --model qwen3-14b --scale rep --stages qual,h1,h2 > /workspace/e1_14b.log 2>&1 &
P1=$!
$PY e2.py --model qwen3-14b --n 50 --T 64 --tag rep --Lr $LR --Lw $LW --stages calib,A1,A2,pert > /workspace/e2_14b.log 2>&1 &
P2=$!
wait $P1; R1=$?; wait $P2; R2=$?
echo "$(date '+%F %T') [14b] rc e1=$R1 e2=$R2" >> /workspace/job.log
exit $(( R1 | R2 ))
