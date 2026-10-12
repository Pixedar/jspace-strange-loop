#!/bin/bash
# Main 8B runs in parallel on one GPU: E1 main, Huginn, and E2 main at the loop layers chosen by the pre-declared rule.
cd /workspace/code
PY=/opt/conda/bin/python
export V3_OUT=/workspace/v3 OMP_NUM_THREADS=8
$PY sel_layers.py /workspace/v3/e2/qwen3-8b > /workspace/v3/e2/qwen3-8b/layer_choice.txt 2>&1
PAIR=$($PY -c "import json;print(json.load(open('/workspace/v3/e2/qwen3-8b/layer_choice.json'))['choice']['pair'])")
LR=${PAIR%_*}; LW=${PAIR#*_}
echo "$(date '+%F %T') [main] loop layers Lr=$LR Lw=$LW" >> /workspace/job.log
echo "$LR $LW" > /workspace/v3/e2/qwen3-8b/chosen_layers.txt
$PY e1.py --model qwen3-8b --scale main --stages qual,h1,h2,think > /workspace/e1_main.log 2>&1 &
P1=$!
$PY huginn.py --n 100 --K 64 --tag main > /workspace/huginn_main.log 2>&1 &
P2=$!
$PY e2.py --model qwen3-8b --n 100 --T 64 --tag main --Lr $LR --Lw $LW --stages calib,A1,A2,BC,pert > /workspace/e2_main.log 2>&1 &
P3=$!
wait $P1; R1=$?; wait $P2; R2=$?; wait $P3; R3=$?
echo "$(date '+%F %T') [main] rc e1=$R1 huginn=$R2 e2=$R3" >> /workspace/job.log
exit $(( R1 | R2 | R3 ))
