#!/bin/bash
# After the main 8B runs: Huginn and Ouro natural recurrence start at once (small models); the exploratory X arm and the
# epistemic steering test (8B) start once no 14B job holds GPU memory, so the GPU never runs out of memory.
cd /workspace/code
PY=/opt/conda/bin/python
export V3_OUT=/workspace/v3 OMP_NUM_THREADS=8
read LR LW < /workspace/v3/e2/qwen3-8b/chosen_layers.txt
$PY huginn.py --n 100 --K 64 --tag main > /workspace/huginn_main.log 2>&1 &
P2=$!
$PY ouro.py --n 100 --K 8 --tag main > /workspace/ouro_main.log 2>&1 &
P3=$!
while pgrep -f "e2.py --model qwen3-14b" > /dev/null || pgrep -f "e1.py --model qwen3-14b" > /dev/null; do sleep 20; done
$PY e2_x.py --model qwen3-8b --n 100 --T 64 --tag main --Lr $LR --Lw $LW --stages X > /workspace/e2x_main.log 2>&1 &
P1=$!
$PY e1.py --model qwen3-8b --scale main --stages epi > /workspace/e1epi_main.log 2>&1 &
P4=$!
wait $P1; R1=$?; wait $P2; R2=$?; wait $P3; R3=$?; wait $P4; R4=$?
echo "$(date '+%F %T') [extra] rc x=$R1 huginn=$R2 ouro=$R3 epi=$R4" >> /workspace/job.log
exit $(( R1 | R2 | R3 | R4 ))
