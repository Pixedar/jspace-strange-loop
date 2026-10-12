#!/bin/bash
# Pull compact result files (JSON summaries, figures, small logs) from the instance, or from the HF dataset if the
# instance is gone. Raw arrays stay on HF (Pixedar/jspace-v3, prefix v3/).
#   bash pull_results.sh [ssh|hf]
DST=${DST:-./results}
mkdir -p $DST/figs
MODE=${1:-ssh}
FILES="e1/qwen3-8b/main/results_e1.json e1/qwen3-8b/main/qual/qual.json e1/qwen3-8b/main/think/think.json
e1/qwen3-8b/pilot2/results_e1.json e1/qwen3-14b/rep/results_e1.json e1/qwen3-14b/rep/qual/qual.json
e2/qwen3-8b/main/results_e2.json e2/qwen3-8b/main/calib/calib.json e2/qwen3-8b/main/calib/fidelity.json e2/qwen3-8b/main/gains.json
e2/qwen3-8b/layer_choice.json e2/qwen3-14b/rep/results_e2.json e2/qwen3-14b/rep/gains.json e2/qwen3-14b/rep/calib/fidelity.json
huginn/main/results_huginn.json huginn/main/provenance.json phase0/results.json ouro/main/results_ouro.json atlas.json e2/qwen3-8b/layer_choice.txt
e1/qwen3-8b/main/results_h2_label.json e1/qwen3-14b/rep/results_h2_label.json"
if [ "$MODE" = ssh ]; then
  SSHOPT="-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o IdentitiesOnly=yes -i $HOME/.ssh/id_rsa"
  for f in $FILES; do
    out=$DST/$(echo $f | tr '/' '_')
    scp -q -P 40560 $SSHOPT root@174.78.228.101:/workspace/v3/$f "$out" 2>/dev/null && echo "got $f"
  done
  scp -q -P 40560 $SSHOPT "root@174.78.228.101:/workspace/v3/figs/*.png" $DST/figs/ 2>/dev/null && echo "got figs"
  scp -q -P 40560 $SSHOPT root@174.78.228.101:/workspace/job.log $DST/job.log 2>/dev/null && echo "got job.log"
else
  TOK=$(tr -d ' \r\n' < ~/.cache/huggingface/token)
  for f in $FILES; do
    out=$DST/$(echo $f | tr '/' '_')
    curl -sL --ssl-no-revoke -m 300 -H "Authorization: Bearer $TOK" -o "$out" \
      "https://huggingface.co/datasets/Pixedar/jspace-v3/resolve/main/v3/$f" -w "%{http_code} $f\n"
  done
  for f in $(curl -sL --ssl-no-revoke -m 60 -H "Authorization: Bearer $TOK" \
      "https://huggingface.co/api/datasets/Pixedar/jspace-v3/tree/main/v3/figs" | python -c "import sys,json;[print(x['path']) for x in json.load(sys.stdin) if x['path'].endswith('.png')]"); do
    curl -sL --ssl-no-revoke -m 120 -H "Authorization: Bearer $TOK" -o "$DST/figs/$(basename $f)" \
      "https://huggingface.co/datasets/Pixedar/jspace-v3/resolve/main/$f" -w "%{http_code} $f\n"
  done
fi
