"""Which frequent function words are in the transport candidate set, and where do they fall in the leading modes?"""
import json, os, sys, pathlib
import numpy as np
sys.path.insert(0, os.environ.get('V3_CODE', str(pathlib.Path(__file__).resolve().parents[2]/'common')))
import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from common import english_vocab
rev = 'b968826d9c46dd6066d109eabc6255188de91218'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B', revision=rev)
idx = json.loads(open(hf_hub_download('Qwen/Qwen3-8B', 'model.safetensors.index.json', revision=rev)).read())['weight_map']
with safe_open(hf_hub_download('Qwen/Qwen3-8B', idx['lm_head.weight'], revision=rev), 'pt') as h: W = h.get_tensor('lm_head.weight')
keep, words = english_vocab(tok, W)
z = np.load('/workspace/v5/transport/transport.npz'); cand = z['cand']; M = z['M'].astype(np.float64)
cw = [words[i] for i in cand]; pos = {w: j for j, w in enumerate(cw)}
ev, vec = np.linalg.eig(M); o = np.argsort(-ev.real); v1 = vec[:, o[0]].real; v1 = v1 if v1.sum() >= 0 else -v1
rank = {cw[j]: r for r, j in enumerate(np.argsort(-v1))}
for w in ['the', 'and', 'that', 'with', 'for', 'this', 'All', 'And', 'Well', 'answer', 'yeah', 'james', 'Alice']:
    print(w, 'in set' if w in pos else 'NOT in set', 'rank in mode 1:', rank.get(w), 'self-gain:', round(float(M[pos[w], pos[w]]), 2) if w in pos else None)
print('zipf-rank corr with mode-1 loading:', float(np.corrcoef(np.arange(len(cand)), v1)[0, 1]))
