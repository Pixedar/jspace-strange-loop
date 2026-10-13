"""Frequency-controlled reading of the transport operator: residualise self-gain and leading-mode loading on log word
frequency (zipf) and list the words that stand out once frequency is accounted for."""
import json, os, sys, pathlib
import numpy as np
sys.path.insert(0, os.environ.get('V3_CODE', str(pathlib.Path(__file__).resolve().parents[2]/'common')))
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from wordfreq import zipf_frequency
from common import english_vocab
rev = 'b968826d9c46dd6066d109eabc6255188de91218'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B', revision=rev)
idx = json.loads(open(hf_hub_download('Qwen/Qwen3-8B', 'model.safetensors.index.json', revision=rev)).read())['weight_map']
with safe_open(hf_hub_download('Qwen/Qwen3-8B', idx['lm_head.weight'], revision=rev), 'pt') as h: W = h.get_tensor('lm_head.weight')
keep, words = english_vocab(tok, W)
z = np.load('/workspace/v5/transport/transport.npz'); cand = z['cand']; M = z['M'].astype(np.float64)
cw = [words[i] for i in cand]; zf = np.array([zipf_frequency(w.lower(), 'en') for w in cw])
ev, vec = np.linalg.eig(M); o = np.argsort(-ev.real); out = {}
X = np.stack([np.ones_like(zf), zf], 1)
def resid(y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]; return y - X @ b, float(np.corrcoef(zf, y)[0, 1])
sg = np.diag(M); r_sg, c_sg = resid(sg)
out['self_gain_corr_with_zipf'] = c_sg
out['self_gain_resid_top'] = [(cw[i], round(float(r_sg[i]), 3), round(float(zf[i]), 2)) for i in np.argsort(-r_sg)[:40]]
out['self_gain_resid_bottom'] = [(cw[i], round(float(r_sg[i]), 3), round(float(zf[i]), 2)) for i in np.argsort(r_sg)[:30]]
for k in range(4):
    v = vec[:, o[k]].real; v = v if v.sum() >= 0 else -v; r_v, c_v = resid(v)
    out[f'mode{k+1}'] = dict(eig=float(ev[o[k]].real), corr_with_zipf=c_v, top_raw=[cw[i] for i in np.argsort(-v)[:15]],
                             top_resid=[cw[i] for i in np.argsort(-r_v)[:15]])
json.dump(out, open('/workspace/v5/transport/freq_control.json', 'w'), indent=1)
print(json.dumps({k: (v if not isinstance(v, list) else v[:20]) for k, v in out.items()}, indent=0)[:4000])
