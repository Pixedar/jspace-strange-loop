"""Export full-dimensional starting-state responses for the interactive experiment.

Requires the selected numeric artifacts from ARTIFACTS.json. No model inference.
All 60 recorded contexts and all four copies are included for both interfaces.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

root=Path(__file__).resolve().parents[1]
probes=torch.randn(12,8,generator=torch.Generator().manual_seed(4242))
probes=(probes/probes.norm(dim=-1,keepdim=True)).numpy()
data={'schema':1,'distance':'Euclidean distance in the original eight-dimensional unit control state; range 0–2.',
      'probes':probes.tolist(),'conditions':[],'sources':[]}
times=None
for name,label in [('dev_s0_seed0_spot','Describes its own responses'),('another_s1_spot','Describes another loop’s recordings')]:
    path=root/'self_reference/results/v24'/name/'fields_iface_3000.npz'
    with np.load(path) as z:
        dest=z['dest']
        assert dest.shape==(60,4,12,8)
        if times is not None: assert np.array_equal(times,z['times'])
        times=z['times'].copy()
        assert np.max(np.abs(np.linalg.norm(dest,axis=-1)-1))<1e-5
        data['conditions'].append({'id':name,'label':label,'destinations':dest.tolist()})
    data['sources'].append({'path':str(path.relative_to(root)).replace('\\','/'),
                            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'array':'dest[context,copy,starting_state,coordinate]'})
data['tokens']=times.tolist()
data['default_context']=int(np.flatnonzero(times==600)[0])
data['cut_source']='self_reference/results/v24/dev_s0_seed0_spot/self_formation.json'
data['cuts']=json.loads((root/data['cut_source']).read_text(encoding='utf-8'))['jumps']
data['measurements']=json.loads((root/'self_reference/results/verified_summary.json').read_text(encoding='utf-8'))['runs']
out=root/'interactive/probe_responses.json'
out.write_bytes((json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8'))
for condition in data['conditions']:
    d=np.array(condition['destinations'])[30,0]
    pair=np.linalg.norm(d[:,None]-d[None,:],axis=-1)
    print(condition['id'],'token 600, copy 1: maximum next-state separation',pair.max())
print('Exported',len(times)*4*12*2,'measured next states;',out.stat().st_size,'bytes')
