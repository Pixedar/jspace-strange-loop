"""Recompute the completed late-evening reader metrics, without a language model."""
from pathlib import Path
import os,json
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import numpy as np
R=Path(__file__).resolve().parent;z=np.load(R/'late_runs.npz');expected=json.loads((R/'late_expected.json').read_text());meta=json.loads(str(z['metadata']));G=z['genre'];sg=np.array([np.bincount(G[i:i+100]).argmax() for i in range(0,3000,100)]);error=0.
for m in meta:
 p=m['group']+'/'+m['run']+'/';L=z[p+'lens_seg'].astype(float);own=[];other=[];same=[];different=[]
 for i in range(3,len(L)):
  for j in range(i+3,len(L)):
   for c in range(4):
    v=float(np.corrcoef(L[i,c],L[j,c])[0,1]);own.append(v);other.append(np.mean([np.corrcoef(L[i,c],L[j,k])[0,1] for k in range(4) if k!=c]));(same if sg[i]==sg[j] else different).append(v)
 row=next(r for r in expected['rows'] if r['run']==m['run']);values=dict(persistence=float(np.mean(own)-np.mean(other)),responsiveness=float(np.mean(same)-np.mean(different)),attention_apart=float(np.nanmean(z[p+'att_apart'][1500:])),turn_per_token=float(np.nanmean(z[p+'turn'][300:])));s=z[p+'s'][255].astype(float);u=s/np.linalg.norm(s,axis=-1,keepdims=True);registered=all(u[a]@u[b]<=.999 for a in range(4) for b in range(a+1,4));assert registered==row['pasts_register']
 delta=max(abs(values[k]-row[k]) for k in values);error=max(error,delta);assert delta<1e-10,(m['run'],delta)
print(json.dumps(dict(records_verified=len(meta),including_reference_comparators=24,max_error=error,language_model_loaded=False),indent=2))
