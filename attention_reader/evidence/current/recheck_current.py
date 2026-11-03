"""Recompute current recorded-run aggregates from unchanged compact arrays."""
from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parent;z=np.load(R/'current_runs.npz');metadata=json.loads(str(z['metadata']));expected=json.loads((R/'expected_aggregates.json').read_text());G=z['genre'];sg=np.array([np.bincount(G[i:i+100]).argmax() for i in range(0,3000,100)])
def corr(a,b):return float(np.corrcoef(a,b)[0,1])
names={'pj_c8_SIx':'reference','pk_c8_L25':'own_quarter','pk_c8_Ln25':'other_quarter','pk_c8_L50':'own_half','pk_c8_Ln50':'other_half','pk_c8_L':'own_full','pk_c8_Ln':'other_full'}
max_error=0
for m in metadata:
    prefix=m['group']+'/'+m['run']+'/';L=z[prefix+'lens_seg'].astype(float);own=[];other=[];same=[];diff=[]
    for i in range(3,len(L)):
        for j in range(i+3,len(L)):
            for c in range(4):
                v=corr(L[i,c],L[j,c]);own.append(v);other.append(np.mean([corr(L[i,c],L[j,k]) for k in range(4) if k!=c]));(same if sg[i]==sg[j] else diff).append(v)
    values=dict(persistence=float(np.mean(own)-np.mean(other)),responsiveness=float(np.mean(same)-np.mean(diff)),attention_apart=float(np.nanmean(z[prefix+'att_apart'][1500:])))
    arm=m['run'].rsplit('_seed',1)[0]
    label=names[arm] if m['group']=='dose' else arm.replace('pa_c8_','')
    source=expected['dose' if m['group']=='dose' else 'prototype']['rows'];row=next(r for r in source if r['seed']==m['seed'] and r['arm']==label)
    err=max(abs(values[k]-row[k]) for k in values);max_error=max(max_error,err);assert err<1e-10,(m['run'],values,row)
print(json.dumps(dict(verified_runs=len(metadata),max_aggregate_error=max_error,language_model_loaded=False),indent=2))
