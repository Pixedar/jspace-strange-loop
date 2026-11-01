"""Prespecified phenotype screen; unusual is not a self-awareness verdict."""
import os,pathlib,json,time,sys
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np,torch
from sklearn.cluster import KMeans
R=pathlib.Path(__file__).parent;OUT=R/'results/long';rows=[]
protocol=json.loads((OUT/'PROTOCOL.json').read_text());genre=np.load(OUT/'input_index.npz')['genre'];N=len(genre)
def normalize(x):
 x=x-x.mean(-1,keepdims=True);return x/np.linalg.norm(x,axis=-1,keepdims=True).clip(1e-12)
for seed in protocol['cohorts']:
 for arm in protocol['arms']:
    path=OUT/f'{arm}_{seed}/trace.npz'
    if not path.exists():continue
    z=np.load(path);L=z['profiles'].astype(float)[:-1];V=normalize(L);S=z['s'];P=z['picture'];errors=z['schema_error'];meanerrors=z['mean_error'];entropy=z['attention_entropy']
    n,B,d=V.shape;ix=np.arange(n);mask=(ix[:,None]>=3)&(ix[None]>=3)&(ix[None]>=ix[:,None]+5)
    sg=np.array([np.bincount(genre[i*100:(i+1)*100].clip(0)).argmax() for i in range(n)])
    centers=KMeans(n_clusters=2,random_state=774,n_init=10).fit(P[600:1400].reshape(-1,32)).cluster_centers_
    totalV=V.sum(1)
    for c in range(B):
        C=V[:,c]@V[:,c].T;O=V[:,c]@(totalV-V[:,c]).T/(B-1)
        persistence=float((C-O)[mask].mean());same=mask&(sg[:,None]==sg[None]);different=mask&(sg[:,None]!=sg[None])
        responsiveness=float(C[same].mean()-C[different].mean())
        x=S[300::5,c].astype(float);center=x-x.mean(0);sv=np.linalg.svd(center,compute_uv=False);v=sv**2;effective=float(v.sum()**2/(v**2).sum())
        full=S[300:,c].astype(float);lag1=np.linalg.norm(np.diff(full,axis=0),axis=1).mean();lag2=np.linalg.norm(full[2:]-full[:-2],axis=1).mean()
        pp=P[-4000:,c].astype(float);dist=((pp[:,None]-centers[None])**2).sum(-1).min(-1);spread=((pp-P[600:1400,c].mean(0))**2).sum(-1).mean()
        novelty=float(dist.mean()/max(spread,1e-12))
        rows.append(dict(cohort=seed,condition=arm,copy=c,persistence=persistence,responsiveness=responsiveness,
            stance_effective_dimension=effective,lag2_over_lag1=float(lag2/max(lag1,1e-12)),stance_variance=float((center**2).mean()),
            late_two_prototype_error_ratio=novelty,forecast_error_over_private_mean=float(errors[300:,c].sum()/max(meanerrors[300:,c].sum(),1e-12)),
            attention_entropy=float(entropy[300:,c].mean())))
    print('analyzed',seed,arm,flush=True)
own=[r for r in rows if r['condition']=='own'];assert len(own)==32,'Both own cohorts must be complete before selection';keys=['persistence','responsiveness','stance_effective_dimension','late_two_prototype_error_ratio','forecast_error_over_private_mean']
X=np.array([[r[k] for k in keys] for r in own]);med=np.median(X,0);mad=np.median(abs(X-med),0).clip(.01);Z=(X-med)/mad
dist=np.sqrt(((Z[:,None]-Z[None])**2).sum(-1));dist.sort(axis=1);outlier=dist[:,1:5].mean(1)
for r,s in zip(own,outlier):
 r['outlier_distance']=float(s);r['passes_basic_dynamics_filter']=bool(r['stance_variance']>1e-4 and r['lag2_over_lag1']>.25 and np.isfinite(s))
ranked=sorted([r for r in own if r['passes_basic_dynamics_filter']],key=lambda r:r['outlier_distance'],reverse=True)
chosen=ranked[0] if ranked else own[0];cohort=chosen['cohort'];group=[r for r in own if r['cohort']==cohort];selected=[]
for c in [chosen['copy'],sorted(group,key=lambda r:r['persistence'])[len(group)//2]['copy'],max(group,key=lambda r:r['persistence'])['copy'],0,5,10,15]:
 if c not in selected:selected.append(c)
 if len(selected)==4:break
snapshot=torch.load(OUT/f'own_{cohort}/state_{N}.pt',map_location='cpu',weights_only=False)
for name in ['s','prev_g','hist','habit','Z','C','shift']:
 v=snapshot['state'].get(name)
 if isinstance(v,torch.Tensor) and v.shape[0]==16:snapshot['state'][name]=v[selected].clone()
snapshot['state']['config']['copies']=4
snapshot['ordinary_mean']['total']=snapshot['ordinary_mean']['total'][selected].clone()
torch.save(snapshot,OUT/'selected_long.pt')
p=np.load(OUT/f'own_{cohort}/trace.npz')['picture'];templates=p[-1000:,selected].mean(0)
selection=dict(cohort=cohort,copies=selected,readings=N,selection='largest robust nearest-neighbor phenotype distance among basic-dynamics-filtered own trajectories, plus median/persistent/representative controls',
               interpretation='unusual among these 32 own trajectories; not a confirmed rare phenotype or self-awareness result',copy_templates=templates.tolist())
result=dict(protocol=protocol,rows=rows,ranked_unusual=ranked,selection=selection,
 warnings=['recorded LLM contexts reset between streams','larger cohort is not social interaction; copies have independent online states apart from calibration and comparison','rank is exploratory; ordinary controls and replay decide its meaning','linear dimensions and distances are not counts of mental faculties'])
(OUT/'analysis.json').write_text(json.dumps(result,indent=2));(OUT/'selection.json').write_text(json.dumps(selection,indent=2))
print(json.dumps(selection));print('own trajectories',len(own))
