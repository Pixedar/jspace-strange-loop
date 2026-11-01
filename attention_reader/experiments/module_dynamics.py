"""Inspect the actual dimensional bottleneck; do not name coordinates as faculties."""
import pathlib,json,io
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).parent;rows=[]
for seed in [10,12,27]:
 z=np.load(R/f'results/signals_seed{seed}.npz')
 for module in ['stance','reading','schema']:
  X=z[module][300:3000]
  for c in range(4):
   x=X[:,c].astype(float);center=x-x.mean(0);total=(center**2).sum();p=np.arange(len(x))%2
   pred=np.stack([x[p==i].mean(0) for i in range(2)])[p]
   parity=1-((x-pred)**2).sum()/max(total,1e-12)
   lag1=np.linalg.norm(np.diff(x,axis=0),axis=1).mean();lag2=np.linalg.norm(x[2:]-x[:-2],axis=1).mean()
   means=x.reshape(-1,2,x.shape[-1]).mean(1);fraction=float(((means-means.mean(0))**2).mean()/max((center**2).mean(),1e-12))
   sv=np.linalg.svd(center,compute_uv=False);energy=sv**2;eff=float(energy.sum()**2/max((energy**2).sum(),1e-12))
   row=dict(seed=seed,copy=c,module=module,parity_variance=float(parity),lag2_over_lag1=float(lag2/max(lag1,1e-12)),pair_mean_variance_fraction=fraction,effective_dimension=eff,
       first_component_fraction=float(energy[0]/energy.sum()),first_three_fraction=float(energy[:3].sum()/energy.sum()))
   if module=='reading':
    train=x[:1100];test=x[1400:];fit=KMeans(n_clusters=2,random_state=532,n_init=10).fit(train)
    distances=((test[:,None]-fit.cluster_centers_[None])**2).sum(-1)
    score=1-distances.min(1).sum()/((test-train.mean(0))**2).sum()
    row['two_center_held_out_r2']=float(score);row['later_cluster_fractions']=[float((distances.argmin(1)==i).mean()) for i in range(2)]
   rows.append(row)
out=dict(rows=rows,scope='three selected recorded lives; steps 300–2999; within-copy centering',
 caveats=['descriptive follow-up after inspecting projections','fixed parity misses drifting cycles','linear effective dimension is not a count of concepts or states','two-center fit trained on steps 300–1399 and tested 1700–2999; not a causal replacement experiment'])
(R/'results/module_dynamics.json').write_text(json.dumps(out,indent=2))
selected=[r for r in rows if r['module']=='reading'];fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True);yy=np.arange(12)
axes[0].barh(yy,[r['first_component_fraction'] for r in selected],color='#447b98');axes[1].barh(yy,[r['two_center_held_out_r2'] for r in selected],color='#b97843')
axes[0].set_yticks(yy,[f'Life {r["seed"]}, copy {r["copy"]}' for r in selected]);axes[0].invert_yaxis();axes[0].set(xlim=(0,1),xlabel='Fraction of within-copy variation',title='One direction captures most summary variation')
axes[1].set(xlim=(0,1),xlabel='Later fit relative to the training-mean baseline',title='Two earlier summary centers also compress later readings')
fig.suptitle('The 32-number summary does not use 32 equally varying directions',y=1.01)
for ext in ['png','svg']:
 b=io.BytesIO();fig.savefig(b,format=ext,dpi=170,bbox_inches='tight');p=R/f'figures/14-summary-bottleneck.{ext}';q=p.with_suffix('.new');q.write_bytes(b.getvalue());q.replace(p)
plt.close(fig)
print(json.dumps(selected,indent=2))
