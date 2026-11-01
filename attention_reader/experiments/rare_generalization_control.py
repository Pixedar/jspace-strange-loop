"""Check whether the unusual summary score is a narrow fitting-context artifact."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import pathlib,json,numpy as np
from sklearn.cluster import KMeans
R=pathlib.Path(__file__).parent;genre=np.load(R/'results/long/input_index.npz')['genre'];rows=[]
for cohort in [83,84]:
 p=np.load(R/f'results/long/own_{cohort}/trace.npz')['picture'];late=p[-4000:];early=p[272:3000]
 for n in [2,4,8]:
  km=KMeans(n_clusters=n,random_state=774,n_init=10).fit(early.reshape(-1,32));centers=km.cluster_centers_
  for c in range(16):
   dist=((late[:,c,None]-centers[None])**2).sum(-1).min(-1);spread=((late[:,c]-early[:,c].mean(0))**2).sum(-1).mean()
   rows.append(dict(cohort=cohort,copy=c,prototypes=n,late_error_ratio=float(dist.mean()/max(spread,1e-12)),by_genre={str(g):float(dist[genre[-4000:]==g].mean()) for g in range(5)}))
  print(cohort,n,flush=True)
out=dict(rows=rows,original_fit_genre_counts=np.bincount(genre[600:1400],minlength=5).tolist(),broad_fit_genre_counts=np.bincount(genre[272:3000],minlength=5).tolist(),protocol='Shared prototypes fitted on all 16 copies over the complete first stream after warm-up, readings 272..2999. Evaluate final 4000 readings. Original unusual-copy selection is retained; this follow-up checks its interpretation, not a new outlier search.',limitation='Reconstruction with more centers is descriptive; it is not proof of more meanings or a causal replacement experiment.')
p=R/'results/extension/rare_generalization.json';q=p.with_suffix('.new');q.write_text(json.dumps(out,indent=2));q.replace(p);print([r for r in rows if r['cohort']==83 and r['copy']==0])
