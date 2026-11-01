"""Verify the carry invariant with explicit before/after alignment."""
import pathlib,json,numpy as np
R=pathlib.Path(__file__).parent;rows=[]
for cohort in [83,84]:
 for arm in ['own','unused','neighbor','mean']:
  z=np.load(R/f'results/long/{arm}_{cohort}/trace.npz');s=z['s'].astype(float);g=z['g'].astype(float)
  # trace.s[t] is the PRE-update stance; g[t] is the current decoded direction.
  # Pair s[t] with g[t-1], never with g[t].
  s=s[300:];previous=g[299:-1];norm=np.linalg.norm(s,axis=-1);gn=np.linalg.norm(previous,axis=-1)
  alignment=(s*previous).sum(-1)/(norm*gn)
  for c in range(16):rows.append(dict(cohort=cohort,condition=arm,copy=c,first_alignment=float(alignment[0,c]),mean_alignment=float(alignment[:,c].mean()),max_alignment_drift=float(abs(alignment[:,c]-alignment[0,c]).max()),max_stance_norm_drift=float(abs(norm[:,c]-norm[0,c]).max())))
out=dict(rows=rows,maximum_alignment_drift=max(r['max_alignment_drift'] for r in rows),maximum_norm_drift=max(r['max_stance_norm_drift'] for r in rows),indexing='Pre-update s[t] paired with previous g[t-1], t=300..16719.',derivation='The same orthogonal carry rotation maps previous decoded direction to current direction and rotates the stance, conserving their angle. This is a designed geometric memory, not learned self-recognition.')
p=R/'results/long/conserved_alignment.json';q=p.with_suffix('.new');q.write_text(json.dumps(out,indent=2));q.replace(p);print(out['maximum_alignment_drift'],out['maximum_norm_drift'])
