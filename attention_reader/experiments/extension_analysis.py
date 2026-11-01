"""Summarize the executed extension; aggregate families before paired uncertainty."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import pathlib,json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).parent;O=R/'results/extension';O.mkdir(exist_ok=True)
C=R/'specificity/results/cloud/58704070/results';F=R/'figures'
def save(p,x):
 q=p.with_suffix('.new');q.write_text(json.dumps(x,indent=2),encoding='utf-8');q.replace(p)
def fig(name):
 plt.tight_layout();plt.savefig(F/(name+'.png'),dpi=160,bbox_inches='tight');plt.savefig(F/(name+'.svg'),bbox_inches='tight');plt.close()
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafbf9','axes.facecolor':'#fafbf9'})
long=json.loads((R/'results/long/analysis.json').read_text());rows=long['rows'];arms=['own','unused','neighbor','mean'];labels=['Own context forecast','Forecast unused','Other copy forecast','Private running average']
fig_,ax=plt.subplots(1,3,figsize=(12,3.6))
for j,(key,title) in enumerate([('persistence','Distinctive selection over time'),('responsiveness','Selection follows input genre'),('forecast_error_over_private_mean','Prediction error / average error')]):
 for seed in [83,84]:
  ax[j].plot(range(4),[np.mean([r[key] for r in rows if r['cohort']==seed and r['condition']==a]) for a in arms],'-o',label=f'Cohort {seed}')
 ax[j].set_xticks(range(4),labels,rotation=22,ha='right');ax[j].set_title(title);ax[j].set_ylabel('Correlation difference' if j<2 else 'Error ratio')
ax[0].legend();fig('16-longer-cohorts')
unusual=long['ranked_unusual'];x=[r['persistence'] for r in unusual];y=[r['late_two_prototype_error_ratio'] for r in unusual]
plt.figure(figsize=(7,4));plt.scatter(x,y,c=[r['cohort'] for r in unusual],cmap='coolwarm',s=35);r=unusual[0];plt.annotate(f"Selected: cohort {r['cohort']}, copy {r['copy']}",(x[0],y[0]),xytext=(12,8),textcoords='offset points');plt.xlabel('Continuing distinctive selection (correlation difference)');plt.ylabel('Late summary error / late summary spread');plt.title('The unusual case is not the most persistent case');fig('17-unusual-case')
direction=json.loads((R/'results/directions/analysis.json').read_text());fig_,ax=plt.subplots(1,2,figsize=(11,3.6))
colors={'stance_toward_reading':'#226582','stance_random':'#888888','picture_neighbor':'#b57232','picture_random':'#6a9c4e','picture_neighbor_turn_blocked':'#b04378'}
for r in direction['rows']:
 if r['at']!=15000 or r['condition'] not in colors:continue
 name={'stance_toward_reading':'Stance toward reading','stance_random':'Stance in a random direction','picture_neighbor':'Summary toward another copy','picture_random':'Summary in a random direction','picture_neighbor_turn_blocked':'Summary change; turn route blocked'}[r['condition']]
 ax[0].plot(r['attention_curve'],label=name,c=colors[r['condition']]);ax[1].plot(np.rad2deg(r['stance_angle_curve']),label=name,c=colors[r['condition']])
cone=json.loads((R/'results/directions/stance_cone.json').read_text());cr=next(r for r in cone['rows'] if r['at']==15000);ax[0].plot(cr['attention_curve'],ls='--',c='#683ca0',label='Stance change preserving old angle');ax[1].plot(cr['stance_angle_degrees_curve'],ls='--',c='#683ca0');
ax[0].set_ylabel('Attention distribution difference');ax[1].set_ylabel('Stance distance from twin (degrees)')
for a in ax:a.set_xlabel('Readings after a single 15-degree change')
ax[0].legend(fontsize=8);fig('18-direction-routes')
sens=json.loads((R/'results/sensitivities/analysis.json').read_text());ratios=[]
for r in sens['rows']:
 if r['mode']=='context_feedback_only':
  b=next(x for x in sens['rows'] if x['scene']==r['scene'] and x['copy']==r['copy'] and x['mode']=='own_context');ratios.append(r['jacobian_norm']/b['jacobian_norm'])
modes=['own_context','own_frozen','private_mean','unused','context_feedback_only'];fig_,ax=plt.subplots(1,2,figsize=(11,3.6))
ax[0].scatter(range(len(ratios)),100*np.array(ratios),s=18);ax[0].axhline(100*np.median(ratios),ls='--',c='gray');ax[0].set_xlabel('Scene/copy local test (32 cases; 4 families)');ax[0].set_ylabel('Context-response derivative / full derivative (%)');ax[0].set_title('Isolating the changing forecast at the same base state')
for j,k in enumerate(['J_lens_basis_fraction','activity_PCA_fraction']):ax[1].bar(np.arange(5)+j*.25,[100*np.mean([r[k] for r in sens['rows'] if r['mode']==m]) for m in modes],width=.25,label=['J-lens-derived 64 directions','Activity PCA, 64 directions'][j])
ax[1].axhline(100*64/4096,ls=':',c='gray',label='Isotropic rank fraction');ax[1].set_xticks(np.arange(5)+.125,['Context','Frozen value','Private mean','Unused','Context route only'],rotation=22,ha='right');ax[1].set_ylabel('Local sensitivity energy in subspace (%)');ax[1].legend(fontsize=8);fig('19-context-response')
common=[]
for p in C.glob('common_context_*.json'):
 d=json.loads(p.read_text());common += [dict(r,family=d['scene']['family']) for r in d['rows']]
for row in common:row['return_alignment']=row['copy_template_return']/max(row['reading_change'],1e-12)
assert len({r['scene'] for r in common})==16
conditions=['full','unused_schema','mean_memory','random_act','rotated_act','fixed_act','neighbor_act'];layers=[12,16,20,24,28];metrics=['prediction_kl','next_token_logprob_change','processed_return_norm','reading_change','attention_change','copy_template_return','return_alignment']
families=sorted({r['family'] for r in common});lookup={(r['scene'],r['position'],r['layer'],r['copy'],r['condition']):r for r in common};mismatch=[];rotation=[]
for r in common:
 b=lookup[(r['scene'],r['position'],r['layer'],r['copy'],'full')];mismatch.append(abs(r['effective_action_norm']-b['effective_action_norm'])/max(b['effective_action_norm'],1e-9))
 if r.get('rotation_direction_cosine') is not None:rotation.append(r['rotation_direction_cosine'])
summary=[];contrasts=[];rng=np.random.default_rng(661)
for layer in layers:
 for condition in conditions:
  selected=[r for r in common if r['layer']==layer and r['condition']==condition]
  summary.append(dict(layer=layer,condition=condition,**{k:float(np.mean([r[k] for r in selected])) for k in metrics}))
  if condition=='full':continue
  for key in metrics:
   differences=[]
   for family in families:
    rr=[r for r in selected if r['family']==family];differences.append(float(np.mean([lookup[(r['scene'],r['position'],layer,r['copy'],'full')][key]-r[key] for r in rr])))
   boot=np.mean(rng.choice(differences,size=(4000,len(differences)),replace=True),axis=1)
   contrasts.append(dict(layer=layer,control=condition,metric=key,full_minus_control=float(np.mean(differences)),family_differences=differences,descriptive_family_bootstrap95=np.percentile(boot,[2.5,97.5]).tolist()))
result=dict(scenes=16,families=families,rows=len(common),summary=summary,contrasts=contrasts,
 qualification=dict(maximum_relative_effective_norm_mismatch=max(mismatch),rotation_direction_cosine_minimum=min(rotation),layer28_return_max=max(r['processed_return_norm'] for r in common if r['layer']==28)),
 sensitivity_median_context_derivative_ratio=float(np.median(ratios)),
 limits=['One-step counterfactuals share the same text and host cache; closed-loop generations do not.', 'Copies, probe positions and paired variants are nested in eight scene families.', 'KL measures strength of a distribution change, not improvement or awareness.', 'Positive template return means movement toward the copy-specific long-run summary direction, not recognized ownership.'])
save(O/'common_context.json',result)
fig_,ax=plt.subplots(1,3,figsize=(13,3.8))
for c in ['full','mean_memory','unused_schema','random_act','rotated_act','fixed_act']:
 rr=[next(r for r in summary if r['layer']==l and r['condition']==c) for l in layers]
 for j,k in enumerate(['prediction_kl','processed_return_norm','reading_change']):ax[j].plot(layers,[r[k] for r in rr],'-o',label=c.replace('_',' '),ms=3)
for a,title,ylabel in zip(ax,['Changes to the next-token distribution','Activity returning to the reader','Changes to the 32-number reading'],['KL from unchanged model','Layer-24 activity distance','Euclidean reading distance']):a.set_title(title);a.set_xlabel('Insertion layer (zero-based)');a.set_ylabel(ylabel);a.set_xticks(layers)
ax[0].legend(fontsize=8);fig('20-action-layer-comparison')
fig_,ax=plt.subplots(1,2,figsize=(11,3.8))
for j,key in enumerate(['return_alignment','next_token_logprob_change']):
 for c in ['full','mean_memory','unused_schema','random_act','rotated_act','fixed_act']:
  rr=[next(r for r in summary if r['layer']==l and r['condition']==c) for l in layers];ax[j].plot(layers,[np.nan if key=='return_alignment' and r['layer']==28 else r[key] for r in rr],'-o',label=c.replace('_',' '),ms=3)
 ax[j].axhline(0,c='gray',lw=.7);ax[j].set_xlabel('Insertion layer (zero-based)');ax[j].set_xticks(layers)
ax[0].set_ylabel('Cosine of reading change with own-minus-other template');ax[0].set_title('Does the returned reading reinforce its earlier tendency?');ax[1].set_ylabel('Change in log-probability of the authored next token');ax[1].set_title('A local prediction test, not a general quality score');ax[0].legend(fontsize=8);fig('21-return-direction')
if (R/'results/forecast_direction/DONE').exists():
 forecasts=[];fig_,ax=plt.subplots(1,2,figsize=(11,3.8))
 for seed in [83,84]:
  for arm in ['full','frozen_direction','rotated_direction','random_fixed_direction']:
   z=np.load(R/f'results/forecast_direction/{seed}_{arm}.npz');P=z['profiles'].astype(float)[:-1];P-=P.mean(-1,keepdims=True);P/=np.linalg.norm(P,axis=-1,keepdims=True).clip(1e-12);s=P.sum(1);ix=np.arange(len(P));mask=ix[None]>=ix[:,None]+5
   values=[]
   for c in range(16):values.append(float(((P[:,c]@P[:,c].T)-(P[:,c]@(s-P[:,c]).T/15))[mask].mean()))
   forecasts.append(dict(cohort=seed,condition=arm,persistence=values,mean=float(np.mean(values))))
   if seed==83:
    picture=z['picture'][:,0];centers=np.load(R/'results/long/own_83/trace.npz')['picture'][600:1400].reshape(-1,32)
    from sklearn.cluster import KMeans
    centers=KMeans(n_clusters=2,random_state=774,n_init=10).fit(centers).cluster_centers_;error=((picture[:,None]-centers[None])**2).sum(-1).min(-1)
    ax[1].plot(np.arange(8000,16720,100),np.array([error[i:i+100].mean() for i in range(0,len(error),100)]),label=arm.replace('_',' '))
 for seed in [83,84]:ax[0].plot(range(4),[r['mean'] for r in forecasts if r['cohort']==seed],'-o',label=f'Cohort {seed}')
 ax[0].set_xticks(range(4),['Own changing forecast','Initial direction','Rotated forecast','Fixed random direction'],rotation=22,ha='right');ax[0].set_ylabel('Distinctive selection (correlation difference)');ax[0].legend();ax[1].set_xlabel('Reading');ax[1].set_ylabel('Squared distance to early prototype');ax[1].set_title('Selected unusual copy: departure from earlier summaries');ax[1].legend(fontsize=8);fig('22-forecast-pattern-control');save(O/'forecast_direction.json',dict(rows=forecasts,scope='8720-reading continuation of both 16-copy cohorts; norm profiles yoked per copy/time to full replay; not a self-contained replacement controller'))
rare=json.loads((O/'rare_generalization.json').read_text());plt.figure(figsize=(7,4))
for cohort in [83,84]:
 for copy in range(16):
  v=[next(x['late_error_ratio'] for x in rare['rows'] if x['cohort']==cohort and x['copy']==copy and x['prototypes']==n) for n in [2,4,8]]
  selected=cohort==83 and copy==0;plt.plot([2,4,8],v,'-o',color='#b66731' if selected else '#8399a855',lw=2.5 if selected else .8,ms=5 if selected else 2,label='Selected unusual copy' if selected else None)
plt.xticks([2,4,8]);plt.xlabel('Number of shared prototypes fitted in the first stream');plt.ylabel('Later squared error / earlier-mean baseline error');plt.title('Four earlier prototypes describe the unusual later summary');plt.legend();fig('24-rare-generalization')
print(json.dumps(result['qualification']));print('Common-context summary',json.dumps([r for r in summary if r['layer']==16]));print('Figures complete')
