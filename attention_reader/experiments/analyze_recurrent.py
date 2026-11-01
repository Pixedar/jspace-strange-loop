"""Family-paired effects of an attention model versus information-driven recurrence."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import pathlib,json,collections,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).parent;C=R/'recurrent/results/cloud/58714365/results';O=R/'results/extension';conditions=['full','recurrent_forecast','recurrent_selector','neighbor_forecast'];names=['Attention model','Recurrent forecast replacement','Whole recurrent selector','Other-copy forecast']
sem=[]
for p in sorted(C.glob('semantic_information_*.json')):
 d=json.loads(p.read_text());sem.extend([dict(r,scene=d['scene']['id'],family=d['scene']['family'],correct_option=d['correct_option'],options=d['options'],stem=d['stem']) for r in d['rows']])
assert len({r['scene'] for r in sem})==16
families=sorted({r['family'] for r in sem});rng=np.random.default_rng(7801)
def interval(v):return np.percentile(np.mean(rng.choice(v,size=(8000,len(v)),replace=True),axis=1),[2.5,97.5]).tolist()
summary=[];contrasts=[]
for c in conditions:
 means=[float(np.mean([r['action_change_in_correct_log_odds'] for r in sem if r['condition']==c and r['family']==f])) for f in families];summary.append(dict(condition=c,mean=float(np.mean(means)),family_means=means,descriptive_family_bootstrap95=interval(means)))
full=next(x for x in summary if x['condition']=='full')
for x in summary[1:]:
 v=np.array(full['family_means'])-np.array(x['family_means']);contrasts.append(dict(control=x['condition'],full_minus_control=float(v.mean()),family_differences=v.tolist(),descriptive_family_bootstrap95=interval(v)))
lookup={(r['scene'],r['copy'],r['condition']):r for r in sem};errors=[]
for r in sem:
 b=lookup[(r['scene'],r['copy'],'full')]['effective_action_norm'];errors.append(abs(r['effective_action_norm']-b)/max(b,1e-9))
common=[]
for p in C.glob('recurrent_common_*.json'):
 d=json.loads(p.read_text());common+=d['rows']
cs=[]
if len({r['scene'] for r in common})==16:
 for c in conditions[:3]:cs.append(dict(condition=c,prediction_kl=float(np.mean([r['prediction_kl'] for r in common if r['condition']==c])),processed_return_norm=float(np.mean([r['processed_return_norm'] for r in common if r['condition']==c]))))
result=dict(semantic_rows=sem,summary=summary,contrasts=contrasts,families=families,common_context=cs,maximum_relative_effective_norm_mismatch=max(errors),qualification=json.loads((R/'recurrent/inputs/recurrent_qualification.json').read_text()),
 interpretation='Opposite facts reverse the correct option within each family. Action-induced changes subtract the matched unchanged model. A fixed preference for one option is not sufficient for a positive family-paired effect.',
 scope='One action at the end of a fact-conditioned prefix, with later candidate tokens scored without additional actions. Magnitudes yoked to full reference. Eight authored families, not population-wide awareness evidence.')
p=O/'recurrent_analysis.json';q=p.with_suffix('.new');q.write_text(json.dumps(result,indent=2));q.replace(p)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafbf9','axes.facecolor':'#fafbf9'})
fig,ax=plt.subplots(1,2,figsize=(12,4))
for j,k in enumerate(['prediction_kl','processed_return_norm']):
 ax[j].bar(range(len(cs)),[x[k] for x in cs],color=['#b66731','#7398ae','#7fa385'][:len(cs)]);ax[j].set_xticks(range(len(cs)),names[:len(cs)],rotation=17,ha='right')
ax[0].set_ylabel('KL from unchanged next-token distribution');ax[1].set_ylabel('Change in activity returning at layer 24');ax[0].set_title('Same content writer, different control mechanism');plt.tight_layout()
for ext in ['png','svg']:plt.savefig(R/f'figures/26-recurrent-interface.{ext}',dpi=160,bbox_inches='tight')
plt.close();fig,ax=plt.subplots(1,2,figsize=(13,4))
means=np.array([r['mean'] for r in summary]);lo=np.array([r['descriptive_family_bootstrap95'][0] for r in summary]);hi=np.array([r['descriptive_family_bootstrap95'][1] for r in summary]);ax[0].bar(range(4),means,color=['#b66731','#7398ae','#7fa385','#95658d']);ax[0].errorbar(range(4),means,yerr=np.maximum(0,np.stack([means-lo,hi-means])),fmt='none',c='#344a52',capsize=4);ax[0].set_xticks(range(4),names,rotation=20,ha='right');ax[0].set_ylabel('Action-induced change in correct/incorrect log odds');ax[0].axhline(0,c='gray',lw=.8)
for r,name in zip(summary,names):ax[1].plot(families,r['family_means'],'-o',label=name,ms=3)
ax[1].axhline(0,c='gray',lw=.8);ax[1].set_xticks(range(8),families,rotation=20,ha='right');ax[1].set_ylabel('Change in fact-supported conclusion log odds');ax[1].legend(fontsize=8);ax[0].set_title('Does the action help use the changed fact?');plt.tight_layout()
for ext in ['png','svg']:plt.savefig(R/f'figures/27-fact-conditioned-action.{ext}',dpi=160,bbox_inches='tight')
print('Semantic summaries',json.dumps(summary));print('Paired contrasts',json.dumps(contrasts));print('Common',cs);print('Norm mismatch',max(errors))
