"""Counterbalanced, fact-paired information use; retain every qualification flag."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import pathlib,json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).parent;C=R/'recurrent/results/cloud/58714365/results';O=R/'results/extension';files=list(C.glob('choice_*_[01].json'));assert len(files)==64,len(files)
records=[json.loads(p.read_text()) for p in files];assert all(r.get('prefix_mode','').startswith('native bulk') for r in records),'Do not mix serial-prefix attempt with the final assay'
conditions=['full','recurrent_forecast','recurrent_selector','neighbor_forecast'];families=sorted({r['scene']['family'] for r in records});rng=np.random.default_rng(8882);rows=[];contrasts=[]
def ci(v):return np.percentile(np.mean(rng.choice(v,size=(8000,len(v)),replace=True),axis=1),[2.5,97.5]).tolist()
for wording in ['original','paraphrase','combined']:
 for c in conditions:
  group=[r for r in records if wording=='combined' or r['scene']['wording']==wording];means=[];qualified=[]
  for family in families:
   rr=[r for r in group if r['scene']['family']==family];values=[x for r in rr for x in r['rows'] if x['condition']==c];means.append(float(np.mean([x['action_change_in_correct_log_odds'] for x in values])));qualified.append(all(r['baseline_format_qualified'] for r in rr) and all(x['answer_format_probability']>=.5 for x in values))
  values=[x for r in group for x in r['rows'] if x['condition']==c];rows.append(dict(wording=wording,condition=c,mean=float(np.mean(means)),family_means=means,qualified_families=qualified,descriptive_family_bootstrap95=ci(means),conditional_choice_accuracy=float(np.mean([x['correct_label_log_odds']>0 for x in values])),baseline_conditional_choice_accuracy=float(np.mean([x['baseline_correct_label_log_odds']>0 for x in values])),wrong_to_correct=sum(x['correct_label_log_odds']>0>=x['baseline_correct_label_log_odds'] for x in values),correct_to_wrong=sum(x['correct_label_log_odds']<=0<x['baseline_correct_label_log_odds'] for x in values),reinforces_existing_label_margin=float(np.mean([np.sign(x['baseline_correct_label_log_odds'])*x['action_change_in_correct_log_odds'] for x in values]))))
 for c in conditions[1:]:
  a=next(r for r in rows if r['wording']==wording and r['condition']=='full');b=next(r for r in rows if r['wording']==wording and r['condition']==c);v=np.array(a['family_means'])-np.array(b['family_means']);contrasts.append(dict(wording=wording,control=c,full_minus_control=float(v.mean()),family_differences=v.tolist(),descriptive_family_bootstrap95=ci(v)))
norm=[]
for r in records:
 for x in r['rows']:
  b=next(a for a in r['rows'] if a['condition']=='full' and a['copy']==x['copy'])['effective_action_norm'];norm.append(abs(x['effective_action_norm']-b)/max(b,1e-9))
result=dict(records=records,rows=rows,contrasts=contrasts,families=families,baseline_format_qualified=sum(r['baseline_format_qualified'] for r in records),contexts=64,maximum_relative_effective_norm_mismatch=max(norm),protocol=json.loads((C/'choice_protocol.json').read_text()),
 limits=['Two label orders and four copies are nested, not independent replications.','Paraphrases test wording stability of the same eight fact families.','One final-token action is distinct from continuing actions during free generation.','Only single-label task behavior is measured; no self-awareness score.'])
p=O/'information_choice.json';q=p.with_suffix('.new');q.write_text(json.dumps(result,indent=2));q.replace(p)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafbf9','axes.facecolor':'#fafbf9'});names=['Attention model','Recurrent forecast','Recurrent selector','Other-copy forecast'];colors=['#b66731','#7398ae','#7fa385','#95658d'];fig,ax=plt.subplots(1,2,figsize=(12,4))
for j,wording in enumerate(['original','paraphrase']):
 v=[next(r for r in rows if r['wording']==wording and r['condition']==c) for c in conditions];pos=np.arange(4)+(j-.5)*.3;ax[0].bar(pos,[r['mean'] for r in v],width=.3,label=['Original wording','New wording'][j]);low=np.array([r['descriptive_family_bootstrap95'][0] for r in v]);high=np.array([r['descriptive_family_bootstrap95'][1] for r in v]);m=np.array([r['mean'] for r in v]);ax[0].errorbar(pos,m,yerr=np.maximum(0,np.array([m-low,high-m])),fmt='none',c='#344a52',capsize=3)
ax[0].set_xticks(range(4),names,rotation=20,ha='right');ax[0].set_ylabel('Change in supported-answer label log odds');ax[0].legend(fontsize=8)
for c,name,color in zip(conditions,names,colors):
 v=next(r for r in rows if r['condition']==c and r['wording']=='combined');ax[1].plot(families,v['family_means'],'-o',label=name,c=color,ms=3)
for a in ax:a.axhline(0,c='gray',lw=.8)
ax[1].set_xticks(range(8),families,rotation=20,ha='right');ax[1].set_ylabel('Change after averaging opposite facts and label orders');ax[1].legend(fontsize=8);ax[0].set_title('Does the action help use the supplied fact?');plt.tight_layout()
for ext in ['png','svg']:plt.savefig(R/f'figures/28-counterbalanced-information.{ext}',dpi=160,bbox_inches='tight')
print('Format-qualified contexts',result['baseline_format_qualified'],'/64');print('Rows',json.dumps([{k:v for k,v in x.items() if k not in ['family_means','qualified_families']} for x in rows]));print('Contrasts',json.dumps(contrasts));print('Norm mismatch',max(norm))
