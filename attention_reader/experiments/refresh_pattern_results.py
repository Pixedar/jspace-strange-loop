"""Attach the completed narrow factual screen and readable comparison figures."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import pathlib,json,collections,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).parent;O=R/'results/extension';C=R/'specificity/results/cloud/58704070/results';p=O/'generation.json';d=json.loads(p.read_text());q=json.loads((C/'constraint_rater_qualification.json').read_text());assert q['passed'];ratings=[]
def clean(s):return ' '.join(s.strip().strip('"“”').split())
for f in sorted(C.glob('constraint_rating_*.json')):
 x=json.loads(f.read_text());source=json.loads((C/x['file']).read_text());x['condition']=source['condition'] if x['row']<4 else 'unchanged';x['family']=source['scene']['family'];v=x['verdict'];grounded=(v=='UNCLEAR' and x['evidence'].lower()=='none') or (bool(x['evidence'].strip()) and clean(x['evidence']) in clean(x['response']));assert grounded==x['grounded'];ratings.append(x)
assert len(ratings)==800;d['ratings']=ratings;d['rater_qualification']=q
counts={}
for c in sorted({r['condition'] for r in ratings}):counts[c]=dict(collections.Counter(r['verdict'] if r['valid'] and r['grounded'] else 'UNVALIDATED' for r in ratings if r['condition']==c))
d['fact_counts']=counts;temp=p.with_suffix('.new');temp.write_text(json.dumps(d));temp.replace(p)
names={'full':'Full controller','unused_schema':'Forecast unused','mean_memory':'Private average','random_act':'Fresh noise','rotated_act':'Fixed rotation','fixed_act':'Constant direction','unchanged':'Unchanged model'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafbf9','axes.facecolor':'#fafbf9'})
fig,ax=plt.subplots(1,2,figsize=(11,4));cs=['full','unused_schema','mean_memory','random_act','rotated_act','fixed_act'];xx=np.arange(6)
ax[0].bar(xx,[next(x['mean_similarity'] for x in d['summary'] if x['condition']==c) for c in cs],color='#799bac');ax[0].set_ylim(0,1);ax[0].set_ylabel('Sentence-embedding cosine with unchanged response');ax[0].set_title('Content changes are similar across action controls')
for j,mode in enumerate(['surface','embedding']):
 rr=[next(r for r in d['identity'] if r['condition']==c and r['representation']==mode) for c in cs];ax[1].bar(xx+(j-.5)*.32,[r['accuracy'] for r in rr],width=.32,label=['Surface form','Sentence embedding'][j]);ax[1].scatter(xx+(j-.5)*.32,[r['null95'] for r in rr],marker='_',s=80,c='#444444')
ax[1].axhline(.25,c='gray',ls='--',label='Chance');ax[1].set_ylim(0,.55);ax[1].set_ylabel('Copy recognition on an unseen family and sample');ax[1].legend(fontsize=8)
for a in ax:a.set_xticks(xx,[names[c] for c in cs],rotation=25,ha='right')
plt.tight_layout()
for ext in ['png','svg']:plt.savefig(R/f'figures/23-repeated-response-controls.{ext}',dpi=160,bbox_inches='tight')
plt.close();families=sorted({r['family'] for r in ratings});plt.figure(figsize=(10,4))
for j,c in enumerate(['unchanged','full','mean_memory','random_act']):
 values=[]
 for family in families:
  rr=[r for r in ratings if r['family']==family and r['condition']==c];values.append(sum(r['verdict']=='PRESERVED' and r['valid'] and r['grounded'] for r in rr)/len(rr))
 plt.plot(families,values,'-o',label=names[c])
plt.ylim(-.04,1.04);plt.ylabel('Responses with a grounded PRESERVED verdict / all responses');plt.xticks(rotation=20,ha='right');plt.legend(fontsize=8);plt.title('The factual screen distinguishes tasks, not overall answer quality');plt.tight_layout()
for ext in ['png','svg']:plt.savefig(R/f'figures/25-fact-preservation.{ext}',dpi=160,bbox_inches='tight')
print('qualification',q['correct_grounded'],'/',q['n']);print(json.dumps(counts));print('finished',len(ratings))
