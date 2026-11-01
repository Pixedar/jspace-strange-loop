"""Repeated-sample behavior controls; no broad psychological ratings."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['TOKENIZERS_PARALLELISM']='false'
import pathlib,json,re,ast,hashlib,numpy as np,torch
from transformers import AutoTokenizer,AutoModel
R=pathlib.Path(__file__).parent;C=R/'specificity/results/cloud/58704070/results';O=R/'results/extension';O.mkdir(exist_ok=True)
def save(p,x):
 q=p.with_suffix('.new');q.write_text(json.dumps(x),encoding='utf-8');q.replace(p)
files=sorted(C.glob('pattern_generation_*.json'));assert len(files)==192,f'Need all balanced cells: {len(files)}/192'
# Reuse the exact earlier mechanical definitions without executing its analysis.
tree=ast.parse((R/'analyze_behavior.py').read_text(encoding='utf-8'));nodes=[n for n in tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='lex' for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name=='metrics')]
scope=dict(re=re,np=np);exec(compile(ast.Module(body=nodes,type_ignores=[]),'mechanical_definitions','exec'),scope);metrics=scope['metrics']
records=[];by={};norm_errors=[];base_errors=[]
for p in files:
 d=json.loads(p.read_text());seqs=[s[:s.index(151645)] if 151645 in s else s for s in d['token_ids']];d['token_ids']=seqs
 d['file']=p.name;by[(d['scene']['id'],d['seed'],d['condition'])]=d
 records.append(d)
texts=list(dict.fromkeys(t for d in records for t in d['texts']));torch.set_num_threads(2)
tok=AutoTokenizer.from_pretrained(R/'cache/minilm-local',local_files_only=True);model=AutoModel.from_pretrained(R/'cache/minilm-local',local_files_only=True).eval();vectors=[]
with torch.inference_mode():
 for i in range(0,len(texts),24):
  enc=tok(texts[i:i+24],return_tensors='pt',padding=True,truncation=True,max_length=256);y=model(**enc).last_hidden_state;m=enc.attention_mask[:,:,None];v=(y*m).sum(1)/m.sum(1);v/=v.norm(dim=-1,keepdim=True);vectors.extend(v.numpy())
  print('embedded',min(i+24,len(texts)),len(texts),flush=True)
vec=dict(zip(texts,vectors));flat=[]
for d in records:
 full=by[(d['scene']['id'],d['seed'],'full')];base_errors.append(d['token_ids'][4]!=full['token_ids'][4]);actual=np.array(d['action_norms']);target=np.array(full['action_norms']);valid=np.arange(len(actual))[:,None] < (np.array([len(s) for s in d['token_ids'][:4]])+d['prompt_tokens'])[None,:]
 mask=valid & (target>1e-6);norm_errors.extend((abs(actual-target)/target.clip(1e-9))[mask].tolist())
 for c in range(4):
  m=metrics(d['texts'][c],d['token_ids'][c],d['token_ids'][4]);m.pop('baseline_token_agreement',None)
  flat.append(dict(scene=d['scene']['id'],family=d['scene']['family'],seed=d['seed'],condition=d['condition'],copy=c,similarity_to_baseline=float(vec[d['texts'][c]]@vec[d['texts'][4]]),metrics=m))
 assert len(d['trace'])==np.load(C/d['file'].replace('pattern_generation','pattern_J24').replace('.json','.npz'))['J24'].shape[0]
families=sorted({d['scene']['family'] for d in records});conditions=sorted({d['condition'] for d in records});features=sorted(flat[0]['metrics']);identity=[];lookup={(x['scene'],x['seed'],x['condition'],x['copy']):x for x in flat}
def accuracy(X,labels):
 correct=[]
 for held in range(8):
  for trainseed,testseed in [(0,1),(1,0)]:
   mask=np.arange(8)!=held;train=X[mask,:,trainseed];lab=labels[mask];sd=train.reshape(-1,X.shape[-1]).std(0).clip(1e-4)
   centers=np.stack([train[np.broadcast_to(lab[:,None,:],train.shape[:-1])==c].mean(0)/sd for c in range(4)])
   test=X[held,:,testseed]/sd;pred=((test[...,None,:]-centers)**2).mean(-1).argmin(-1);truth=np.broadcast_to(labels[held],pred.shape);correct.extend((pred==truth).ravel().tolist())
 return float(np.mean(correct))
for cond in conditions:
 for mode in ['surface','embedding']:
  X=[]
  for family in families:
   scenes=sorted({r['scene']['id'] for r in records if r['scene']['family']==family});xx=[]
   for scene in scenes:
    ss=[]
    for seed in [202,303]:
     dd=by[(scene,seed,cond)];ss.append([([lookup[(scene,seed,cond,c)]['metrics'][k] for k in features] if mode=='surface' else vec[dd['texts'][c]]) for c in range(4)])
    xx.append(ss)
   X.append(xx)
  X=np.array(X,float);X-=X.mean(3,keepdims=True);labels=np.tile(np.arange(4),(8,1));actual=accuracy(X,labels);rng=np.random.default_rng(393);null=[accuracy(X,np.stack([rng.permutation(4) for _ in families])) for _ in range(400)]
  identity.append(dict(condition=cond,representation=mode,accuracy=actual,permutation_p=(1+sum(a>=actual for a in null))/401,null95=float(np.quantile(null,.95)),responses=128,chance=.25))
  print(identity[-1],flush=True)
rating=[];qual=None
if (C/'constraint_rater_qualification.json').exists():
 qual=json.loads((C/'constraint_rater_qualification.json').read_text())
 if qual['passed']:
  for p in sorted(C.glob('constraint_rating_*.json')):
   x=json.loads(p.read_text());x['condition']=json.loads((C/x['file']).read_text())['condition'] if x['row']<4 else 'unchanged';rating.append(x)
summary=[]
for c in conditions:
 rr=[r for r in flat if r['condition']==c];summary.append(dict(condition=c,mean_similarity=float(np.mean([r['similarity_to_baseline'] for r in rr])),mean_surface_difference={k:float(np.mean([r['metrics'][k]-metrics(by[(r['scene'],r['seed'],c)]['texts'][4],[],[])[k] for r in rr])) for k in features}))
out=dict(batches=192,controller_outputs=768,unique_unchanged=32,authored_scenes=16,families=8,sampling_seeds=[202,303],selected_source_copies=[0,11,7,5],baseline_mismatches=sum(base_errors),effective_norm_matching=dict(max_relative_error=max(norm_errors),median_relative_error=float(np.median(norm_errors)),p99_relative_error=float(np.quantile(norm_errors,.99))),identity=identity,summary=summary,ratings=rating,rater_qualification=qual,
 identity_protocol='Leave whole scene family out and train on one sampling stream, test the other; both seed directions. Center within scene/seed over copies. Nearest training centroid, training-fold scaling. 400 family-specific copy permutations shared by both variants and seeds.',
 interpretation='Stable copy recognition can also be produced by fixed private directions; compare controls before treating it as a self-model property.',
 records=[{k:v for k,v in d.items() if k not in ['trace','action_norms']} for d in records],mechanical=flat)
save(O/'generation.json',out)
# Fixed axes per scene and module, shared by every condition; valid response frames only.
paths={}
for scene in sorted({d['scene']['id'] for d in records}):
 for seed in [202,303]:
  group=[d for d in records if d['scene']['id']==scene and d['seed']==seed];data={}
  for d in group:
   trace=d['trace'];t=np.array([a['t'] for a in trace]);J=np.load(C/d['file'].replace('pattern_generation','pattern_J24').replace('.json','.npz'))['J24'];lengths=np.array([len(s) for s in d['token_ids']]);valid=t[:,None]<d['prompt_tokens']+lengths[None,:]
   data[d['condition']]=dict(t=t,valid=valid,J=J,stance=np.array([a['s'] for a in trace]),reading=np.array([a['picture'] for a in trace]))
  projected={};variance={}
  for key in ['J','stance','reading']:
   points=np.concatenate([v[key][v['valid'][:,:v[key].shape[1]]] for v in data.values()]).astype(np.float64);mean=points.mean(0);_,sv,V=np.linalg.svd(points-mean,full_matrices=False);basis=V[:3].T;variance[key]=float((sv[:3]**2).sum()/(sv**2).sum());scale=np.std((points-mean)@basis,axis=0).max()
   for c,v in data.items():projected.setdefault(c,{})[key]=np.round(((v[key]-mean)@basis)/max(scale,1e-8),5).transpose(1,0,2).tolist()
  for c,v in data.items():projected[c]['positions']=v['t'].tolist();projected[c]['valid']=v['valid'].T.tolist()
  paths[scene+'__'+str(seed)]=dict(conditions=projected,variance=variance,prompt_tokens=group[0]['prompt_tokens'])
save(O/'live_paths.json',paths)
np.savez_compressed(O/'text_embeddings.npz',embeddings=np.stack(vectors),text_sha256=np.array([hashlib.sha256(t.encode()).hexdigest() for t in texts]))
print('All analysis complete',out['effective_norm_matching'])
