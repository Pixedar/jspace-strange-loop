"""Text comparison for matched full and recurrent controls; no inferred personalities."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['TOKENIZERS_PARALLELISM']='false'
import pathlib,json,ast,re,numpy as np,torch
from transformers import AutoTokenizer,AutoModel
R=pathlib.Path(__file__).parent;C=R/'recurrent/results/cloud/58714365/results';O=R/'results/extension';files=sorted(C.glob('recurrent_generation_*.json'));assert len(files)==96
tree=ast.parse((R/'analyze_behavior.py').read_text(encoding='utf-8'));nodes=[n for n in tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='lex' for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name=='metrics')];scope=dict(np=np,re=re);exec(compile(ast.Module(body=nodes,type_ignores=[]),'mechanical_definitions','exec'),scope);metrics=scope['metrics'];records=[]
for p in files:
 d=json.loads(p.read_text());d['file']=p.name;d['token_ids']=[s[:s.index(151645)] if 151645 in s else s for s in d['token_ids']];records.append(d)
texts=list(dict.fromkeys(t for r in records for t in r['texts']));torch.set_num_threads(2);tok=AutoTokenizer.from_pretrained(R/'cache/minilm-local',local_files_only=True);model=AutoModel.from_pretrained(R/'cache/minilm-local',local_files_only=True).eval();vectors=[]
with torch.inference_mode():
 for i in range(0,len(texts),24):
  e=tok(texts[i:i+24],return_tensors='pt',padding=True,truncation=True,max_length=256);h=model(**e).last_hidden_state;m=e.attention_mask[:,:,None];v=(h*m).sum(1)/m.sum(1);v/=v.norm(dim=-1,keepdim=True);vectors.extend(v.numpy())
vec=dict(zip(texts,vectors));summary=[];norm=[];baselines=[]
for cond in ['full','recurrent_forecast','recurrent_selector']:
 similarities=[]
 for d in [r for r in records if r['condition']==cond]:
  full=next(r for r in records if r['condition']=='full' and r['scene']['id']==d['scene']['id'] and r['seed']==d['seed']);baselines.append(d['token_ids'][4]==full['token_ids'][4]);fs={x['t']:np.array(x['actual']) for x in full['action_norms']}
  for x in d['action_norms']:
   valid=x['t']<d['prompt_tokens']+np.array([len(s) for s in d['token_ids'][:4]]);b=fs[x['t']];norm.extend((abs(np.array(x['actual'])-b)/b.clip(1e-9))[valid&(b>1e-8)].tolist())
  d['mechanical']=[metrics(t,d['token_ids'][c],d['token_ids'][4]) for c,t in enumerate(d['texts'])];similarities.extend([float(vec[t]@vec[d['texts'][4]]) for t in d['texts'][:4]])
 summary.append(dict(condition=cond,mean_sentence_similarity_to_unchanged=float(np.mean(similarities)),controller_outputs=128))
out=dict(records=[{k:v for k,v in r.items() if k not in ['trace','action_norms','token_ids']} for r in records],summary=summary,batches=96,controller_outputs=384,unique_unchanged=32,baseline_checks=sum(baselines),baseline_comparisons=len(baselines),maximum_relative_effective_norm_mismatch=max(norm),median_relative_effective_norm_mismatch=float(np.median(norm)),scope='Same thirteen-row generation with all three conditions. Two sampling streams and sixteen authored situations; source copies are dependent. Sentence similarity and surface measures do not validate a self-related semantic trait.')
p=O/'recurrent_generation.json';q=p.with_suffix('.new');q.write_text(json.dumps(out));q.replace(p);print(json.dumps(summary));print('Norm mismatch',max(norm),'baseline',sum(baselines),'/',len(baselines))
