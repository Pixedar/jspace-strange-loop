"""Independent sentence-embedding and task-specific textual checks.
Embedding closeness is a semantic-similarity proxy, not semantic equivalence.
"""
import os,pathlib,json,re,hashlib
R=pathlib.Path(__file__).parent
os.environ['HF_HOME']=str(R/'cache');os.environ['TOKENIZERS_PARALLELISM']='false'
import requests,numpy as np,torch
from transformers import AutoTokenizer,AutoModel
torch.set_num_threads(2)
name='sentence-transformers/all-MiniLM-L6-v2';rev=requests.get('https://huggingface.co/api/models/'+name,timeout=30).json()['sha']
folder=R/'cache/minilm-local';folder.mkdir(parents=True,exist_ok=True)
for file in ['config.json','tokenizer.json','tokenizer_config.json','vocab.txt','special_tokens_map.json','model.safetensors']:
    p=folder/file
    if p.exists():continue
    print('download',file,flush=True)
    res=requests.get(f'https://huggingface.co/{name}/resolve/{rev}/{file}',timeout=(15,30),stream=True);res.raise_for_status()
    part=p.with_suffix('.partial')
    with part.open('wb') as f:
        for chunk in res.iter_content(1024*1024):f.write(chunk)
    part.replace(p)
tok=AutoTokenizer.from_pretrained(folder,local_files_only=True);model=AutoModel.from_pretrained(folder,local_files_only=True).eval()
b=json.loads((R/'results/behavior_analysis.json').read_text());texts=list(dict.fromkeys(t for r in b['records'] for t in r['texts']));vectors=[]
with torch.inference_mode():
 for a in range(0,len(texts),24):
    enc=tok(texts[a:a+24],return_tensors='pt',padding=True,truncation=True,max_length=256)
    y=model(**enc).last_hidden_state;mask=enc.attention_mask[:,:,None];v=(y*mask).sum(1)/mask.sum(1);v/=v.norm(dim=-1,keepdim=True);vectors.extend(v.numpy())
    print('embedded',min(a+24,len(texts)),len(texts),flush=True)
vec=dict(zip(texts,vectors));rows=[];taskchecks=[]
for r in b['records']:
 for row in range(4):
    rows.append(dict(task=r['prompt']['id'],life=r['life'],condition=r['condition'],row=row,cosine=float(vec[r['texts'][row]]@vec[r['texts'][4]])))
 if r['prompt']['id']=='revision':
  for row,t in enumerate(r['texts']):
    # Inspect only the rewritten claim, before the explanation (which may quote errors).
    head=re.split(r'(?:Most Important Change|Explanation|Important Change)',t,flags=re.I)[0]
    universal=bool(re.search(r'\ball (?:drinking )?water\b',head,re.I))
    no_checks=bool(re.search(r'no (?:further|other|additional) checks|no further testing|no additional testing',head,re.I))
    taskchecks.append(dict(life=r['life'],condition=r['condition'],row=row,universal_claim_retained=universal,no_checks_retained=no_checks,rewritten_portion=head))
summaries=[]
for task in sorted({r['task'] for r in rows}):
 for cond in sorted({r['condition'] for r in rows}):
    vals=[r['cosine'] for r in rows if r['task']==task and r['condition']==cond]
    if vals:summaries.append(dict(task=task,condition=cond,mean_cosine=float(np.mean(vals)),min_cosine=float(min(vals)),max_cosine=float(max(vals))))
out=dict(model=name,revision=rev,max_tokens=256,rows=rows,summary=summaries,revision_task_checks=taskchecks,
    caution='embedding cosine describes one pretrained similarity space; shared content can coexist with consequential differences in decisions or factual errors; textual checks require inspection of negation and context')
(R/'results/semantic_comparison.json').write_text(json.dumps(out,indent=2))
np.savez_compressed(R/'results/text_embeddings.npz',embeddings=np.stack(vectors),text_sha256=np.array([hashlib.sha256(t.encode()).hexdigest() for t in texts]))
print('full task similarity',[(r['task'],round(r['mean_cosine'],3)) for r in summaries if r['condition']=='full'])
print('revision checks',sum(r['universal_claim_retained'] for r in taskchecks if r['row']<4),'/',sum(r['row']<4 for r in taskchecks))
