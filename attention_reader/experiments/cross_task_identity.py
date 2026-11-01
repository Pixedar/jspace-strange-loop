"""Does a text-only signature identify the same copy on an unseen task?
Nearest-centroid classifier, fixed protocol; permutation labels within each task.
Exploratory on already examined prompts, not a preregistered personality test.
"""
import pathlib,json,numpy as np,hashlib
R=pathlib.Path(__file__).parent;b=json.loads((R/'results/behavior_analysis.json').read_text())
emb=np.load(R/'results/text_embeddings.npz');lookup=dict(zip(emb['text_sha256'],emb['embeddings']))
records=b['records'];metrics={(r['task'],r['life'],r['condition'],r['row']):r['metrics'] for r in b['mechanical']}
tasks=sorted({r['prompt']['id'] for r in records});rows=[]
features=[k for k in next(iter(metrics.values())) if k!='baseline_token_agreement']
def accuracy(X,labels):
    # Shape tasks x copies x dimensions; task centering is label-free.
    pred=[];truth=[]
    for held in range(len(X)):
        train=np.delete(X,held,0);lab=np.delete(labels,held,0);sd=train.reshape(-1,train.shape[-1]).std(0).clip(1e-4)
        means=np.stack([train[lab==c].mean(0)/sd for c in range(4)])
        xx=X[held]/sd
        dist=((xx[:,None]-means[None])**2).mean(-1)
        pred.extend(dist.argmin(1).tolist());truth.extend(labels[held].tolist())
    return float(np.mean(np.array(pred)==truth))
for life in ['10','12']:
 for cond in sorted({r['condition'] for r in records}):
    group=[next((r for r in records if r['prompt']['id']==task and r['life']==life and r['condition']==cond),None) for task in tasks]
    if not all(group):continue
    for mode in ['surface','embedding']:
        X=[]
        for r in group:
            if mode=='surface':X.append([[metrics[(r['prompt']['id'],life,cond,c)][k] for k in features] for c in range(4)])
            else:X.append([lookup[hashlib.sha256(t.encode()).hexdigest()] for t in r['texts'][:4]])
        X=np.array(X,float);X-=X.mean(1,keepdims=True)
        # Standardize measurement scales once from all tasks without labels; each fold applies its training scaling too.
        labels=np.tile(np.arange(4),(len(X),1));actual=accuracy(X,labels)
        rng=np.random.default_rng(941);null=[]
        for i in range(400):null.append(accuracy(X,np.stack([rng.permutation(4) for _ in tasks])))
        row=dict(life=life,condition=cond,representation=mode,accuracy=actual,chance=.25,
             permutation_p=(1+sum(x>=actual for x in null))/(1+len(null)),null95=float(np.quantile(null,.95)),
             tasks=len(X),heldout_responses=4*len(X),features=len(features) if mode=='surface' else X.shape[-1])
        rows.append(row);print(json.dumps(row),flush=True)
(R/'results/cross_task_identity.json').write_text(json.dumps(dict(rows=rows,
 protocol='leave one complete task out; center representations within task across copies; nearest training centroid; 400 independent within-task copy-label shuffles',
 caveats=['two selected histories and one sampling stream','exploratory after seeing outputs','correlated representations and conditions; no multiple-comparison discovery claim','copy recognition is not a personality or consciousness measurement','surface features include form, not only meaning','embedding representation is an imperfect pretrained similarity space']),indent=2))
