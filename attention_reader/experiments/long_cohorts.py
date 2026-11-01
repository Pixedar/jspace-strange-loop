"""Finite, unattended local extension: 16 copies, six streams, ordinary controls."""
import os,pathlib,sys,json,time,hashlib,gc
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np,torch
R=pathlib.Path(__file__).parent;sys.path.insert(0,str(R/'src'));from reader import Reader,Life
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
OUT=R/'results/long';OUT.mkdir(exist_ok=True)
cohorts=[83,84];arms=['own','unused','neighbor','mean'];B=16
sources=[];warm=[];common=[];genres=[];ids=[]
for stream in range(1,7):
 p=pathlib.Path(f'C:/Users/assas/jspace_v11_runs/core32/states_8B_life_s{stream}.npz');z=np.load(p);h=z['h'];g=z['genre'];tok=z['ids']
 for row in range(4):warm.append(h[row,:256])
 common.append(h[0,256:3000]);genres.append(g[0,256:3000]);ids.append(tok[0,256:3000]);sources.append(dict(stream=stream,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),common_readings=[256,3000]));del h,z
pick=np.random.default_rng(773).choice(len(warm),B,replace=False);W=torch.tensor(np.stack(warm)[pick],device='cuda',dtype=torch.float32)
H=torch.tensor(np.concatenate(common),device='cuda',dtype=torch.float32);G=np.r_[np.full(256,-1),np.concatenate(genres)];TOK=np.concatenate(ids);N=256+len(H)
np.savez_compressed(OUT/'input_index.npz',genre=G,common_ids=TOK,warmup_indices=pick)
protocol=dict(copies=B,cohorts=cohorts,arms=arms,readings=N,warmup=256,streams=sources,stream_boundaries=[256+2744*i for i in range(6)],
 interpretation='Continuous controller state across six independently recorded LLM streams. Not a continuous LLM context. Same common activity for every condition after warm-up.',
 selection='Cohorts, lengths, controls and warm-up selection fixed before running. Unusual cases require replay and causal tests; no self-awareness endpoint is assumed.')
(OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
reader=Reader(R/'src/c8_reference.pt').cuda()
# A larger batch must preserve exact symmetry when histories are identical.
null=Life(reader,seed=83,copies=B);maxsym=0.
for t in range(512):
 hh=W[:1,t].repeat(B,1) if t<256 else H[t-256:t-255].repeat(B,1)
 rec=null.step(hh);maxsym=max(maxsym,float((rec['s']-rec['s'][:1]).abs().max()))
assert maxsym==0
(OUT/'symmetry_qualification.json').write_text(json.dumps(dict(copies=B,readings=512,max_abs_stance_difference=maxsym,passed=True)))
del null
start=time.time()
for seed in cohorts:
 for arm in arms:
    dest=OUT/f'{arm}_{seed}';dest.mkdir(exist_ok=True)
    if (dest/'DONE').exists():continue
    reader=Reader(R/'src/c8_reference.pt').cuda();life=Life(reader,seed=seed,copies=B,schema='none' if arm=='unused' else ('neighbor' if arm=='neighbor' else 'own'))
    total=torch.zeros(B,256,device='cuda');weight=0.;mean_scale=None;last={};base_read=reader.read
    def read(s,hist,habit,extra):
        global mean_scale
        if arm=='mean':
            extra=None
            if life.t>=300:
                pull=total/max(weight,1e-12)
                if mean_scale is None:mean_scale=float(.5*life.cap/pull.norm(dim=-1).mean().clamp_min(1e-12))
                extra=(mean_scale*pull).view(B,4,64)
        out=base_read(s,hist,habit,extra);last['att'],last['keys']=out[2],out[3];return out
    reader.read=read
    trace={k:np.zeros((N,B,d),np.float32) for k,d in [('s',8),('g',8),('picture',32),('att',16)]}
    scalar={k:np.zeros((N,B),np.float32) for k in ['schema_error','schema_total','mean_error','attention_entropy','attention_recency','pull_norm']}
    blocks=torch.zeros((N+99)//100,B,4096,device='cuda')
    for t in range(N):
        hh=W[:,t] if t<256 else H[t-256:t-255].expand(B,-1)
        rec=life.step(hh);a=rec['att'];l=a.shape[-1]
        for key in ['s','g','picture']:trace[key][t]=rec[key].cpu().numpy()
        trace['att'][t,:,-l:]=a.cpu().numpy()
        for key in ['schema_error','schema_total']:scalar[key][t]=rec[key].cpu().numpy()
        target=(torch.einsum('nhl,nlhd->nhd',last['att'],last['keys'])-last['keys'].mean(1)).reshape(B,256)
        meanpred=total/max(weight,1e-12);scalar['mean_error'][t]=(target-meanpred).square().sum(-1).cpu().numpy()
        scalar['pull_norm'][t]=target.norm(dim=-1).cpu().numpy()
        scalar['attention_entropy'][t]=(-(a*(a+1e-12).log()).sum(-1)).cpu().numpy()
        scalar['attention_recency'][t]=(a*torch.linspace(0,1,l,device='cuda')).sum(-1).cpu().numpy()
        total=.998*total+target;weight=.998*weight+1
        attended=torch.einsum('bl,bld->bd',a,life.hist);center=(attended-attended.mean(0,keepdim=True))/attended.square().mean(-1,keepdim=True).sqrt();blocks[t//100]+=center/100
        if (t+1)%1000==0 or t+1==N:
            state=life.snapshot();state['ordinary_mean']=dict(total=total.cpu(),weight=weight,scale=mean_scale,active=arm=='mean')
            p=dest/f'state_{t+1}.pt';torch.save(state,p.with_suffix('.partial'));p.with_suffix('.partial').replace(p)
            (OUT/'progress.json').write_text(json.dumps(dict(seed=seed,arm=arm,t=t+1,total=N,elapsed=time.time()-start)))
            print(seed,arm,t+1,'/',N,round(time.time()-start),flush=True)
    np.savez_compressed(dest/'trace.npz',**trace,**scalar,profiles=blocks.half().cpu().numpy())
    (dest/'DONE').write_text(str(time.time()));del trace,scalar,blocks,life,reader;gc.collect();torch.cuda.empty_cache()
(OUT/'DONE').write_text('all finite cohorts completed')
