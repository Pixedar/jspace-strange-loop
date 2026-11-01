"""New, bounded full-rank native bridge fit; validation chooses regularization.
This is an exploratory requalification on the prior test stream, not fresh evidence.
"""
import pathlib,sys,json,gc,copy,time
import numpy as np,torch
from torch import nn
R=pathlib.Path(__file__).parent;sys.path.insert(0,str(R/'src'))
from reader import Reader,Life
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
B=R.parent/'results/cloud/58674995/results'
H8=np.load(B/'native8_observations.npz')['h'].astype(np.float32)
original=Reader(R/'src/c8_reference.pt').cuda()
with torch.inference_mode():
    y=[]
    for h in H8:
        a=original.rln(torch.tensor(h,device='cuda'));y.append(torch.cat([original.rk(a),original.rv(a)],-1).cpu().numpy())
Y=np.stack(y)
reports=[]
for label,file in [('14B','native14_observations.npz'),('Omni30B','omni_observations.npz')]:
    H=np.load(B/file)['h'].astype(np.float32)
    # Use all four actual early histories, not four copies of row 0.
    X=torch.tensor(H[:,:1000].reshape(-1,H.shape[-1]),device='cuda');T=torch.tensor(Y[:,:1000].reshape(-1,512),device='cuda')
    mean=X.mean(0);sd=X.std(0).clamp_min(.1);ym=T.mean(0);A=(X-mean)/sd
    gram=A@A.T;scale=float(gram.diag().mean());eye=torch.eye(len(X),device='cuda')
    xv=torch.tensor(H[:,1000:1500].reshape(-1,H.shape[-1]),device='cuda');yv=torch.tensor(Y[:,1000:1500].reshape(-1,512),device='cuda')
    candidates=[];best=None
    for alpha in [.001,.01,.1]:
        W=A.T@torch.linalg.solve(gram+alpha*scale*eye,T-ym)
        pred=((xv-mean)/sd)@W+ym;r2=float(1-(pred-yv).square().sum()/(yv-ym).square().sum())
        candidates.append(dict(alpha=alpha,validation_r2=r2));print(label,'alpha',alpha,'validation',r2,flush=True)
        if best is None or r2>best[0]:best=(r2,alpha,W.clone())
    W=best[2]/sd[:,None];bias=ym-mean@W
    mapped=Reader(R/'src/c8_reference.pt').cuda();mapped.d=H.shape[-1];mapped.rln=nn.Identity();mapped.rk=nn.Linear(mapped.d,256).cuda();mapped.rv=nn.Linear(mapped.d,256).cuda()
    with torch.no_grad():
        mapped.rk.weight.copy_(W[:,:256].T);mapped.rk.bias.copy_(bias[:256]);mapped.rv.weight.copy_(W[:,256:].T);mapped.rv.bias.copy_(bias[256:])
    mapped.eval().requires_grad_(False)
    with torch.inference_mode():
        xt=torch.tensor(H[0,1500:3000],device='cuda');yt=torch.tensor(Y[0,1500:3000],device='cuda')
        p=xt@W+bias;r2=float(1-(p-yt).square().sum()/(yt-ym).square().sum())
        s=torch.randn(4,8,generator=torch.Generator().manual_seed(881)).cuda();s/=s.norm(dim=-1,keepdim=True);tv=[];pc=[];dc=[]
        for t in range(1500,3000,32):
            a=torch.tensor(H8[0,t-15:t+1],device='cuda')[None].repeat(4,1,1);b=torch.tensor(H[0,t-15:t+1],device='cuda')[None].repeat(4,1,1)
            p,g,att,_=original.read(s,a,None,None);q,h,other,_=mapped.read(s,b,None,None)
            tv.extend((.5*(att-other).abs().sum(-1).mean(-1)).cpu().tolist());pc.extend(torch.nn.functional.cosine_similarity(p,q).cpu().tolist());dc.extend(torch.nn.functional.cosine_similarity(g,h).cpu().tolist())
    passed=r2>=.8 and np.mean(tv)<=.15 and min(np.mean(pc),np.mean(dc))>=.9
    report=dict(model=label,training='all four actual histories, readings 0–999',validation='all four histories, readings 1000–1499',
      test='copy 0 readings 1500–2999; previously used for the older bridge, so this is exploratory',
      candidates=candidates,chosen_alpha=best[1],held_out_r2=r2,attention_tv=float(np.mean(tv)),picture_cosine=float(np.mean(pc)),destination_cosine=float(np.mean(dc)),passed=bool(passed),
      thresholds=dict(input_r2=.8,attention_tv=.15,picture_cosine=.9,destination_cosine=.9),
      interpretation='full-rank ridge map, standardized inputs, more early histories; no base-model training; no claim of multimodal behavior from text transfer')
    torch.save(dict(weight=W.cpu(),bias=bias.cpu(),report=report),R/f'results/native_bridge_{label}.pt')
    reports.append(report);(R/'results/native_bridge_revision.json').write_text(json.dumps(reports,indent=2))
    print(json.dumps(report),flush=True)
    if passed:
        # Only qualified interfaces enter an actual new recorded population.
        H[:,256:]=H[:1,256:];hh=torch.tensor(H[:,:3000],device='cuda');rows=[]
        for seed in range(40,46):
          for arm in ['own','none','neighbor']:
            life=Life(mapped,seed=seed,schema=arm);S=[];selected=[]
            for t in range(3000):
                rec=life.step(hh[:,t]);S.append(rec['s'].cpu().numpy())
                # Compare selected key/value content in common 512-dimensional space.
                kval=torch.cat([mapped.rk(life.hist),mapped.rv(life.hist)],-1)
                selected.append(torch.einsum('bl,bld->bd',rec['att'],kval).cpu().numpy())
            P=np.stack(selected).reshape(30,100,4,512).mean(1);V=P/(np.linalg.norm(P,axis=-1,keepdims=True)+1e-9)
            own=[];other=[]
            for a in range(3,30):
             for b in range(a+3,30):
              own.extend(np.sum(V[a]*V[b],axis=-1));other.extend([V[a,i]@V[b,j] for i in range(4) for j in range(4) if i!=j])
            score=float(np.mean(own)-np.mean(other));rows.append(dict(seed=seed,arm=arm,persistence=score))
            np.savez_compressed(R/f'results/native_{label}_{seed}_{arm}.npz',s=np.stack(S),selected_blocks=P)
            print('population',label,seed,arm,score,flush=True)
        (R/f'results/native_population_{label}.json').write_text(json.dumps(rows,indent=2))
    del X,T,A,gram,eye,xv,yv,W,mapped,H,best;gc.collect();torch.cuda.empty_cache()
