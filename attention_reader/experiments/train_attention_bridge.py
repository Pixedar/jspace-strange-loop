"""Fine tune the key map against the actual attention operation.
No base weights change; choose checkpoint on validation, evaluate the old test once.
"""
import pathlib,sys,json,copy,gc
import numpy as np,torch
from torch import nn
import torch.nn.functional as F
R=pathlib.Path(__file__).parent;sys.path.insert(0,str(R/'src'))
from reader import Reader,Life
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
base=R.parent/'results/cloud/58674995/results'
reader=Reader(R/'src/c8_reference.pt').cuda();H8=np.load(base/'native8_observations.npz')['h'].astype(np.float32)
reports=[]
for label,file in [('14B','native14_observations.npz'),('Omni30B','omni_observations.npz')]:
    H=np.load(base/file)['h'].astype(np.float32);ck=torch.load(R/f'results/native_bridge_{label}.pt',map_location='cuda',weights_only=False)
    X=torch.tensor(H,device='cuda');Y=torch.tensor(H8,device='cuda')
    with torch.inference_mode():K=reader.rk(reader.rln(Y)).detach().clone()
    mean=X[:,:1000].mean((0,1));sd=X[:,:1000].std((0,1)).clamp_min(.1);Xn=(X-mean)/sd
    key=nn.Linear(X.shape[-1],256).cuda()
    with torch.no_grad():key.weight.copy_((sd[:,None]*ck['weight'][:,:256]).T);key.bias.copy_(mean@ck['weight'][:,:256]+ck['bias'][:256])
    optimizer=torch.optim.AdamW(key.parameters(),lr=2e-5,weight_decay=.01)
    gen=torch.Generator(device='cuda').manual_seed(702);torch.manual_seed(702)
    def batch(a,b,n=64,seed=None):
        g=gen if seed is None else torch.Generator(device='cuda').manual_seed(seed)
        copy=torch.randint(4,(n,),generator=g,device='cuda');end=torch.randint(a,b,(n,),generator=g,device='cuda')
        ti=end[:,None]-torch.arange(15,-1,-1,device='cuda')[None];xx=Xn[copy[:,None],ti];kk=K[copy[:,None],ti].view(n,16,4,64)
        s=torch.randn(n,8,generator=g,device='cuda');s/=s.norm(dim=-1,keepdim=True)
        # Include whole-feature query perturbations approximating the extra habit/schema scale.
        extra=torch.randn(n,256,generator=g,device='cuda');extra*=.5*reader.rq.weight.norm()/extra.norm(dim=-1,keepdim=True)
        q=(reader.rq_bias+reader.rq(s)+extra).view(n,4,64)
        logits=torch.einsum('bhd,blhd->bhl',q,kk)/8+reader.rpos[None,None]
        return xx,kk,q,logits.softmax(-1)
    history=[];best=None
    for step in range(601):
        xx,kk,q,target=batch(16,1000);pred=key(xx).view(-1,16,4,64)
        logits=torch.einsum('bhd,blhd->bhl',q,pred)/8+reader.rpos[None,None]
        kl=F.kl_div(logits.log_softmax(-1),target,reduction='batchmean')/4
        mse=(pred-kk).square().mean()/K[:,:1000].var().clamp_min(1e-8)
        loss=kl+.1*mse;optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(key.parameters(),1.);optimizer.step()
        if step%100==0:
            with torch.inference_mode():
                vx,vk,vq,vt=batch(1016,1500,n=128,seed=900);vp=key(vx).view(-1,16,4,64)
                va=(torch.einsum('bhd,blhd->bhl',vq,vp)/8+reader.rpos[None,None]).softmax(-1)
                tv=float(.5*(va-vt).abs().sum(-1).mean());history.append(dict(step=step,loss=float(loss),validation_attention_tv=tv))
                if best is None or tv<best[0]:best=(tv,copy.deepcopy(key.state_dict()),step)
            print(label,step,'loss',float(loss),'val_tv',tv,flush=True)
    key.load_state_dict(best[1]);W=ck['weight'].clone();W[:,:256]=key.weight.T/sd[:,None];bias=ck['bias'].clone();bias[:256]=key.bias-mean@W[:,:256]
    mapped=Reader(R/'src/c8_reference.pt').cuda();mapped.d=H.shape[-1];mapped.rln=nn.Identity();mapped.rk=nn.Linear(mapped.d,256).cuda();mapped.rv=nn.Linear(mapped.d,256).cuda()
    with torch.no_grad():mapped.rk.weight.copy_(W[:,:256].T);mapped.rk.bias.copy_(bias[:256]);mapped.rv.weight.copy_(W[:,256:].T);mapped.rv.bias.copy_(bias[256:])
    mapped.eval().requires_grad_(False)
    with torch.inference_mode():
        actual=torch.cat([reader.rk(reader.rln(Y[0,1500:3000])),reader.rv(reader.rln(Y[0,1500:3000]))],-1)
        pred=X[0,1500:3000]@W+bias;ym=torch.cat([reader.rk(reader.rln(Y[:,:1000])),reader.rv(reader.rln(Y[:,:1000]))],-1).mean((0,1))
        r2=float(1-(pred-actual).square().sum()/(actual-ym).square().sum());tv=[];pc=[];dc=[]
        s=torch.randn(4,8,generator=torch.Generator().manual_seed(881)).cuda();s/=s.norm(dim=-1,keepdim=True)
        for t in range(1500,3000,32):
            a=Y[0,t-15:t+1][None].repeat(4,1,1);b=X[0,t-15:t+1][None].repeat(4,1,1)
            p,g,att,_=reader.read(s,a,None,None);q,h,other,_=mapped.read(s,b,None,None)
            tv.extend((.5*(att-other).abs().sum(-1).mean(-1)).cpu().tolist());pc.extend(F.cosine_similarity(p,q).cpu().tolist());dc.extend(F.cosine_similarity(g,h).cpu().tolist())
        vx,vk,vq,vt=batch(1516,3000,n=256,seed=901);vp=key(vx).view(-1,16,4,64);va=(torch.einsum('bhd,blhd->bhl',vq,vp)/8+reader.rpos[None,None]).softmax(-1)
        perturbed_tv=float(.5*(va-vt).abs().sum(-1).mean())
    passed=r2>=.8 and np.mean(tv)<=.15 and min(np.mean(pc),np.mean(dc))>=.9 and perturbed_tv<=.15
    report=dict(model=label,training='600 AdamW steps on early windows; attention KL plus 0.1 normalized key MSE; values frozen',
      history=history,selected_step=best[2],input_r2=r2,attention_tv=float(np.mean(tv)),picture_cosine=float(np.mean(pc)),destination_cosine=float(np.mean(dc)),
      whole_query_perturbation_attention_tv=perturbed_tv,passed=bool(passed),
      boundary='exploratory reuse of the prior test stream; attention gate <=0.15 including whole-query perturbations; no multimodal claim from text fitting')
    reports.append(report);(R/'results/attention_bridge_training.json').write_text(json.dumps(reports,indent=2))
    torch.save(dict(weight=W.cpu(),bias=bias.cpu(),report=report),R/f'results/attention_bridge_{label}.pt')
    print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
    del X,Y,K,Xn,key,mapped,optimizer,ck,W;gc.collect();torch.cuda.empty_cache()
