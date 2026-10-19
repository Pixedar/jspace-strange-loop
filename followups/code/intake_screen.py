import os
"""Matched final intake screening. Cached features, no model feedback/language loss during training.
Position warm-up uses no divergence, and each step presents four distinct factorial contexts.
Both intake arms use identical initialization, schedules, probes and optimization.
"""
import copy, math, time, json, traceback
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
unit=lambda x:x/x.norm(dim=-1,keepdim=True).clamp_min(1e-8)
screen_dir=Path(os.environ.get('JSPACE_MULTIMODAL_OUTPUT','artifacts/multimodal'))
class ScreenReader(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.rln=nn.LayerNorm(d)
        self.rk,self.rv=nn.Linear(d,256),nn.Linear(d,256)
        self.rq=nn.Linear(8,256)
        self.rq_bias=nn.Parameter(torch.randn(256)*.1)
        self.rpos=nn.Parameter(torch.zeros(16))
        self.to_pic=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,32))
        self.dec=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,8))
        self.field=nn.Sequential(nn.Linear(40,128),nn.GELU(),nn.Linear(128,8))

    def read(self,s,h,clamp=False):
        a=self.rln(h)
        k,v=self.rk(a).view(-1,16,4,64),self.rv(a).view(-1,16,4,64)
        q=(self.rq_bias[None]+(0 if clamp else self.rq(s))).view(-1,4,64)
        if q.shape[0]==1: q=q.expand(len(s),-1,-1)
        att=torch.softmax(torch.einsum('nhd,nlhd->nhl',q,k)/8+self.rpos[None,None],-1)
        r=torch.einsum('nhl,nlhd->nhd',att,v).reshape(-1,256)
        return math.sqrt(32)*unit(self.to_pic(r)),att.mean(1)

    def nxt(self,p): return unit(self.dec(p))

    def predict(self,p,S):
        return self.field(torch.cat([p.expand(len(S),-1),S],-1))


def nce(pred,targets,b):
    d2=((pred[None]-targets)**2).sum(-1)
    tau=d2.detach().median().clamp_min(1e-6)
    return F.cross_entropy((-d2/tau).T,torch.full((len(pred),),b,device='cuda',dtype=torch.long))


def training(f,steps,centered,seed,record=None,other=None):
    opt=torch.optim.Adam(f.parameters(),lr=.002)
    g=torch.Generator().manual_seed(seed)
    schedule=np.stack([np.random.default_rng(seed+t).permutation(4) for t in range(steps)])
    s=unit(torch.randn(4,8,generator=g)).cuda()
    loss_chunk=0.
    values=[]
    for t in range(steps):
        if time.time()-screen_started>900:
            raise RuntimeError('Bounded screening stopped at 15-minute deadline')
        idx=torch.tensor(schedule[t],device='cuda')
        h=H[idx]
        S=unit(torch.randn(12,8,generator=g)).cuda()
        SS=S.repeat(4,1)
        hp=h[:,None].expand(-1,12,-1,-1).reshape(48,16,-1)
        P,_=f.read(s,h)
        next_s=f.nxt(P)
        target=f.nxt(f.read(SS,hp)[0]).view(4,12,8)-S[None]
        if other is not None: target=other[t].to('cuda')
        if record is not None: record.append(target.detach().cpu())
        if centered:
            target=target+S[None]
            target=target-target.mean(1,keepdim=True)
        loss=0.
        for b in range(4):
            prediction=f.predict(P[b:b+1],S)
            if centered:
                prediction=prediction+S
                prediction=prediction-prediction.mean(0,keepdim=True)
            loss=loss+nce(prediction,target,b)/4
            if not centered:
                far=S[(target[b].detach().norm(dim=-1)).argmax()]
                loss=loss+0.0*(next_s[b]-far).norm()/4
        loss_chunk=loss_chunk+loss/8
        s=next_s
        if (t+1)%8==0:
            opt.zero_grad()
            loss_chunk.backward()
            torch.nn.utils.clip_grad_norm_(f.parameters(),10.)
            opt.step()
            s=s.detach()
            loss_chunk=0.
        if (t+1)%500==0:
            values.append({'step':t+1,'loss':float(loss.detach())})
            print('SCREEN_TRAIN',seed,centered,other is not None,t+1,float(loss.detach()),flush=True)
    return values


@torch.no_grad()
def evaluate(f,h):
    S=unit(torch.randn(32,8,generator=torch.Generator().manual_seed(4242))).cuda()
    s=unit(torch.randn(4,8,generator=torch.Generator().manual_seed(7))).cuda()
    for _ in range(64): s=f.nxt(f.read(s,h)[0])
    destinations=f.nxt(f.read(S.repeat(4,1),h[:,None].expand(-1,32,-1,-1).reshape(128,16,-1))[0]).view(4,32,8)
    res=destinations-destinations.mean(1,keepdim=True)
    field=destinations-S[None]
    den=((field-field.mean((0,1),keepdim=True))**2).sum().clamp_min(1e-8)
    P=f.read(s,h)[0]
    predicted=torch.stack([f.predict(P[b:b+1],S)+S for b in range(4)])
    predicted=predicted-predicted.mean(1,keepdim=True)
    corr=F.cosine_similarity(res.flatten()[None],predicted.flatten()[None]).item()
    clamped=f.nxt(f.read(S.repeat(4,1),h[:,None].expand(-1,32,-1,-1).reshape(128,16,-1),clamp=True)[0]).view(4,32,8)
    return {'state_dependent_share':float(res.square().sum()/den), 'pattern_correlation':corr,
            'unscaled_variance_explained':float(1-(res-predicted).square().sum()/res.square().sum().clamp_min(1e-8)),
            'clamped_destination_variance':float((clamped-clamped.mean(1,keepdim=True)).square().mean())}



intake_models={}
intake_result={'complete':False,'kind':'paired cached-native-intake screening; not full live v24',
    'seeds':[0,1],'position_steps':2000,'centered_steps':3000,'warmup_divergence_weight':0,
    'input_schedule':'deterministic permutations of four distinct contexts per step',
    'model_feedback_during_training':False,'prediction_loss_during_training':False,'results':[]}
try:
    for intake in ('suffix','balanced'):
        screen_started=time.time()
        H=torch.stack([torch.tensor(balanced_features[f'p0_r{r}_n{n}_joint_'+intake],device='cuda') for r in (0,1) for n in (1,2)])
        H_test=torch.stack([torch.tensor(balanced_features[f'p1_r{r}_n{n}_joint_'+intake],device='cuda') for r in (0,1) for n in (1,2)])
        for seed in (0,1):
            torch.manual_seed(seed)
            start=ScreenReader(H.shape[-1]).cuda()
            warmup=training(start,2000,False,100+seed)
            warmup_eval=evaluate(start,H)
            own,foreign=copy.deepcopy(start),copy.deepcopy(start)
            twin=[]
            curve_self=training(own,3000,True,200+seed,record=twin)
            curve_other=training(foreign,3000,True,200+seed,other=twin)
            for condition,f,curve in [('self',own,curve_self),('recorded_other',foreign,curve_other)]:
                intake_models[(intake,seed,condition)]=f
                intake_result['results'].append({'intake':intake,'seed':seed,'condition':condition,
                    'warmup':warmup_eval,'training_inputs':evaluate(f,H),'heldout_colours':evaluate(f,H_test),'curve':curve})
                torch.save(f.state_dict(),screen_dir/f'final_{intake}_{seed}_{condition}.pt')
                print('INTAKE_SCREEN_RESULT',json.dumps(intake_result['results'][-1]),flush=True)
    intake_result['complete']=True
except Exception:
    intake_result['error']=traceback.format_exc()
(screen_dir/'intake_screening.json').write_text(json.dumps(intake_result,indent=2))
print('INTAKE_SCREEN_JSON_BEGIN\n'+json.dumps(intake_result)+'\nINTAKE_SCREEN_JSON_END',flush=True)
print('INTAKE_SCREEN_FINISHED',intake_result['complete'],flush=True)
