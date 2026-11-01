"""The complete c8 milestone, isolated from the architecture-search code.

Frozen reader weights; online habit and attention schema. No semantic labels,
personality target, progress reward, text memory or optimization objective.
"""
import copy
import math
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def carry(a, b, x):
    c = (a*b).sum(-1, keepdim=True)
    w = b-c*a
    sn = w.norm(dim=-1, keepdim=True)
    w = w/sn.clamp_min(1e-8)
    xa, xw = (x*a).sum(-1, keepdim=True), (x*w).sum(-1, keepdim=True)
    out = x + (c*xa-sn*xw-xa)*a + (sn*xa+c*xw-xw)*w
    return torch.where(sn > 1e-8, out, x)


class Reader(nn.Module):
    def __init__(self, weights):
        super().__init__()
        w = torch.load(weights, map_location='cpu', weights_only=True)
        self.d = w['rln.weight'].numel()
        self.rln = nn.LayerNorm(self.d)
        self.rk, self.rv = nn.Linear(self.d,256), nn.Linear(self.d,256)
        self.rq = nn.Linear(8,256)
        self.rq_bias = nn.Parameter(torch.zeros(256))
        self.rpos = nn.Parameter(torch.zeros(16))
        self.to_pic = nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,32))
        self.dec = nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,8))
        self.load_state_dict({k:v for k,v in w.items() if k in self.state_dict()}, strict=True)
        self.eval().requires_grad_(False)

    def read(self, s, hist, habit, extra):
        n,l,_ = hist.shape
        a = self.rln(hist)
        k,v = self.rk(a).view(n,l,4,64),self.rv(a).view(n,l,4,64)
        q = (self.rq_bias.expand(n,-1)+self.rq(s)).view(n,4,64)
        if extra is not None: q = q+extra
        if habit is not None: q = q+torch.einsum('nhdm,nm->nhd',habit,s)
        att = torch.softmax(torch.einsum('nhd,nlhd->nhl',q,k)/8 + self.rpos[-l:][None,None],-1)
        r = torch.einsum('nhl,nlhd->nhd',att,v).reshape(n,-1)
        raw = self.to_pic(r)
        picture = math.sqrt(32)*raw/raw.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        dest = self.dec(picture)
        dest = dest/dest.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        return picture,dest,att,k


class Life:
    def __init__(self, reader, seed=40, copies=4, schema='own', habit=True,
                 exchange=None, at=1000, until=2000, freeze_at=None, freeze_habit_at=None):
        self.reader=reader
        self.dev=next(reader.parameters()).device
        self.config=dict(seed=seed,copies=copies,schema=schema,habit=habit,exchange=exchange,
                         at=at,until=until,freeze_at=freeze_at,freeze_habit_at=freeze_habit_at)
        self.t=0
        self.s=unit(torch.randn(copies,8,generator=torch.Generator().manual_seed(seed))).to(self.dev)[:1].repeat(copies,1)
        self.prev_g=None
        self.hist=torch.zeros(copies,0,reader.d,device=self.dev)
        self.habit=torch.zeros(copies,4,64,8,device=self.dev) if habit else None
        self.cap=float(.5*reader.rq.weight.norm())
        self.rate=None
        self.Z=torch.zeros(copies,16,16,device=self.dev)
        self.C=torch.zeros(copies,256,16,device=self.dev)
        self.proj=torch.randn(16,256,generator=torch.Generator().manual_seed(1616)).to(self.dev)/16
        self.kappa=None
        self.shift=None

    @torch.no_grad()
    def step(self, h):
        cfg=self.config
        self.hist=torch.cat([self.hist,h.to(self.dev)[:,None]],1)[:,-16:]
        n,l,_=self.hist.shape
        keys=self.reader.rk(self.reader.rln(self.hist)).view(n,l,4,64)
        z=unit(torch.einsum('ij,bj->bi',self.proj,keys.mean(1).reshape(n,-1)))*4
        A=self.C@torch.linalg.inv(self.Z+1e-3*torch.eye(16,device=self.dev))
        used=A
        in_window=cfg['at'] <= self.t < cfg['until']
        if in_window and cfg['exchange']:
            if cfg['exchange']=='swap': used=A[torch.tensor([1,0]+list(range(2,n)),device=self.dev)]
            elif cfg['exchange']=='random':
                if self.shift is None:
                    gen=torch.Generator().manual_seed(4646)
                    size=(A[0]-A[1]).norm()
                    self.shift=torch.zeros_like(A)
                    for b in (0,1):
                        r=torch.randn(A.shape[1:],generator=gen).to(self.dev)
                        self.shift[b]=r*size/r.norm().clamp_min(1e-12)
                used=A+self.shift
        pull=torch.einsum('bij,bj->bi',used,z)
        extra=None
        if cfg['schema']!='none' and self.t>=300:
            if self.kappa is None:
                self.kappa=float(.5*self.cap/pull.norm(dim=-1).mean().clamp_min(1e-12))
            extra=(self.kappa*pull).view(n,4,64)
        picture,g,att,k=self.reader.read(self.s,self.hist,self.habit,extra)
        kp=(torch.einsum('nhl,nlhd->nhd',att,keys)-keys.mean(1)).reshape(n,-1)
        pred=torch.einsum('bij,bj->bi',A,z)
        kc=(keys-keys.mean(1,keepdim=True)).reshape(n,l,-1)
        expect=torch.einsum('bd,bld->bl',pred,kc)
        record=dict(s=self.s.clone(),g=g.clone(),picture=picture.clone(),att=att.mean(1).clone(),
                    schema_error=((pred-kp)**2).sum(-1),schema_total=(kp**2).sum(-1),
                    expected=expect,schema=A.clone(),habit=None if self.habit is None else self.habit.clone())
        if cfg['freeze_at'] is None or self.t<cfg['freeze_at']:
            zl,kl=(torch.roll(z,-1,0),torch.roll(kp,-1,0)) if cfg['schema']=='neighbor' else (z,kp)
            self.Z=.998*self.Z+torch.einsum('bi,bj->bij',zl,zl)
            self.C=.998*self.C+torch.einsum('bi,bj->bij',kl,zl)
        new_s=self.s if self.prev_g is None else carry(self.prev_g,g,self.s)
        if self.habit is not None and self.prev_g is not None and l==16 and (cfg['freeze_habit_at'] is None or self.t<cfg['freeze_habit_at']):
            kap=torch.einsum('nhl,nlhd->nhd',att,k)-k.mean(1)
            w=torch.einsum('nhd,nm->nhdm',kap,self.s)
            if self.rate is None: self.rate=float(self.cap/(1000*float(w.flatten(1).norm(dim=-1).mean().clamp_min(1e-12))))
            self.habit=self.habit+self.rate*w
            self.habit=self.habit*(self.cap/self.habit.flatten(1).norm(dim=-1).clamp_min(1e-12)).clamp(max=1).view(n,1,1,1)
        self.s,self.prev_g=new_s,g
        self.t+=1
        return record

    def snapshot(self):
        state={k:(v.detach().cpu().clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v))
               for k,v in self.__dict__.items() if k not in ('reader','dev')}
        return dict(format=1,state=state,torch_rng=torch.random.get_rng_state(),numpy_rng=np.random.get_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix(path.suffix+'.partial')
        torch.save(self.snapshot(),tmp); os.replace(tmp,path)

    def restore(self,snapshot):
        for k,v in snapshot['state'].items(): setattr(self,k,v.to(self.dev) if isinstance(v,torch.Tensor) else copy.deepcopy(v))
        torch.random.set_rng_state(snapshot['torch_rng'])
        np.random.set_state(snapshot['numpy_rng'])
        if snapshot['cuda_rng'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(snapshot['cuda_rng'])
        return self
