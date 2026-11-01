"""Portable matched recent-average control for the complete recorded reader."""
import argparse,json,pathlib,sys
import numpy as np
import torch
sys.path.insert(0,str(pathlib.Path(__file__).parent/'src'))
from reader import Reader,Life
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--states',required=True);p.add_argument('--out',required=True)
    p.add_argument('--weights',default=str(pathlib.Path(__file__).parent/'src/c8_reference.pt'))
    p.add_argument('--source',choices=['own','neighbor'],default='own');p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--seed',type=int,default=40);p.add_argument('--warmup',type=int,default=256)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');args=p.parse_args()
    out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    reader=Reader(args.weights).to(args.device);H=np.load(args.states)['h'][:,:args.steps]
    if H.shape[0]==1:H=np.repeat(H,4,0)
    assert H.shape==(4,args.steps,reader.d)
    H=torch.tensor(H,device=args.device,dtype=torch.float32);H[:,args.warmup:]=H[:1,args.warmup:].repeat(4,1,1)
    life=Life(reader,seed=args.seed);total=torch.zeros(4,256,device=args.device);weight=0.;scale=None;last={};original=reader.read
    def read_average(s,hist,habit,extra):
        nonlocal scale
        actual=None
        if life.t>=300:
            pull=total/max(weight,1e-12)
            if scale is None:scale=float(.5*life.cap/pull.norm(dim=-1).mean().clamp_min(1e-12))
            actual=(scale*pull).view(4,4,64)
        result=original(s,hist,habit,actual);last['att'],last['keys']=result[2],result[3];return result
    reader.read=read_average
    L=torch.zeros((args.steps+99)//100,4,reader.d,device=args.device);S=[];apart=[]
    with torch.no_grad():
        for t in range(args.steps):
            r=life.step(H[:,t]);S.append(r['s'].cpu().numpy());a=r['att']
            v=torch.einsum('bl,bld->bd',a,life.hist);v=(v-v.mean(0,keepdim=True))/v.pow(2).mean(-1,keepdim=True).sqrt();L[t//100]+=v/100
            apart.append(float((.5*(a[:,None]-a[None]).abs().sum(-1)).sum()/12))
            target=(torch.einsum('nhl,nlhd->nhd',last['att'],last['keys'])-last['keys'].mean(1)).reshape(4,256)
            if args.source=='neighbor':target=torch.roll(target,-1,0)
            total=.998*total+target;weight=.998*weight+1
    np.savez_compressed(out/'life.npz',s=np.stack(S),lens_seg=L.half().cpu().numpy(),att_apart=np.array(apart))
    state=life.snapshot();state['mean_controller']=dict(total=total.cpu(),weight=weight,scale=scale,source=args.source)
    torch.save(state,out/'state.pt');(out/'config.json').write_text(json.dumps(vars(args),indent=2));print('saved',out)
if __name__=='__main__':main()
