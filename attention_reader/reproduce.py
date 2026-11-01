"""Portable recorded-state runner for the selected complete mechanism."""
import argparse,json,pathlib,sys,time
import numpy as np
import torch
sys.path.insert(0,str(pathlib.Path(__file__).parent/'src'))
from reader import Reader,Life
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--states',required=True,help='NPZ with h of shape (4,time,hidden), from a specified model/layer')
    p.add_argument('--weights',default=str(pathlib.Path(__file__).parent/'src/c8_reference.pt'))
    p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--seed',type=int,default=40);p.add_argument('--warmup',type=int,default=256)
    p.add_argument('--schema',choices=['own','none','neighbor'],default='own')
    p.add_argument('--exchange',choices=['swap','random']);p.add_argument('--at',type=int,default=1000);p.add_argument('--until',type=int,default=2000)
    p.add_argument('--freeze-schema',type=int);p.add_argument('--freeze-habit',type=int)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--resume',help='Trusted local full controller checkpoint. Continue from its saved next observation index.')
    args=p.parse_args();out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);reader=Reader(args.weights).to(args.device)
    z=np.load(args.states);H=z['h'][:,:args.steps]
    if H.shape[0]==1:H=np.repeat(H,4,axis=0)
    assert H.shape[0]==4 and H.shape[-1]==reader.d and H.shape[1]==args.steps
    H=torch.tensor(H,device=args.device,dtype=torch.float32);H[:,args.warmup:]=H[:1,args.warmup:].repeat(4,1,1)
    life=Life(reader,seed=args.seed,schema=args.schema,exchange=args.exchange,at=args.at,until=args.until,
              freeze_at=args.freeze_schema,freeze_habit_at=args.freeze_habit)
    if args.resume:life.restore(torch.load(args.resume,map_location='cpu',weights_only=False))
    trace=[];t0=time.time()
    for t in range(life.t,args.steps):
        r=life.step(H[:,t]);trace.append({k:r[k].cpu().numpy() for k in ('s','g','picture','schema_error','schema_total')})
        if (t+1)%500==0:life.save(out/'state.pt');print('reading',t+1,'seconds',round(time.time()-t0),flush=True)
    if trace:np.savez_compressed(out/'trace.npz',**{k:np.stack([r[k] for r in trace]) for k in trace[0]})
    life.save(out/'state.pt');(out/'config.json').write_text(json.dumps(vars(args),indent=2))
    print('saved',str(out))
if __name__=='__main__':main()
