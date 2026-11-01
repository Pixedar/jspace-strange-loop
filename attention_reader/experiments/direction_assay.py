"""Causal routes between stance, reading, decoded direction and next selection."""
import pathlib,sys,json,copy,math
import numpy as np,torch
R=pathlib.Path(__file__).parent;sys.path.insert(0,str(R/'src'));from reader import Reader,Life,unit,carry
torch.set_num_threads(2);OUT=R/'results/directions';OUT.mkdir(exist_ok=True)
selection=json.loads((R/'results/long/selection.json').read_text());cohort=selection['cohort'];chosen=selection['copies']
hh=[]
for stream in range(1,7):hh.append(np.load(f'C:/Users/assas/jspace_v11_runs/core32/states_8B_life_s{stream}.npz')['h'][0,256:3000])
H=torch.tensor(np.concatenate(hh),device='cuda',dtype=torch.float32)
arms=['baseline','stance_toward_reading','stance_random','stance_random_query_blocked','picture_neighbor','picture_random','picture_neighbor_turn_blocked','picture_fixed_angle_random','picture_decoder_sensitive','picture_decoder_sensitive_turn_blocked']
def rotate(x,target,angle):
    r=target-(x*target).sum(-1,keepdim=True)*x/x.square().sum(-1,keepdim=True)
    fallback=torch.randn(x.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(187))
    fallback-= (x*fallback).sum(-1,keepdim=True)*x/x.square().sum(-1,keepdim=True)
    r=torch.where(r.norm(dim=-1,keepdim=True)<1e-7,fallback,r);r=unit(r)*x.norm(dim=-1,keepdim=True)
    return angle.cos()*x+angle.sin()*r
class DirectedReader(Reader):
 def read(self,s,hist,habit,extra):
    qs=self.baseline_query[self.frame] if self.mode=='stance_random_query_blocked' else s
    p,g,att,k=super().read(qs,hist,habit,extra);self.plain_g=g.clone();self.plain_p=p.clone()
    if self.frame==0 and self.mode.startswith('picture'):
        other=torch.roll(p,-1,0);cos=(p*other).sum(-1,keepdim=True)/32;angle=torch.acos(cos.clamp(-1,1)).clamp(max=math.radians(15))
        if 'fixed_angle' in self.mode or 'decoder_sensitive' in self.mode:angle=torch.full_like(angle,math.radians(15))
        if 'decoder_sensitive' in self.mode:
            directions=[]
            for c in range(4):
                with torch.enable_grad():
                    x=p[c].detach().clone().requires_grad_(True)
                    def f(q):return carry(self.step_prev[c:c+1],unit(self.dec(q[None])),self.step_s[c:c+1])[0]
                    J=torch.autograd.functional.jacobian(f,x);u=unit(x.detach());J=J-(J@u)[:,None]*u[None]
                    _,_,vh=torch.linalg.svd(J,full_matrices=False);directions.append(vh[0])
            target=torch.stack(directions)
        else:target=other if 'neighbor' in self.mode else torch.randn(p.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(188))
        p=rotate(p,target,angle);g=unit(self.dec(p));self.applied_picture_angle=angle
    return p,g,att,k
rows=[]
for at in [8000,15000]:
    snapshot=torch.load(R/f'results/long/own_{cohort}/state_{at}.pt',map_location='cpu',weights_only=False)
    for name in ['s','prev_g','hist','habit','Z','C','shift']:
        v=snapshot['state'].get(name)
        if isinstance(v,torch.Tensor) and v.shape[0]==16:snapshot['state'][name]=v[chosen].clone()
    snapshot['state']['config'].update(copies=4,freeze_at=0,freeze_habit_at=0)
    base=None
    for mode in arms:
        reader=DirectedReader(R/'src/c8_reference.pt').cuda();reader.mode=mode;reader.baseline_query=None if base is None else torch.tensor(base['before_s'],device='cuda')
        life=Life(reader).restore(snapshot)
        if mode.startswith('stance'):
            target=life.prev_g if mode=='stance_toward_reading' else torch.randn(life.s.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(189))
            life.s=rotate(life.s,target,torch.full((4,1),math.radians(15),device='cuda'))
        traces={k:[] for k in ['before_s','s','picture','g','att']}
        for frame in range(200):
            reader.frame=frame;old_s=life.s.clone();old_prev=life.prev_g.clone();reader.step_s=old_s;reader.step_prev=old_prev;rec=life.step(H[at-256+frame:at-255+frame].expand(4,-1))
            if frame==0 and mode.startswith('picture') and mode.endswith('turn_blocked'):life.s=carry(old_prev,reader.plain_g,old_s);life.prev_g=reader.plain_g
            traces['before_s'].append(old_s.cpu().numpy());traces['s'].append(life.s.cpu().numpy())
            for key in ['picture','g','att']:traces[key].append(rec[key].cpu().numpy())
        traces={k:np.stack(v) for k,v in traces.items()}
        if mode=='baseline':base=traces
        tv=.5*abs(traces['att']-base['att']).sum(-1);av=traces['s'].astype(float);bv=base['s'].astype(float);sd=np.arccos(np.clip((av*bv).sum(-1)/(np.linalg.norm(av,axis=-1)*np.linalg.norm(bv,axis=-1)),-1,1))
        pp=np.linalg.norm(traces['picture']-base['picture'],axis=-1);gg=np.linalg.norm(traces['g']-base['g'],axis=-1)
        row=dict(at=at,condition=mode,source_copies=chosen,applied_picture_angles_degrees=(reader.applied_picture_angle.cpu().numpy().ravel()*180/np.pi).tolist() if mode.startswith('picture') else None,attention_curve=tv.mean(1).tolist(),stance_angle_curve=sd.mean(1).tolist(),
             picture_curve=pp.mean(1).tolist(),direction_curve=gg.mean(1).tolist(),mean_attention_change=float(tv.mean()),
             final_stance_angle=float(sd[-1].mean()),max_picture_change=float(pp.max()),
             blocked_route_max_attention_change=float(tv.max()) if mode.endswith('blocked') else None)
        rows.append(row);path=OUT/f'{at}_{mode}.npz';tmp=path.with_suffix('.new')
        with tmp.open('wb') as f:np.savez_compressed(f,**traces)
        tmp.replace(path);print(at,mode,row['mean_attention_change'],flush=True)
(OUT/'analysis.json').write_text(json.dumps(dict(rows=rows,protocol=dict(source=selection,steps=200,moments=[8000,15000],angle_degrees=15,
     learning='habit and schema updates frozen in every branch to isolate the fast causal routes',
     controls='stance perturbation with baseline query clamped; picture perturbation with the turn and previous destination restored to the baseline update',
     scope='local frozen-controller causal assay, not a new live LLM or a claim about conscious direction meanings')),indent=2))
