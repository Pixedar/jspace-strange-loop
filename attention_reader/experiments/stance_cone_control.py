"""Can stance orientation matter while preserving the designed scalar invariant?"""
import pathlib,sys,json,math,numpy as np,torch
R=pathlib.Path(__file__).parent;sys.path.insert(0,str(R/'src'));from reader import Reader,Life,unit
torch.set_num_threads(2);selection=json.loads((R/'results/long/selection.json').read_text());chosen=selection['copies'];rows=[]
H=torch.tensor(np.concatenate([np.load(f'C:/Users/assas/jspace_v11_runs/core32/states_8B_life_s{s}.npz')['h'][0,256:3000] for s in range(1,7)]),device='cuda',dtype=torch.float32)
for at in [8000,15000]:
 snapshot=torch.load(R/f'results/long/own_83/state_{at}.pt',map_location='cpu',weights_only=False)
 for name in ['s','prev_g','hist','habit','Z','C','shift']:
  x=snapshot['state'].get(name)
  if isinstance(x,torch.Tensor) and x.shape[0]==16:snapshot['state'][name]=x[chosen].clone()
 snapshot['state']['config'].update(copies=4,freeze_at=0,freeze_habit_at=0)
 life=Life(Reader(R/'src/c8_reference.pt').cuda()).restore(snapshot);s=life.s;g=unit(life.prev_g);norm=s.norm(dim=-1,keepdim=True);s0=s/norm;c=(s0*g).sum(-1,keepdim=True);u=unit(s0-c*g)
 v=torch.randn(s.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(901));v=unit(v-(v*g).sum(-1,keepdim=True)*g-(v*u).sum(-1,keepdim=True)*u)
 theta=torch.acos(((math.cos(math.radians(15))-c*c)/(1-c*c)).clamp(-1,1));new=(c*g+torch.sqrt(1-c*c)*(theta.cos()*u+theta.sin()*v))*norm
 angle=torch.acos((unit(new)*s0).sum(-1).clamp(-1,1))*180/math.pi;invariant_error=float(((unit(new)*g).sum(-1)-c[:,0]).abs().max());life.s=new
 base=np.load(R/f'results/directions/{at}_baseline.npz');ss=[];aa=[]
 for t in range(200):
  rec=life.step(H[at-256+t:at-255+t].expand(4,-1));ss.append(life.s.cpu().numpy());aa.append(rec['att'].cpu().numpy())
 ss=np.stack(ss).astype(float);aa=np.stack(aa);bs=base['s'].astype(float);angular=np.arccos(np.clip((ss*bs).sum(-1)/(np.linalg.norm(ss,axis=-1)*np.linalg.norm(bs,axis=-1)),-1,1));tv=.5*abs(aa-base['att']).sum(-1)
 rows.append(dict(at=at,applied_angle_degrees=angle.cpu().tolist(),initial_alignment_error=invariant_error,mean_attention_change=float(tv.mean()),final_stance_angle_degrees=float(np.rad2deg(angular[-1]).mean()),attention_curve=tv.mean(1).tolist(),stance_angle_degrees_curve=np.rad2deg(angular).mean(1).tolist()))
out=R/'results/directions/stance_cone.json';temp=out.with_suffix('.new');temp.write_text(json.dumps(dict(rows=rows,protocol='One 15-degree stance perturbation constrained to preserve its dot product with the previous decoded reading; 200 readings, same frozen-learning setup and source copies as direction assay. Standard geometry is not an awareness measure.'),indent=2));temp.replace(out);print([(r['at'],r['mean_attention_change'],r['final_stance_angle_degrees']) for r in rows])
