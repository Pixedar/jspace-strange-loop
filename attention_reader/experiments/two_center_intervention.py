"""Causal follow-up to the observed reading-summary bottleneck.
Shared codebook across copies, to avoid encoding identity in private prototypes.
"""
import pathlib,json,sys,math
import numpy as np,torch
from sklearn.cluster import KMeans
R=pathlib.Path(__file__).parent;sys.path.insert(0,str(R/'src'));from reader import Reader,Life,unit
torch.set_num_threads(2)
protocol=dict(question='Does replacing the reading summary by two shared earlier prototypes preserve later organization, beyond a size-matched random distortion?',
 lives=[10,12,27],fit_steps=[300,1400],intervention_step=1700,end=3000,
 conditions=['unchanged','two_shared_centers','matched_angle_random'],
 specification='Centers fitted jointly to all copies, then normalized to the original radius. Random controls match the conditional angular distance to the nearest center.',
 boundary='exploratory follow-up after inspecting these selected lives, not fresh population confirmation; the rest of the controller remains high dimensional')
(R/'results/two_center_protocol.json').write_text(json.dumps(protocol,indent=2))
class QReader(Reader):
 def read(self,*args,**kwargs):
    p,g,att,k=super().read(*args,**kwargs)
    if self.mode!='unchanged' and self.step>=1700:
        ix=((p[:,None]-self.centers[None])**2).sum(-1).argmin(-1);target=self.centers[ix]
        if self.mode=='two_shared_centers':p=target
        else:
            cos=(p*target).sum(-1,keepdim=True)/32
            r=torch.randn(p.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(80000+self.step))
            r=unit(r-(r*p).sum(-1,keepdim=True)*p/32)*math.sqrt(32)
            p=cos*p+(1-cos.square()).clamp_min(0).sqrt()*r
        g=unit(self.dec(p))
    return p,g,att,k
reader=QReader(R/'src/c8_reference.pt').cuda();z=np.load('C:/Users/assas/jspace_v11_runs/core32/states_8B_life_s2.npz');H=torch.tensor(z['h'][:,:3000],device='cuda',dtype=torch.float32);H[:,256:]=H[:1,256:].repeat(4,1,1)
rows=[]
for seed in protocol['lives']:
    prior=np.load(R/f'results/signals_seed{seed}.npz');centers=KMeans(n_clusters=2,random_state=553,n_init=10).fit(prior['reading'][300:1400].reshape(-1,32)).cluster_centers_
    reader.centers=unit(torch.tensor(centers,device='cuda'))*math.sqrt(32)
    init=57 if seed==27 else json.loads(pathlib.Path(f'D:/jspace_v11_runs/core32_case58/cs_c8_S_seed{seed}/run.json').read_text())['init_seed']
    baseline=None
    for mode in protocol['conditions']:
        reader.mode=mode;life=Life(reader,seed=init);S=[];A=[];P=torch.zeros(30,4,4096,device='cuda');pre_error=0.
        for t in range(3000):
            reader.step=t;rec=life.step(H[:,t]);S.append(rec['s'].cpu().numpy());a=rec['att'];pad=np.zeros((4,16));pad[:,-a.shape[-1]:]=a.cpu().numpy();A.append(pad)
            attended=torch.einsum('bl,bld->bd',a,life.hist);center=(attended-attended.mean(0,keepdim=True))/attended.square().mean(-1,keepdim=True).sqrt();P[t//100]+=center/100
            if t<1700:pre_error=max(pre_error,float((rec['s']-torch.tensor(prior['stance'][t],device='cuda')).abs().max()))
        S=np.stack(S);A=np.stack(A);profiles=P.cpu().numpy();V=profiles[17:].astype(float);V-=V.mean(-1,keepdims=True);V/=np.linalg.norm(V,axis=-1,keepdims=True).clip(1e-12);own=[];other=[]
        for a in range(len(V)):
         for b in range(a+3,len(V)):
          for c in range(4):
            own.append(V[a,c]@V[b,c]);other.extend(V[a,c]@V[b,d] for d in range(4) if d!=c)
        if mode=='unchanged':baseline=A.copy()
        row=dict(seed=seed,condition=mode,post_persistence=float(np.mean(own)-np.mean(other)),
             mean_attention_difference=float(.5*abs(A[1700:]-baseline[1700:]).sum(-1).mean()),pre_intervention_max_stance_error=pre_error)
        rows.append(row);np.savez_compressed(R/f'results/two_center_{seed}_{mode}.npz',s=S,att=A,profiles=profiles)
        print(json.dumps(row),flush=True)
(R/'results/two_center_intervention.json').write_text(json.dumps(dict(protocol=protocol,rows=rows),indent=2))
