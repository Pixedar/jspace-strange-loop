"""Local causal derivatives: isolate context-dependent schema feedback from bias.
No LLM weights needed locally. This is an experimenter's microscope, not
knowledge that the controller itself has of its own influence.
"""
import os,pathlib,json,sys
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import torch,numpy as np
R=pathlib.Path(__file__).parent;C=R/'specificity/results/cloud/58704070/results';OUT=R/'results/sensitivities';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'src'));from reader import Reader,Life,unit,carry
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
spec=json.loads((R/'specificity/inputs/specificity_spec.json').read_text());scenes=spec['scenes']
assert all((C/f'common_states_{s["id"]}.npz').exists() for s in scenes),'Wait for verified common-context inputs'
reader=Reader(R/'src/c8_reference.pt').cuda();B=torch.load(C/'specificity_lens.pt',map_location='cuda',weights_only=True)['basis']
train_ids=[s['id'] for s in scenes if s['family'] in ['sensors','promise','perspective','study']]
X=torch.tensor(np.concatenate([np.load(C/f'common_states_{i}.npz')['host24'] for i in train_ids]).astype(np.float32),device='cuda');X-=X.mean(0)
torch.manual_seed(392);_,sv,v=torch.pca_lowrank(X,q=80,center=False,niter=4);P=v[:,:64]
random=[]
for i in range(5):
 g=torch.Generator(device='cuda').manual_seed(393+i);q,_=torch.linalg.qr(torch.randn(4096,64,generator=g,device='cuda'),mode='reduced');random.append(q)
rows=[];effects=[];qualification=[]
conditions=['full','unused_schema','mean_memory','random_act','rotated_act','fixed_act','neighbor_act']
for scene in [s for s in scenes if s['family'] not in ['sensors','promise','perspective','study']]:
    ident=scene['id'];snapshot=torch.load(C/f'source_state_{ident}.pt',map_location='cpu',weights_only=False);life=Life(reader).restore(snapshot)
    data=np.load(C/f'return_h24_{ident}.npz');h=torch.tensor(data['h24'],device='cuda');base=h[-1]
    A=life.C@torch.linalg.inv(life.Z+1e-3*torch.eye(16,device='cuda'))
    meanpull=snapshot['ordinary_mean']['total'].cuda()/snapshot['ordinary_mean']['weight'];allJ=[]
    for copy_index in range(4):
        prior=life.hist[copy_index:copy_index+1,-15:].detach();s=life.s[copy_index:copy_index+1].detach();old=life.prev_g[copy_index:copy_index+1].detach();habit=life.habit[copy_index:copy_index+1].detach();aa=A[copy_index:copy_index+1].detach()
        def expected(x):
            hist=torch.cat([prior,x[None,None]],1);keys=reader.rk(reader.rln(hist)).mean(1);z=unit(keys@life.proj.T)*4
            return torch.einsum('bij,bj->bi',aa,z)*life.kappa
        fixed=expected(base).detach();mean=unit(meanpull[copy_index:copy_index+1])*fixed.norm(dim=-1,keepdim=True)
        def function(mode,x):
            hist=torch.cat([prior,x[None,None]],1)
            extra=expected(x) if mode=='own_context' else (fixed if mode=='own_frozen' else (mean if mode=='private_mean' else None))
            p,g,att,k=reader.read(s,hist,habit,None if extra is None else extra.view(1,4,64))
            return carry(old,g,s)[0]
        modes=['own_context','own_frozen','private_mean','unused'];Js={};values={}
        for mode in modes:
            x=base.detach().clone().requires_grad_(True)
            J=torch.autograd.functional.jacobian(lambda q:function(mode,q),x,vectorize=True).detach();Js[mode]=J;values[mode]=function(mode,base).detach()
            energy=J.square().sum().clamp_min(1e-30)
            r=[float((J@q).square().sum()/energy) for q in random]
            rows.append(dict(scene=ident,family=scene['family'],copy=copy_index,mode=mode,jacobian_norm=float(energy.sqrt()),
                J_lens_basis_fraction=float((J@B).square().sum()/energy),activity_PCA_fraction=float((J@P).square().sum()/energy),random_basis_fractions=r,
                input='one current 4096-number model observation',output='next 8-number stance; before learning updates'))
        assert float((values['own_context']-values['own_frozen']).abs().max())==0,'Frozen forecast must match at the unperturbed point'
        deltaJ=Js['own_context']-Js['own_frozen'];energy=deltaJ.square().sum().clamp_min(1e-30)
        rows.append(dict(scene=ident,family=scene['family'],copy=copy_index,mode='context_feedback_only',jacobian_norm=float(energy.sqrt()),
            relative_to_full=float(deltaJ.norm()/Js['own_context'].norm().clamp_min(1e-30)),
            J_lens_basis_fraction=float((deltaJ@B).square().sum()/energy),activity_PCA_fraction=float((deltaJ@P).square().sum()/energy),
            random_basis_fractions=[float((deltaJ@q).square().sum()/energy) for q in random]))
        for j,condition in enumerate(conditions):
            actual=h[j*4+copy_index];dh=actual-base;full=function('own_context',actual).detach()-values['own_context'];frozen=function('own_frozen',actual).detach()-values['own_frozen'];prediction=Js['own_context']@dh
            effects.append(dict(scene=ident,family=scene['family'],copy=copy_index,condition=condition,
                model_return_size=float(dh.norm()),stance_change=float(full.norm()),context_feedback_change=float((full-frozen).norm()),
                feedback_gain_per_return=float((full-frozen).norm()/dh.norm().clamp_min(1e-12)),
                linear_prediction_relative_error=float((prediction-full).norm()/full.norm().clamp_min(1e-12)),
                signal_resolved=bool(full.norm()>1e-6)))
        # Exact formula check against the saved-state simulator.
        twin=Life(reader).restore(snapshot);copybase=base[None].expand(4,-1);twin.step(copybase)
        error=float((twin.s[copy_index]-values['own_context']).abs().max());assert error<2e-6
        qualification.append(dict(scene=ident,copy=copy_index,maximum_formula_error=error))
        allJ.append(torch.stack([Js[m] for m in modes]+[deltaJ]).cpu().numpy())
    np.savez_compressed(OUT/f'jacobians_{ident}.npz',jacobians=np.stack(allJ))
    print('sensitivity',ident,'done',flush=True)
protocol=dict(question='How much does allowing the attention forecast to respond to new model activity change the stance response, beyond an identical frozen forecast value?',
    modes=['own_context','own_frozen','private_mean','unused'],
    control='own_context and own_frozen give exactly the same base query and output; only their response to a changed input differs',
    bases='rank-64 J-lens-derived linear basis; activity PCA fitted on four other scene families; five random orthonormal rank-64 bases',
    evaluation_families=['debate','editing','imagination','planning'],
    scope='local derivatives and finite returns at one probe position in eight scenes; experimenter-supplied analysis, not learned self-attribution')
(OUT/'analysis.json').write_text(json.dumps(dict(protocol=protocol,rows=rows,effects=effects,qualification=qualification),indent=2))
print('Sensitivity analysis complete',len(rows),'rows')
