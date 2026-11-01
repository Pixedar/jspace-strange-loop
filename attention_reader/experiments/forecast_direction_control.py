"""Long continuation: context-dependent forecast versus a private fixed bias.
All query-bias magnitudes are yoked to the full reference at each reading.
"""
import pathlib,sys,json,gc
import numpy as np,torch
R=pathlib.Path(__file__).parent;sys.path.insert(0,str(R/'src'));from reader import Reader,Life,unit
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
OUT=R/'results/forecast_direction';OUT.mkdir(exist_ok=True)
arrays=[np.load(f'C:/Users/assas/jspace_v11_runs/core32/states_8B_life_s{i}.npz')['h'][0,256:3000] for i in range(1,7)]
H=torch.tensor(np.concatenate(arrays),device='cuda',dtype=torch.float32);AT=8000;END=16720;B=16;arms=['full','frozen_direction','rotated_direction','random_fixed_direction']
protocol=dict(start=AT,end=END,cohorts=[83,84],copies=B,arms=arms,
 mechanism='Each branch starts at the same developed state. Full reference query-bias norms are replayed to every branch at every reading. Only forecast directions change; habit and schema training continue, but altered branches do not use their own forecast output.',
 controls='Frozen direction is each copy’s full prediction at the intervention moment. Rotated direction is a fixed signed coordinate permutation of the reference prediction stream. Random fixed is a norm-matched constant direction.',
 scope='Yoked diagnostic replays, not newly self-contained deployed models; tests whether current-context forecast directions add beyond a developed private bias.')
(OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
for cohort in protocol['cohorts']:
    snapshot=torch.load(R/f'results/long/own_{cohort}/state_{AT}.pt',map_location='cpu',weights_only=False)
    prior=np.load(R/f'results/long/own_{cohort}/trace.npz');priorS=prior['s'];priorP=prior['picture'];source=[];fixed=None
    g=torch.Generator(device='cuda').manual_seed(441);perms=torch.stack([torch.randperm(256,generator=g,device='cuda') for _ in range(B)]);signs=torch.where(torch.rand(B,256,generator=g,device='cuda')>.5,1.,-1.);random=unit(torch.randn(B,256,generator=g,device='cuda'))
    for arm in arms:
        reader=Reader(R/'src/c8_reference.pt').cuda();life=Life(reader).restore(snapshot);base=reader.read;frame=[0];errors=[]
        def controlled(s,hist,habit,extra):
            nonlocal_placeholder=None
            global fixed
            if arm=='full':
                source.append(extra.flatten(1).detach().clone())
                if fixed is None:fixed=unit(source[-1]).clone()
            else:
                reference=source[frame[0]];norm=reference.norm(dim=-1,keepdim=True)
                if arm=='frozen_direction':actual=fixed*norm
                elif arm=='rotated_direction':actual=reference.gather(1,perms)*signs
                else:actual=random*norm
                extra=actual.view(B,4,64)
                errors.append(float((actual.norm(dim=-1)-norm[:,0]).abs().max()))
            return base(s,hist,habit,extra)
        reader.read=controlled;trace={k:[] for k in ['s','g','picture','att']};profiles=torch.zeros((END-AT+99)//100,B,4096,device='cuda');replayS=0.;replayP=0.
        for t in range(AT,END):
            frame[0]=t-AT;rec=life.step(H[t-256:t-255].expand(B,-1))
            for k in trace:trace[k].append(rec[k].cpu().numpy())
            if arm=='full':
                replayS=max(replayS,float((rec['s']-torch.tensor(priorS[t],device='cuda')).abs().max()));replayP=max(replayP,float((rec['picture']-torch.tensor(priorP[t],device='cuda')).abs().max()))
            attended=torch.einsum('bl,bld->bd',rec['att'],life.hist);center=(attended-attended.mean(0,keepdim=True))/attended.square().mean(-1,keepdim=True).sqrt();profiles[(t-AT)//100]+=center/100
        data={k:np.stack(v) for k,v in trace.items()};data['profiles']=profiles.half().cpu().numpy()
        p=OUT/f'{cohort}_{arm}.npz';tmp=p.with_suffix('.new')
        with tmp.open('wb') as f:np.savez_compressed(f,**data)
        tmp.replace(p)
        (OUT/f'{cohort}_{arm}.json').write_text(json.dumps(dict(cohort=cohort,arm=arm,reference_stance_error=replayS,reference_picture_error=replayP,maximum_query_norm_mismatch=max(errors) if errors else 0.)))
        if arm=='full':assert replayS==0 and replayP==0
        print(cohort,arm,'done; reference error',replayS,'norm error',max(errors) if errors else 0,flush=True)
        del data,trace,profiles,life,reader;gc.collect();torch.cuda.empty_cache()
(OUT/'DONE').write_text('completed finite continuation controls')
