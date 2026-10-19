"""Frozen input interventions and factorial response analysis; no further training."""
import json,traceback,io,base64
modality_result={'complete':False,'kind':'frozen full-native input interventions','rows':[]}
modality_features={k:v for k,v in balanced_features.items() if k.endswith('_balanced')}
modality_arrays={}
try:
    # Full native forwards preserve normal encoders and contextualization under each input intervention.
    for pi,pair in enumerate(palette):
        for reverse in (False,True):
            _,h,_,_=paired_intake(rule,picture(pair,reverse),sound(0))
            modality_features[f'p{pi}_r{int(reverse)}_silent']=h
    for count in (1,2):
        _,h,_,_=paired_intake(rule,picture(palette[0],False,grey=True),sound(count))
        modality_features[f'grey_n{count}']=h
    S=unit(torch.randn(32,8,generator=torch.Generator().manual_seed(4242))).cuda()
    for seed in (0,1):
        for condition in ('self','recorded_other'):
            f=intake_models[('balanced',seed,condition)].eval()
            fields={};pictures={};mass={}
            with torch.no_grad():
                for key,h in modality_features.items():
                    ht=torch.tensor(h,device='cuda')[None].expand(32,-1,-1)
                    p,a=f.read(S,ht)
                    fields[key]=f.nxt(p).cpu().numpy()
                    pictures[key]=p.cpu().numpy()
                    mass[key]=a.mean(0).cpu().numpy()
                    modality_arrays[f's{seed}_{condition}_{key}_dest']=fields[key]
                for pi in (0,1):
                    ds=np.stack([fields[f'p{pi}_r{r}_n{n}_joint_balanced'] for r in (0,1) for n in (1,2)]).reshape(2,2,32,8)
                    image=float(np.sqrt(np.mean((ds[1]-ds[0])**2)))
                    audio=float(np.sqrt(np.mean((ds[:,1]-ds[:,0])**2)))
                    interaction=float(np.sqrt(np.mean((ds[1,1]-ds[1,0]-ds[0,1]+ds[0,0])**2)))
                    grey=np.stack([fields[f'grey_n{n}'] for r in (0,1) for n in (1,2)]).reshape(2,2,32,8)
                    silent=np.stack([fields[f'p{pi}_r{r}_silent'] for r in (0,1) for n in (1,2)]).reshape(2,2,32,8)
                    attention=np.stack([mass[f'p{pi}_r{r}_n{n}_joint_balanced'] for r in (0,1) for n in (1,2)]).mean(0)
                    modality_result['rows'].append({'seed':seed,'condition':condition,'palette':pi,
                        'image_reversal_destination_rms':image,'spoken_direction_destination_rms':audio,
                        'factorial_destination_interaction_rms':interaction,
                        'grey_input_destination_rms':float(np.sqrt(np.mean((ds-grey)**2))),
                        'silence_input_destination_rms':float(np.sqrt(np.mean((ds-silent)**2))),
                        'mean_reader_mass':{'image':float(attention[:4].sum()),'audio':float(attention[4:8].sum()),'text':float(attention[8:].sum())}})
    modality_result['complete']=True
except Exception:modality_result['error']=traceback.format_exc()
(screen_dir/'modality_check.json').write_text(json.dumps(modality_result,indent=2))
np.savez_compressed(screen_dir/'modality_features.npz',**modality_features)
np.savez_compressed(screen_dir/'modality_fields.npz',**modality_arrays)
print('MODALITY_JSON_BEGIN\n'+json.dumps(modality_result)+'\nMODALITY_JSON_END',flush=True)
buf=io.BytesIO();np.savez_compressed(buf,**modality_arrays)
print('MODALITY_FINISHED',modality_result['complete'],flush=True)
