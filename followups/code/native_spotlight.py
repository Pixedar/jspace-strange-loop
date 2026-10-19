"""One-shot native shared-spotlight check after cached-reader screening.
Not live online loop training: the learned reading biases the native decoder's last
text queries, then the changed native states are read once. All weights are frozen.
"""
import math,json,time,traceback
spot_result={'complete':False,'kind':'one-shot frozen spotlight check; no training or temporal recurrence','rows':[]}
spot_start=time.time()

@torch.no_grad()
def spot_batch(question,image,audio):
    messages=[{'role':'system','content':[{'type':'text','text':'You are a helpful assistant.'}]},
              {'role':'user','content':[{'type':'image','image':'synthetic-image'},{'type':'audio','audio':'synthetic-audio'},{'type':'text','text':question}]}]
    text=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
    batch=processor(text=text,images=[image],audio=[audio],sampling_rate=16000,return_tensors='pt',padding=True).to('cuda')
    for k in batch:
        if torch.is_floating_point(batch[k]):batch[k]=batch[k].to(torch.float16)
    return batch

@torch.no_grad()
def spot_forward(batch,positions,bias=None):
    handles=[]
    if bias is not None:
        def before(module,args,kwargs):
            mask=kwargs.get('attention_mask')
            if mask is None or mask.ndim!=4:raise RuntimeError('Explicit 4D native causal mask required')
            mask=mask.clone()
            for query in range(mask.shape[-2]-9,mask.shape[-2]):
                mask[:,: ,query,positions]+=bias.to(mask.dtype)
            kwargs=dict(kwargs);kwargs['attention_mask']=mask
            return args,kwargs
        for layer in model.model.layers[18:25]:
            handles.append(layer.self_attn.register_forward_pre_hook(before,with_kwargs=True))
    try:return model(**batch,output_hidden_states=True,use_cache=False,return_dict=True)
    finally:
        for handle in handles:handle.remove()

try:
    relative=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-12))
    s=unit(torch.randn(1,8,generator=torch.Generator().manual_seed(870))).cuda()
    for pi,pair in enumerate(palette):
        for reverse in (False,True):
            for count in (1,2):
                if time.time()-spot_start>600:raise RuntimeError('Ten-minute one-shot limit')
                key=f'p{pi}_r{int(reverse)}_n{count}_joint'
                positions=torch.tensor(intake_positions[key],device='cuda')
                batch=spot_batch(rule,picture(pair,reverse),sound(count))
                base=spot_forward(batch,positions)
                h0=base.hidden_states[read_index][0,positions].float()[None]
                l0=base.logits[0,-1].float()
                zero=spot_forward(batch,positions,torch.zeros(16,device='cuda'))
                zero_error=float((zero.logits[0,-1].float()-l0).abs().max()); del zero
                for condition in ('self','recorded_other'):
                    f=intake_models[('balanced',0,condition)].eval()
                    p0,a=f.read(s,h0)
                    bias=1.5*torch.clamp(torch.log(a[0]*16+1e-12),-math.log(16),math.log(16))
                    for arm,b in [('own',bias),('permuted',bias.flip(0))]:
                        out=spot_forward(batch,positions,b)
                        h1=out.hidden_states[read_index][0,positions].float()[None]
                        p1,_=f.read(s,h1)
                        l1=out.logits[0,-1].float();lp0=l0.log_softmax(-1);lp1=l1.log_softmax(-1)
                        row={'key':key,'condition':condition,'arm':arm,'zero_bias_max_logit_error':zero_error,
                             'kl_to_no_spotlight':float((lp0.exp()*(lp0-lp1)).sum()),
                             'top_prediction_changed':bool(l1.argmax()!=l0.argmax()),
                             'read_states_relative_change':relative(h1,h0),
                             'picture_relative_change':relative(p1,p0),
                             'control_relative_change':relative(f.nxt(p1),f.nxt(p0)),
                             'reader_mass':{'image':float(a[0,:4].sum()),'audio':float(a[0,4:8].sum()),'text':float(a[0,8:].sum())}}
                        spot_result['rows'].append(row)
                        del out,h1,p1,l1
                del base,h0,l0,batch
                print('SPOT_PROGRESS',key,flush=True)
    spot_result['complete']=True
except Exception:
    spot_result['error']=traceback.format_exc()
spot_result['elapsed_seconds']=time.time()-spot_start
(screen_dir/'native_spotlight.json').write_text(json.dumps(spot_result,indent=2))
print('SPOT_JSON_BEGIN\n'+json.dumps(spot_result)+'\nSPOT_JSON_END',flush=True)
print('SPOT_FINISHED',spot_result['complete'],flush=True)
