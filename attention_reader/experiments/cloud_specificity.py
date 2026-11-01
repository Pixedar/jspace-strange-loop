"""Matched-context action specificity, insertion layers, and closed-loop writing.
The common-context assay separates representation alignment from noise and
ordinary memory. It does not measure recognized ownership or consciousness.
"""
import os,pathlib,sys,json,time,copy,gc,traceback
os.environ['HF_HOME']='/workspace/hf';os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np,torch
from torch import nn
from transformers import DynamicCache
ROOT=pathlib.Path('/workspace/consolidation');sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'inputs'))
from reader import Reader,Life,unit
from base_cloud import load_model,sample,progress,save
from base_revision import Revised
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
SPEC=json.loads((ROOT/'inputs/specificity_spec.json').read_text())
CONDITIONS=['full','unused_schema','mean_memory','random_act','rotated_act','fixed_act','neighbor_act']

def clone_life(life):
    obj=Life.__new__(Life)
    obj.__dict__={k:(v.clone() if isinstance(v,torch.Tensor) else (v if k in ['reader','dev'] else copy.deepcopy(v))) for k,v in life.__dict__.items()}
    return obj

def cache_repeat(cache,n,config):
    # New cache layers and tensor storage; the host cache must not be mutated by a probe.
    return DynamicCache(ddp_cache_data=[(layer.keys.repeat_interleave(n,0),layer.values.repeat_interleave(n,0)) for layer in cache.layers],config=config)

def matched_add(h,action,target=None):
    target=action.norm(dim=-1) if target is None else target
    if not bool(target.any()):return h,torch.zeros_like(target)
    direction=unit(action);lo=torch.zeros_like(target);hi=torch.full_like(target,2.)
    for _ in range(12):
        mid=(lo+hi)/2;out=h+(direction*(target*mid)[:,None])[:,None].to(h.dtype)
        actual=(out.float()-h.float()).norm(dim=-1)[:,0]
        lo=torch.where(actual<target,mid,lo);hi=torch.where(actual>=target,mid,hi)
    out=h+(direction*(target*((lo+hi)/2))[:,None])[:,None].to(h.dtype)
    return out,(out.float()-h.float()).norm(dim=-1)[:,0]

class Host:
    def __init__(self,model):
        self.model=model;self.action=None;self.layer=16;self.enter=None;self.read=None;self.delta=None
        self.hooks=[]
        for layer in [12,16,20,24,28]:self.hooks.append(model.model.layers[layer].register_forward_pre_hook(self.pre(layer)))
        self.hooks += [model.model.layers[16].register_forward_hook(self.post16),model.model.layers[24].register_forward_hook(self.post24)]
    def pre(self,layer):
        def hook(module,args):
            h=args[0]
            if layer==16:self.enter=h.detach().float()
            if self.action is not None and self.layer==layer:
                changed,self.actual_norm=matched_add(h,self.action);self.actual_delta=(changed.float()-h.float())[:,0];return (changed,)+args[1:]
        return hook
    def post16(self,m,a,o):
        y=o[0] if isinstance(o,tuple) else o;self.delta=float((y.detach().float()-self.enter).norm(dim=-1).mean())
    def post24(self,m,a,o):self.read=(o[0] if isinstance(o,tuple) else o).detach().float()
    def close(self):
        for h in self.hooks:h.remove()

class Shadow:
    def __init__(self,snapshot,kind):
        self.reader=Reader(ROOT/'src/c8_reference.pt').cuda();self.life=Life(self.reader).restore(snapshot)
        self.kind=kind;self.content=None;self.input_history=None;self.last=None
        self.total=snapshot['ordinary_mean']['total'].cuda();self.weight=snapshot['ordinary_mean']['weight'];self.mean_scale=None
        if kind=='unused_schema':self.life.config['schema']='none'
        original=self.reader.read
        def controlled_read(s,hist,habit,extra):
            if kind=='mean_memory':
                pull=self.total/max(self.weight,1e-12)
                if self.mean_scale is None:self.mean_scale=float(.5*self.life.cap/pull.norm(dim=-1).mean().clamp_min(1e-12))
                extra=(self.mean_scale*pull).view(4,4,64)
            result=original(s,hist,habit,extra);self.mean_last=(result[2],result[3]);return result
        self.reader.read=controlled_read
    def update(self,enter,read):
        self.input_history=enter.expand(4,-1,-1) if self.input_history is None else torch.cat([self.input_history,enter.expand(4,-1,-1)],1)[:,-16:]
        self.last=self.life.step(read.expand(4,-1))
        l=self.input_history.shape[1];keys=self.reader.rk(self.reader.rln(self.life.hist[:,-l:])).view(4,l,4,64)
        q=self.reader.rq_bias.view(1,4,64).expand(4,-1,-1)
        a0=(torch.einsum('bhd,blhd->bhl',q,keys)/8+self.reader.rpos[-l:][None,None]).softmax(-1).mean(1)
        self.content=torch.einsum('bl,bld->bd',self.last['att'][:,-l:]-a0,self.input_history)
        a,k=self.mean_last;target=(torch.einsum('nhl,nlhd->nhd',a,k)-k.mean(1)).reshape(4,256)
        self.total=.998*self.total+target;self.weight=.998*self.weight+1

def rotate_parameters(d=4096):
    gen=torch.Generator(device='cuda').manual_seed(9083)
    permutations=torch.stack([torch.randperm(d,generator=gen,device='cuda') for _ in range(4)])
    signs=torch.where(torch.rand((4,d),generator=gen,device='cuda')>.5,1.,-1.)
    fixed=unit(torch.randn((4,d),generator=gen,device='cuda'))
    return permutations,signs,fixed

def direction_controls(raw,unused,mean,index,perms,signs,fixed):
    norms=raw.norm(dim=-1,keepdim=True)
    def normmatch(x):return unit(x)*norms
    gen=torch.Generator(device='cuda').manual_seed(17000+index)
    return dict(full=raw,unused_schema=normmatch(unused),mean_memory=normmatch(mean),
                random_act=normmatch(torch.randn(raw.shape,device='cuda',generator=gen)),
                rotated_act=raw.gather(1,perms)*signs,fixed_act=fixed*norms,
                neighbor_act=normmatch(torch.roll(raw,-1,0)))

def lens(model):
    from huggingface_hub import hf_hub_download
    progress('load_lenses')
    p=hf_hub_download('neuronpedia/jacobian-lens','qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt',revision='e166deb5ad540932100ef803c8c9bcc2a5453694')
    ck=torch.load(p,map_location='cpu',weights_only=True);J15=ck['J'][15].float().cuda();J24=ck['J'][24].float().cuda()
    W=model.get_output_embeddings().weight;gamma=model.model.norm.weight.float();gen=torch.Generator().manual_seed(324)
    ix=torch.randperm(W.shape[0],generator=gen)[:4096].cuda();mean=torch.zeros(W.shape[1],device='cuda')
    for chunk in W.split(4096):mean+=chunk.float().sum(0)
    mean/=W.shape[0];D=((W[ix].float()-mean)*gamma)@J24;D=unit(D)
    torch.manual_seed(324);_,s,v=torch.pca_lowrank(D,q=80,center=False,niter=4);basis=v[:,:64].contiguous()
    save('lens_basis.json',dict(layer=24,rank=64,sampled_rows=4096,dictionary_energy=float((D@basis).square().sum()/D.square().sum()),
          lens_revision='e166deb5ad540932100ef803c8c9bcc2a5453694',
          scope='fixed linear approximation from J-lens directions; not the sparse nonnegative J-space; coordinates have no assigned psychological names'))
    torch.save(dict(basis=basis.cpu()),OUT/'specificity_lens.pt')
    del ck,J24,D;gc.collect();torch.cuda.empty_cache();return J15,basis

@torch.inference_mode()
def common_context(model,tok,snapshot,basis):
    host=Host(model);perms,signs,fixed=rotate_parameters();templates=torch.tensor(SPEC['copy_templates'],device='cuda',dtype=torch.float32)
    contrast=unit(templates-(templates.sum(0,keepdim=True)-templates)/3)
    for scene in SPEC['scenes']:
        if (OUT/f'common_context_{scene["id"]}.json').exists() and (OUT/f'common_states_{scene["id"]}.npz').exists():continue
        shadow={name:Shadow(snapshot,name) for name in ['full','unused_schema','mean_memory']}
        encoded=tok.encode(scene['probe_text']);cache=None;scale=None;norms=[];deltas=[];records=[];states=[];positions=[];host16=[];host24=[]
        for t,token in enumerate(encoded[:-1]):
            if t>=80 and t%16==0:
                actions=direction_controls(shadow['full'].content,shadow['unused_schema'].content,shadow['mean_memory'].content,t,perms,signs,fixed)
                acts=torch.cat([actions[c]*scale for c in CONDITIONS]+[torch.zeros(1,4096,device='cuda')])
                if t==80:
                    source=shadow['full'].life.snapshot();source['ordinary_mean']=dict(total=shadow['full'].total.cpu(),weight=shadow['full'].weight);torch.save(source,OUT/f'source_state_{scene["id"]}.pt')
                    n0=cache.get_seq_length();host.action=torch.zeros_like(acts);host.layer=16
                    nullcache=cache_repeat(cache,len(acts),model.config)
                    y=model(torch.full((len(acts),1),token,device='cuda',dtype=torch.long),past_key_values=nullcache,use_cache=True)
                    err=float((y.logits-y.logits[-1:]).abs().max());assert err==0 and cache.get_seq_length()==n0
                    save(f'null_probe_{scene["id"]}.json',dict(zero_action_batch_max_logit_difference=err,host_cache_length_unchanged=True))
                    del nullcache,y
                for layer in [12,16,20,24,28]:
                    host.action=acts;host.layer=layer;pcache=cache_repeat(cache,len(acts),model.config)
                    y=model(torch.full((len(acts),1),token,device='cuda',dtype=torch.long),past_key_values=pcache,use_cache=True)
                    logits=y.logits[:,-1].float();logp=logits.log_softmax(-1);plain=logp[-1];plainprob=plain.exp();read=host.read[:,-1].clone();baseline_read=read[-1:]
                    states.append((read@basis).half().cpu().numpy());positions.append((t,layer))
                    if t==80 and layer==16:np.savez_compressed(OUT/f'return_h24_{scene["id"]}.npz',h24=read.cpu().numpy(),actions=acts.cpu().numpy(),effective=host.actual_delta.cpu().numpy())
                    # Every row uses the same unacted prefix, input token and source controller state.
                    reference=clone_life(shadow['full'].life);base=reference.step(baseline_read.expand(4,-1));basep=base['picture'];baseatt=base['att']
                    for j,condition in enumerate(CONDITIONS):
                        sl=slice(j*4,j*4+4);test=clone_life(shadow['full'].life);r=test.step(read[sl]);delta=r['picture']-basep
                        kl=(plainprob[None]*(plain[None]-logp[sl])).sum(-1)
                        returnnorm=(read[sl]-baseline_read).norm(dim=-1)
                        for c in range(4):
                            records.append(dict(scene=scene['id'],position=t,layer=layer,condition=condition,copy=c,
                                action_norm=float(acts[j*4+c].norm()),effective_action_norm=float(host.actual_norm[j*4+c]),prediction_kl=float(kl[c]),
                                next_token_logprob_change=float(logp[j*4+c,encoded[t+1]]-plain[encoded[t+1]]),
                                processed_return_norm=float(returnnorm[c]),
                                copy_template_return=float(delta[c]@contrast[c]),
                                attention_change=float(.5*(r['att'][c]-baseatt[c]).abs().sum()),
                                reading_change=float(delta[c].norm()),
                                rotation_direction_cosine=float(torch.nn.functional.cosine_similarity(host.actual_delta[16:20],host.actual_delta[:4].gather(1,perms)*signs)[c]) if condition=='rotated_act' else None,
                                prior_text=tok.decode(encoded[max(0,t-40):t+1]),next_text=tok.decode(encoded[t+1:t+13])))
                        del test
                    del pcache,y,logits,logp,read;gc.collect()
                    progress('common_context_probe',scene=scene['id'],position=t,layer=layer)
                host.action=None
            host.action=None
            y=model(torch.tensor([[token]],device='cuda'),past_key_values=cache,use_cache=True);cache=y.past_key_values
            enter=host.enter.clone();read=host.read[:,-1].clone();delta=host.delta
            for s in shadow.values():s.update(enter,read)
            host16.append(enter[0,-1].half().cpu().numpy());host24.append(read[0].half().cpu().numpy())
            if 32<=t<64:norms.append(float(shadow['full'].content.norm(dim=-1).mean()));deltas.append(delta)
            if t==63:scale=.5*float(np.median(deltas))/max(float(np.median(norms)),1e-8)
        save(f'common_context_{scene["id"]}.json',dict(scene=scene,rows=records,scale=scale,
              matching='same host cache and input; all action norms matched per copy/time; signed coordinate permutation fixed across time',
              limit='one-step counterfactual communication assay on unacted contexts; not a closed-loop developmental result'))
        np.savez_compressed(OUT/f'common_states_{scene["id"]}.npz',J24=np.stack(states),positions=np.array(positions),host16=np.stack(host16),host24=np.stack(host24),token_ids=np.array(encoded))
        del cache,shadow;gc.collect();torch.cuda.empty_cache()
    host.close()

class PatternCoupled(Revised):
    def __init__(self,model,reader,snapshot,condition,basis,norm_schedule=None):
        super().__init__(model,reader,snapshot,condition,basis=None)
        self.B24=basis;self.perms,self.signs,self.fixed=rotate_parameters();self.schedule=norm_schedule;self.used_norms=[];self.model_trace=[]
        self.total=snapshot['ordinary_mean']['total'].cuda();self.mean_weight=snapshot['ordinary_mean']['weight'];self.mean_scale=None;self.orig_read=reader.read
        if condition=='mean_memory':
            def avg(s,hist,habit,extra):
                pull=self.total/max(self.mean_weight,1e-12)
                if self.mean_scale is None:self.mean_scale=float(.5*self.life.cap/pull.norm(dim=-1).mean().clamp_min(1e-12))
                result=self.orig_read(s,hist,habit,(self.mean_scale*pull).view(4,4,64));self.mean_last=(result[2],result[3]);return result
            reader.read=avg
    def pre(self,module,args):
        h=args[0];self.enter=h.detach().float()
        if self.content is None:return None
        raw=self.content*(self.scale if self.scale is not None else 0)
        if self.condition=='random_act':
            gen=torch.Generator(device='cuda').manual_seed(17000+self.i);act=unit(torch.randn(raw.shape,generator=gen,device='cuda'))*raw.norm(dim=-1,keepdim=True)
        elif self.condition=='rotated_act':act=raw.gather(1,self.perms)*self.signs
        elif self.condition=='fixed_act':act=self.fixed*raw.norm(dim=-1,keepdim=True)
        else:act=raw
        if self.schedule is not None:
            direction=torch.where(act.norm(dim=-1,keepdim=True)<1e-8,self.fixed,unit(act));act=direction*torch.tensor(self.schedule[self.i],device='cuda')[:,None]
        allact=torch.cat([act,torch.zeros_like(act[:1])]);changed,actual=matched_add(h,allact)
        self.used_norms.append(actual[:4].cpu().tolist())
        return (changed,)+args[1:]
    def update(self):
        rec=super().update()
        if self.condition=='mean_memory':
            a,k=self.mean_last;target=(torch.einsum('nhl,nlhd->nhd',a,k)-k.mean(1)).reshape(4,256);self.total=.998*self.total+target;self.mean_weight=.998*self.mean_weight+1
        if (self.i-1)%4==0:self.model_trace.append((self.read[:,-1]@self.B24).half().cpu().numpy())
        return rec
    def close(self):super().close();self.reader.read=self.orig_read

@torch.inference_mode()
def generation(model,tok,snapshot,basis):
    for item in SPEC['scenes']:
      for seed in [202,303]:
        schedule=None
        for condition in ['full','unused_schema','mean_memory','random_act','rotated_act','fixed_act']:
            path=OUT/f'pattern_generation_{item["id"]}_{seed}_{condition}.json'
            if path.exists():
                if condition=='full':schedule=json.loads(path.read_text())['action_norms']
                continue
            progress('pattern_generation',scene=item['id'],seed=seed,condition=condition)
            reader=Reader(ROOT/'src/c8_reference.pt').cuda();c=PatternCoupled(model,reader,snapshot,condition,basis,norm_schedule=schedule)
            text=tok.apply_chat_template([dict(role='user',content=item['prompt'])],tokenize=False,add_generation_prompt=True,enable_thinking=False);prompt=tok.encode(text)
            cache=None;last=None;generated=[];rng=np.random.default_rng(seed);finished=torch.zeros(5,dtype=torch.bool,device='cuda')
            for token in prompt:
                y=model(torch.full((5,1),token,device='cuda',dtype=torch.long),past_key_values=cache,use_cache=True);cache=y.past_key_values;last=y.logits[:,-1];c.update()
            for t in range(220):
                token=sample(last,float(rng.random()));token=torch.where(finished,tok.eos_token_id,token);generated.append(token.cpu().numpy());finished|=token==tok.eos_token_id
                y=model(token[:,None],past_key_values=cache,use_cache=True);cache=y.past_key_values;last=y.logits[:,-1];c.update()
                if t%40==0:progress('pattern_tokens',scene=item['id'],seed=seed,condition=condition,t=t)
            if condition=='full':
                # The first pre-hook has no content. Align norm schedule with c.i exactly.
                schedule=[[0.,0.,0.,0.]]+c.used_norms
            ids=np.stack(generated).T.tolist();texts=[]
            for seq in ids:
                if tok.eos_token_id in seq:seq=seq[:seq.index(tok.eos_token_id)]
                texts.append(tok.decode(seq,skip_special_tokens=True))
            save(f'pattern_generation_{item["id"]}_{seed}_{condition}.json',dict(scene=item,seed=seed,condition=condition,texts=texts,token_ids=ids,
              trace=c.trace,action_norms=[[0.,0.,0.,0.]]+c.used_norms,prompt_tokens=len(prompt),max_new_tokens=220,
              matching='all conditions yoked to full-condition per-copy/time action norms; common sampling; no changes to model weights',
              limit='after token divergence, contexts differ; exact temporal-geometry matching applies to the common-context assay, not across these closed-loop trajectories'))
            np.savez_compressed(OUT/f'pattern_J24_{item["id"]}_{seed}_{condition}.npz',J24=np.stack(c.model_trace))
            c.close();del cache,c,y,reader;gc.collect();torch.cuda.empty_cache()

def main():
    snapshot=torch.load(ROOT/'inputs/selected_long.pt',map_location='cpu',weights_only=False)
    started=time.time();model,tok=load_model('Qwen/Qwen3-8B');J15,B24=lens(model)
    common_context(model,tok,snapshot,B24)
    generation(model,tok,snapshot,B24)
    del model,tok,J15,B24;gc.collect();torch.cuda.empty_cache()
    if time.time()-started<5600:
        model,tok=load_model('Qwen/Qwen3-14B');from criterion_judge import run as constraint_review;constraint_review(model,tok)
    save('specificity_finished.json',dict(scenes=len(SPEC['scenes']),sampling_seeds=[202,303],generation_conditions=6,controller_outputs=4*len(SPEC['scenes'])*2*6,
      source=SPEC['source'],conclusion='Inspect matched contrasts before assigning a self-model-specific interpretation.'))
    (ROOT/'QUEUE_DONE').write_text('completed')
if __name__=='__main__':
    try:main()
    except Exception:
        save('failure.json',dict(traceback=traceback.format_exc()));(ROOT/'QUEUE_DONE').write_text('failed; preserve partial evidence');raise
