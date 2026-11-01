"""Information-driven recurrent controls through the SAME attention/content interface."""
import os,pathlib,sys,json,time,gc,traceback,math
os.environ['HF_HOME']='/workspace/hf';os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np,torch
ROOT=pathlib.Path('/workspace/consolidation');sys.path[:0]=[str(ROOT/'src'),str(ROOT/'inputs')]
from reader import Reader,Life,unit
from base_cloud import load_model,sample,save,progress
from specificity_reference import Host,matched_add,cache_repeat
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
SPEC=json.loads((ROOT/'inputs/specificity_spec.json').read_text());KINDS=['full','recurrent_forecast','recurrent_selector'];OUT=ROOT/'results'
PARAM=torch.load(ROOT/'inputs/recurrent_state.pt',map_location='cuda',weights_only=False)

class Selector:
 def __init__(self,snapshot,kind,reference=None):
  self.reader=Reader(ROOT/'src/c8_reference.pt').cuda();self.life=Life(self.reader).restore(snapshot);self.kind=kind;self.reference=reference;self.frame=0;self.history=None;self.content=None;self.trace=[];self.query_norms=[];self.hidden=PARAM['hidden'].clone()
  if kind!='full':self.life.config.update(schema='none',freeze_at=0)
  if kind=='recurrent_selector':self.life.config.update(freeze_habit_at=0)
  self.original=self.reader.read
  def read(s,hist,habit,extra):
   if kind=='full':
    dynamic=self.reader.rq(s).view(4,4,64)
    if habit is not None:dynamic=dynamic+torch.einsum('nhdm,nm->nhd',habit,s)
    if extra is not None:dynamic=dynamic+extra
    self.query_norms.append(dict(total=dynamic.flatten(1).norm(dim=-1).detach(),forecast=torch.zeros(4,device='cuda') if extra is None else extra.flatten(1).norm(dim=-1).detach()))
    return self.original(s,hist,habit,extra)
   keys=self.reader.rk(self.reader.rln(hist)).mean(1);z=4*unit(keys@self.life.proj.T);self.hidden=torch.tanh(self.hidden@PARAM['W'].T+.15*z@PARAM['I'].T);direction=unit(self.hidden@PARAM['V'].T)
   target=self.reference.query_norms[self.frame]['forecast' if kind=='recurrent_forecast' else 'total'];bias=(direction*target[:,None]).view(4,4,64)
   if kind=='recurrent_forecast':return self.original(s,hist,habit,bias)
   # Cancel rq(0), retaining exactly rq_bias and positional bias as the fixed baseline.
   zeros=torch.zeros_like(s);return self.original(zeros,hist,None,bias-self.reader.rq(zeros).view(4,4,64))
  self.reader.read=read
 def update(self,enter,read):
  self.history=enter if self.history is None else torch.cat([self.history,enter],1)[:,-16:]
  rec=self.life.step(read);l=self.history.shape[1];key=self.reader.rk(self.reader.rln(self.life.hist[:,-l:])).view(4,l,4,64);q=self.reader.rq_bias.view(1,4,64).expand(4,-1,-1)
  a0=(torch.einsum('bhd,blhd->bhl',q,key)/8+self.reader.rpos[-l:][None,None]).softmax(-1).mean(1);self.content=torch.einsum('bl,bld->bd',rec['att'][:,-l:]-a0,self.history)
  if self.frame%4==0:self.trace.append(dict(t=self.frame,attention=rec['att'].cpu().tolist(),picture=rec['picture'].cpu().tolist(),stance=rec['s'].cpu().tolist(),recurrent_state=None if self.kind=='full' else self.hidden.cpu().tolist()))
  self.frame+=1;return rec

def selectors(snapshot):
 full=Selector(snapshot,'full');return [full,Selector(snapshot,'recurrent_forecast',full),Selector(snapshot,'recurrent_selector',full)]

def action_rows(objects,scale):
 full=objects[0].content*scale;norm=full.norm(dim=-1,keepdim=True);return torch.cat([full]+[unit(s.content)*norm for s in objects[1:]]+[torch.zeros_like(full[:1])])

@torch.inference_mode()
def common(model,tok,snapshot):
 host=Host(model);host.layer=16
 for scene in SPEC['scenes']:
  progress('recurrent_common',scene=scene['id']);ss=selectors(snapshot);encoded=tok.encode(scene['probe_text']);fact_end=len(tok.encode(scene['probe_text'].split('\nA possible')[0]));positions=sorted(set([min(len(encoded)-1,fact_end),min(len(encoded)-1,fact_end+12)]));cache=None;scale=0.;deltas=[];norms=[];rows=[]
  for t,token in enumerate(encoded):
   if t in positions:
    oldlen=cache.get_seq_length();host.action=action_rows(ss,scale);target=host.action.norm(dim=-1);copied=cache_repeat(cache,13,model.config);y=model(torch.full((13,1),token,device='cuda',dtype=torch.long),past_key_values=copied,use_cache=True);logits=y.logits[:,-1].float();p=logits.softmax(-1);lp=logits.log_softmax(-1);kl=(p*(lp-lp[-1:])).sum(-1);ret=(host.read[:,-1]-host.read[-1:,-1]).norm(dim=-1);actual=host.actual_norm.clone()
    assert cache.get_seq_length()==oldlen
    for j,kind in enumerate(KINDS):
     for c in range(4):rows.append(dict(scene=scene['id'],family=scene['family'],position=t,fact_end=fact_end,condition=kind,copy=c,prediction_kl=float(kl[j*4+c]),processed_return_norm=float(ret[j*4+c]),effective_action_norm=float(actual[j*4+c]),target_action_norm=float(target[j*4+c])))
    if t==positions[-1]:np.savez_compressed(OUT/f'recurrent_return_{scene["id"]}.npz',h24=host.read[:,-1].cpu().numpy(),baseline=host.read[-1:,-1].cpu().numpy())
    del copied,y;host.action=None
   y=model(torch.tensor([[token]],device='cuda'),past_key_values=cache,use_cache=True);cache=y.past_key_values
   if 32<=t<64:deltas.append(host.delta)
   for s in ss:s.update(host.enter.expand(4,-1,-1),host.read[:,-1].expand(4,-1))
   if 32<=t<64:norms.append(float(ss[0].content.norm(dim=-1).mean()))
   if t==63:scale=.5*float(np.median(deltas))/max(float(np.median(norms)),1e-8)
  save(f'recurrent_common_{scene["id"]}.json',dict(scene=scene,rows=rows,positions=positions,matching='same native context/cache, actual action magnitude matched; all three actions select real model content through the same writer'))
  del ss,cache,y;gc.collect();torch.cuda.empty_cache()
 host.close()

class Batch:
 def __init__(self,model,snapshot):
  self.model=model;self.ss=selectors(snapshot);self.i=0;self.scale=None;self.deltas=[];self.norms=[];self.norm_trace=[]
  def pre(m,args):
   h=args[0];self.enter=h.detach().float()
   if self.i<64:return None
   actions=action_rows(self.ss,self.scale);changed,actual=matched_add(h,actions);self.norm_trace.append(dict(t=self.i,actual=actual[:12].reshape(3,4).cpu().tolist()));return(changed,)+args[1:]
  def after16(m,a,o):self.delta=float(((o[0] if isinstance(o,tuple) else o).detach().float()-self.enter).norm(dim=-1).mean())
  def after24(m,a,o):self.read=(o[0] if isinstance(o,tuple) else o).detach().float()
  self.hooks=[model.model.layers[16].register_forward_pre_hook(pre),model.model.layers[16].register_forward_hook(after16),model.model.layers[24].register_forward_hook(after24)]
 def update(self):
  for j,s in enumerate(self.ss):s.update(self.enter[4*j:4*j+4],self.read[4*j:4*j+4,-1])
  if 32<=self.i<64:self.deltas.append(self.delta);self.norms.append(float(self.ss[0].content.norm(dim=-1).mean()))
  if self.i==63:self.scale=.5*float(np.median(self.deltas))/max(float(np.median(self.norms)),1e-8)
  self.i+=1
 def close(self):
  for h in self.hooks:h.remove()

@torch.inference_mode()
def generation(model,tok,snapshot):
 for scene in SPEC['scenes']:
  for seed in [202,303]:
   if time.time()>SPEC['lease_deadline']-240:save('recurrent_budget_stop.json',dict(scene=scene['id'],seed=seed));return
   progress('recurrent_generation',scene=scene['id'],seed=seed);c=Batch(model,snapshot);prompt=tok.encode(tok.apply_chat_template([dict(role='user',content=scene['prompt'])],tokenize=False,add_generation_prompt=True,enable_thinking=False));assert len(prompt)>=64
   cache=None;last=None;generated=[];finished=torch.zeros(13,dtype=torch.bool,device='cuda');rng=np.random.default_rng(seed)
   for token in prompt:
    y=model(torch.full((13,1),token,device='cuda',dtype=torch.long),past_key_values=cache,use_cache=True);cache=y.past_key_values;last=y.logits[:,-1];c.update()
   for t in range(220):
    token=sample(last,float(rng.random()));token=torch.where(finished,tok.eos_token_id,token);generated.append(token.cpu().numpy());finished|=token==tok.eos_token_id;y=model(token[:,None],past_key_values=cache,use_cache=True);cache=y.past_key_values;last=y.logits[:,-1];c.update()
    if t%40==0:progress('recurrent_tokens',scene=scene['id'],seed=seed,t=t)
   ids=np.stack(generated).T.tolist();texts=[]
   for seq in ids:
    if tok.eos_token_id in seq:seq=seq[:seq.index(tok.eos_token_id)]
    texts.append(tok.decode(seq,skip_special_tokens=True))
   for j,kind in enumerate(KINDS):save(f'recurrent_generation_{scene["id"]}_{seed}_{kind}.json',dict(scene=scene,seed=seed,condition=kind,texts=texts[4*j:4*j+4]+texts[-1:],token_ids=ids[4*j:4*j+4]+ids[-1:],trace=c.ss[j].trace,action_norms=[dict(t=x['t'],actual=x['actual'][j]) for x in c.norm_trace],prompt_tokens=len(prompt),rows='four selected source histories and one unchanged model',matching='One 13-row native forward pass; same sampling numbers. Recurrent conditions receive per-copy/time full-controller query and action magnitudes, while directions come from their own recurrent context filter.',limits='Yoked magnitude is an experimenter-supplied control, not a standalone deployment. Only full and recurrent_forecast use stance/habit causally; recurrent_selector retains those signals as unused observers. Post-EOS measurements excluded.'))
   c.close();del c,cache,y;gc.collect();torch.cuda.empty_cache()

def main():
 snapshot=torch.load(ROOT/'inputs/selected_long.pt',map_location='cpu',weights_only=False);model,tok=load_model('Qwen/Qwen3-8B')
 # Zero action must preserve the same native thirteen-row computation exactly.
 ids=tok('A careful comparison changes one relationship at a time.',return_tensors='pt').input_ids.cuda().repeat(13,1);a=model(ids,use_cache=False).logits;hook=model.model.layers[16].register_forward_pre_hook(lambda m,x:(x[0]+torch.zeros_like(x[0]),)+x[1:]);b=model(ids,use_cache=False).logits;hook.remove();error=float((a-b).abs().max());assert error==0;save('recurrent_null_qualification.json',dict(passed=True,max_error=error,batch=13))
 common(model,tok,snapshot);generation(model,tok,snapshot);save('recurrent_finished.json',dict(scenes=16,seeds=2,conditions=KINDS,controller_outputs=384,scope='Causal specificity of learned attention prediction versus an untrained information-driven recurrent filter and a whole recurrent selector. No consciousness score.'));(ROOT/'QUEUE_DONE').write_text('completed')
if __name__=='__main__':
 try:main()
 except Exception:save('failure.json',dict(traceback=traceback.format_exc()));(ROOT/'QUEUE_DONE').write_text('failed; preserve');raise
