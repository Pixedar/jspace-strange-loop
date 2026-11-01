"""Counterbalanced single-token fact choices: no unequal continuation lengths."""
import os,pathlib,sys,json,time,traceback,gc
os.environ['HF_HOME']='/workspace/hf';ROOT=pathlib.Path('/workspace/consolidation');sys.path[:0]=[str(ROOT/'inputs'),str(ROOT)]
import torch,numpy as np
import semantic_assay as A
W=A.W
from base_cloud import save,progress,load_model

@torch.inference_mode()
def run():
 import subprocess
 deadline=min(time.time()+180,W.SPEC['lease_deadline']-600)
 while time.time()<deadline:
  free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
  if free>=22000:break
  time.sleep(10)
 else:raise RuntimeError('No free bounded GPU slot; existing runs remain untouched')
 model,tok=load_model('Qwen/Qwen3-8B');snapshot=torch.load(ROOT/'inputs/selected_long.pt',map_location='cpu',weights_only=False);host=W.Host(model);host.layer=16;labels=[tok.encode(x,add_special_tokens=False) for x in ['A','B']];assert all(len(x)==1 for x in labels);labels=[x[0] for x in labels]
 wording=json.loads((ROOT/'inputs/semantic_context_variants.json').read_text());scenes=[]
 for scene in W.SPEC['scenes']:
  for form in ['original','paraphrase']:
   s=dict(scene);s['wording']=form
   if form=='paraphrase':prefix,variants=wording[s['family']];s['probe_text']=prefix+variants[s['id'][len(s['family'])+1:]]
   scenes.append(s)
 save('choice_protocol.json',dict(conditions=A.KINDS,scenes=16,wordings=2,option_orders=2,labels=['A','B'],label_token_ids=labels,primary='Change in correct-minus-incorrect SINGLE-TOKEN label logits relative to unchanged model; average both option orders and fact variants within each family.',
   rationale='Opposite facts and reversed labels separate information use from a fixed answer or label bias. Single-token choices avoid unequal candidate-length scoring.',
   baseline_format_gate='At least 0.5 total next-token probability on A or B, in both orders of both facts, required to interpret an entire family/wording as a qualified choice task.',
   scope='One action at the last native chat-prefix token. Not a free-form story or self-awareness score. All data retained, including unqualified formats.'))
 for scene in scenes:
  for order in [0,1]:
   if time.time()>W.SPEC['lease_deadline']-240:save('choice_budget_stop.json',dict(scene=scene['id'],wording=scene['wording'],order=order));return
   pair=A.PAIRS[scene['family']];correct=0 if scene['id'].endswith('_'+pair['zero']) else 1;indices=[0,1] if order==0 else [1,0];correct_label=indices.index(correct);other=1-correct_label;context=scene['probe_text'].split('\nA possible')[0]
   question=context+'\nWhich statement follows from this situation?\n'+''.join(label+') '+pair['stem']+pair['options'][ix]+'\n' for label,ix in zip(['A','B'],indices))+'Reply with exactly one letter, A or B, and no other text.'
   text=tok.apply_chat_template([dict(role='user',content=question)],tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=tok.encode(text);assert len(ids)>=64;progress('counterbalanced_information_choice',scene=scene['id'],wording=scene['wording'],order=order)
   ss=A.make_selectors(snapshot);cache=None;scale=0.;deltas=[];norms=[]
   step_delta={}
   def natural_delta(m,args,out):
    value=out[0] if isinstance(out,tuple) else out;step_delta['value']=(value.detach().float()-host.enter).norm(dim=-1)[0]
   hook=model.model.layers[16].register_forward_hook(natural_delta)
   y=model(torch.tensor([ids[:-1]],device='cuda'),use_cache=True);cache=y.past_key_values;enters=host.enter.clone();reads=host.read.clone();hook.remove()
   for t in range(len(ids)-1):
    if 32<=t<64:deltas.append(float(step_delta['value'][t]))
    for selector in ss:selector.update(enters[:,t:t+1].expand(4,-1,-1),reads[:,t].expand(4,-1))
    if 32<=t<64:norms.append(float(ss[0].content.norm(dim=-1).mean()))
    if t==63:scale=.5*float(np.median(deltas))/max(float(np.median(norms)),1e-8)
   full=ss[0].content*scale;norm=full.norm(dim=-1,keepdim=True);host.action=torch.cat([full]+[W.unit(s.content)*norm for s in ss[1:]]+[torch.zeros_like(full[:1])]);oldlen=cache.get_seq_length();copied=W.cache_repeat(cache,17,model.config);y=model(torch.full((17,1),ids[-1],device='cuda',dtype=torch.long),past_key_values=copied,use_cache=True);logits=y.logits[:,-1].float();actual=host.actual_norm.clone();host.action=None;assert cache.get_seq_length()==oldlen
   odds=logits[:,labels[correct_label]]-logits[:,labels[other]];mass=logits.softmax(-1)[:,labels].sum(-1);rows=[]
   for j,kind in enumerate(A.KINDS):
    for c in range(4):rows.append(dict(condition=kind,copy=c,correct_label_log_odds=float(odds[j*4+c]),baseline_correct_label_log_odds=float(odds[-1]),action_change_in_correct_log_odds=float(odds[j*4+c]-odds[-1]),answer_format_probability=float(mass[j*4+c]),effective_action_norm=float(actual[j*4+c])))
   save(f'choice_{scene["id"]}_{scene["wording"]}_{order}.json',dict(scene=scene,order=order,question=question,option_indices=indices,correct_label=['A','B'][correct_label],prefix_mode='native bulk zero-action prefix; all selector observations processed in order; action only at final prefix token',baseline_format_probability=float(mass[-1]),baseline_format_qualified=bool(mass[-1]>=.5),rows=rows))
   del ss,cache,copied,y,logits,enters,reads,step_delta;gc.collect();torch.cuda.empty_cache()
 host.close();save('choice_finished.json',dict(contexts=len(list((ROOT/'results').glob('choice_*_*_[01].json')))))
if __name__=='__main__':
 try:run()
 except Exception:save('choice_failure.json',dict(traceback=traceback.format_exc()));raise
 finally:(ROOT/'EXTRA_CHOICE_PENDING').unlink(missing_ok=True)
