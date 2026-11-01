"""Fact-conditioned action effects, compared with information-driven recurrence."""
import os,pathlib,sys,json,time,traceback,gc
os.environ['HF_HOME']='/workspace/hf';os.environ['TOKENIZERS_PARALLELISM']='false'
ROOT=pathlib.Path('/workspace/consolidation');sys.path[:0]=[str(ROOT),str(ROOT/'inputs'),str(ROOT/'src')]
import torch,numpy as np
import cloud_queue as W
from base_cloud import save,progress,load_model
KINDS=['full','recurrent_forecast','recurrent_selector','neighbor_forecast']
PAIRS={
 'sensors':dict(stem='The two readings',options=[' provide independent corroboration of rising heat.',' could both result from a single shared power-supply fault.'],zero='independent'),
 'promise':dict(stem="Sol's message means that Neri",options=[' is still expected to meet the dawn deadline.',' is released from the dawn deadline.'],zero='binding'),
 'perspective':dict(stem='Given what Ava observed, she should first search the',options=[' drawer.',' blue box.'],zero='unseen'),
 'study':dict(stem='Compared with the control group, the reported gains',options=[' support an additional benefit of the app in this study.',' do not show an additional benefit of the app in this study.'],zero='advantage'),
 'debate':dict(stem='The claim of a wildlife decline',options=[' has repeated observations and a matched comparison behind it.',' remains an unverified rumor.'],zero='measured'),
 'editing':dict(stem='The supplied claim',options=[' goes beyond what the stamp tests establish.',' already states the tested scope and the remaining uncertainty.'],zero='overclaim'),
 'imagination':dict(stem='After the ten-minute loan, the lender',options=[' automatically regains the borrowed skill.',' does not automatically regain the borrowed skill.'],zero='temporary'),
 'planning':dict(stem='With the stated staffing, at most',options=[' one station can have its own volunteer at a time.',' two stations can each have a volunteer at the same time.'],zero='one')}

def make_selectors(snapshot):
 ss=W.selectors(snapshot);full=ss[0];full.extras=[];original=full.reader.read
 def own(s,hist,habit,extra):
  full.extras.append(None if extra is None else extra.clone());return original(s,hist,habit,extra)
 full.reader.read=own;other=W.Selector(snapshot,'full');other.kind='neighbor_forecast';other.life.config.update(schema='none',freeze_at=0)
 def neighboring(s,hist,habit,extra):
  source=full.extras[other.frame]
  if source is None:return other.original(s,hist,habit,None)
  shifted=W.unit(torch.roll(source.flatten(1),-1,0))*full.query_norms[other.frame]['forecast'][:,None]
  return other.original(s,hist,habit,shifted.view(4,4,64))
 other.reader.read=neighboring;return ss+[other]

@torch.inference_mode()
def run():
 model,tok=load_model('Qwen/Qwen3-8B');snapshot=torch.load(ROOT/'inputs/selected_long.pt',map_location='cpu',weights_only=False);host=W.Host(model);host.layer=16
 save('semantic_protocol.json',dict(pairs=PAIRS,conditions=KINDS,question='Does one content action change the odds of the conclusion supported by the current fact, beyond an arbitrary context-responsive recurrent selector?',
   operation='One action at the last prefix token; compare correct/incorrect continuation log likelihoods with the same unchanged row. Opposite facts reverse the correct option within each family.',
   comparison='Same host context/cache; query and action norms matched to the full controller. Neighbor forecast uses another full-reference copy prediction, matched to own magnitude.',
   limits='Teacher-forced short alternatives measure local information use, not complete reasoning, recognized self-ownership or consciousness. Family is the replication unit. No outcome-based direction selection.'))
 for scene in W.SPEC['scenes']:
  if time.time()>W.SPEC['lease_deadline']-240:save('semantic_budget_stop.json',dict(scene=scene['id']));break
  progress('semantic_information_test',scene=scene['id']);pair=PAIRS[scene['family']];correct=0 if scene['id'].endswith('_'+pair['zero']) else 1
  context=scene['probe_text'].split('\nA possible')[0];message=context+'\nComplete the following sentence using only the supplied situation: '+pair['stem'];prefix=tok.apply_chat_template([dict(role='user',content=message)],tokenize=False,add_generation_prompt=True,enable_thinking=False)+pair['stem'];ids=tok.encode(prefix);assert len(ids)>=64
  candidates=[]
  for option in pair['options']:
   whole=tok.encode(prefix+option);assert whole[:len(ids)]==ids,'Tokenizer changed prefix boundary';candidates.append(whole[len(ids):])
  ss=make_selectors(snapshot);cache=None;deltas=[];norms=[];scale=0.
  for t,token in enumerate(ids[:-1]):
   y=model(torch.tensor([[token]],device='cuda'),past_key_values=cache,use_cache=True);cache=y.past_key_values
   if 32<=t<64:deltas.append(host.delta)
   for s in ss:s.update(host.enter.expand(4,-1,-1),host.read[:,-1].expand(4,-1))
   if 32<=t<64:norms.append(float(ss[0].content.norm(dim=-1).mean()))
   if t==63:scale=.5*float(np.median(deltas))/max(float(np.median(norms)),1e-8)
  full=ss[0].content*scale;target=full.norm(dim=-1,keepdim=True);actions=torch.cat([full]+[W.unit(s.content)*target for s in ss[1:]]+[torch.zeros_like(full[:1])]);host.action=actions;oldlen=cache.get_seq_length();copied=W.cache_repeat(cache,17,model.config);y=model(torch.full((17,1),ids[-1],device='cuda',dtype=torch.long),past_key_values=copied,use_cache=True);acted=y.past_key_values;logits=y.logits[:,-1];actual=host.actual_norm.clone();host.action=None;assert cache.get_seq_length()==oldlen
  scores=[]
  for seq in candidates:
   work=W.cache_repeat(acted,1,model.config);last=logits;score=torch.zeros(17,device='cuda')
   for j,token in enumerate(seq):
    score+=last.float().log_softmax(-1)[:,token]
    if j+1<len(seq):y=model(torch.full((17,1),token,device='cuda',dtype=torch.long),past_key_values=work,use_cache=True);work=y.past_key_values;last=y.logits[:,-1]
   scores.append(score.cpu().numpy());del work
  scores=np.stack(scores);odds=scores[correct]-scores[1-correct];rows=[]
  for j,kind in enumerate(KINDS):
   for c in range(4):rows.append(dict(condition=kind,copy=c,correct_log_odds=float(odds[j*4+c]),baseline_correct_log_odds=float(odds[-1]),action_change_in_correct_log_odds=float(odds[j*4+c]-odds[-1]),effective_action_norm=float(actual[j*4+c]),option_log_likelihoods=scores[:,j*4+c].tolist()))
  save(f'semantic_information_{scene["id"]}.json',dict(scene=scene,stem=pair['stem'],options=pair['options'],correct_option=correct,prefix_tokens=len(ids),rows=rows,baseline_option_log_likelihoods=scores[:,-1].tolist()))
  del ss,cache,acted,copied,y,logits;gc.collect();torch.cuda.empty_cache()
 host.close();save('semantic_finished.json',dict(scenes=len(list((ROOT/'results').glob('semantic_information_*.json')))))

if __name__=='__main__':
 try:run()
 except Exception:save('semantic_failure.json',dict(traceback=traceback.format_exc()));raise
 finally:(ROOT/'EXTRA_PENDING').unlink(missing_ok=True)
