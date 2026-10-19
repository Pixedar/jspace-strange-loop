"""Extract matched suffix and modality-balanced windows from the same native multimodal forward passes."""
balanced_features={}
intake_positions={}
intake_details=[]

@torch.no_grad()
def paired_intake(question,image,audio):
    messages=[{'role':'system','content':[{'type':'text','text':'You are a helpful assistant.'}]},
              {'role':'user','content':[{'type':'image','image':'synthetic-image'}, {'type':'audio','audio':'synthetic-audio'}, {'type':'text','text':question}]}]
    text=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
    batch=processor(text=text,images=[image],audio=[audio],sampling_rate=16000,return_tensors='pt',padding=True).to('cuda')
    for k in batch:
        if torch.is_floating_point(batch[k]): batch[k]=batch[k].to(torch.float16)
    ids=batch['input_ids'][0]
    image_positions=(ids==model.config.image_token_index).nonzero().flatten()
    audio_positions=(ids==model.config.audio_token_index).nonzero().flatten()
    assert len(image_positions)>=4 and len(audio_positions)>=4, (len(image_positions),len(audio_positions))
    def sample(pos,n): return pos[torch.linspace(0,len(pos)-1,n,device='cuda').round().long()]
    text_positions=torch.arange(len(ids)-9,len(ids)-1,device='cuda')
    positions=torch.cat([sample(image_positions,4),sample(audio_positions,4),text_positions])
    res=model(**batch,output_hidden_states=True,use_cache=False,return_dict=True)
    native=res.hidden_states[read_index][0].float()
    return native[-16:].cpu().numpy(),native[positions].cpu().numpy(),positions.cpu().tolist(),{'image_positions':len(image_positions),'audio_positions':len(audio_positions),'total_positions':len(ids)}

for pi,pair in enumerate(palette):
    for reverse in (False,True):
        for count in (1,2):
            key=f'p{pi}_r{int(reverse)}_n{count}_joint'
            suffix,balanced,positions,details=paired_intake(rule,picture(pair,reverse),sound(count))
            balanced_features[key+'_suffix']=suffix
            balanced_features[key+'_balanced']=balanced
            intake_positions[key]=positions
            intake_details.append({'key':key,**details})
np.savez_compressed(screen_dir/'paired_intakes.npz',**balanced_features)
print('INTAKE_RESULT_JSON_BEGIN\n'+json.dumps({'complete':True,'details':intake_details,'positions':intake_positions})+'\nINTAKE_RESULT_JSON_END',flush=True)
buf=io.BytesIO();np.savez_compressed(buf,**balanced_features)
print('INTAKES_COMPLETE',flush=True)
