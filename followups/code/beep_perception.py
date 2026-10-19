"""Factorial image/audio binding and contextual-state extraction. Run after the pinned model setup cell."""
import base64
import gc
import io
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from PIL import Image, ImageDraw

output = Path('/content/jspace_pilot')
output.mkdir(exist_ok=True)
started = time.time()
print('MODEL_STATUS', type(model).__name__, torch.cuda.memory_allocated()/2**30, flush=True)
torch.manual_seed(381)
np.random.seed(381)
read_index = len(model.model.layers) * 2 // 3 + 1
sr = 16000
palette = [('red', 'blue'), ('green', 'yellow')]
features, records = {}, []


def picture(pair, reverse=False, grey=False):
    colours = pair[::-1] if reverse else pair
    image = Image.new('RGB', (224, 112), '#eeeeee')
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 12, 103, 100), fill='#888888' if grey else colours[0])
    draw.rectangle((120, 12, 215, 100), fill='#888888' if grey else colours[1])
    return image


def sound(count):
    a = np.zeros(sr, np.float32)
    for j in range(count):
        begin = int((.15 + j * .35) * sr)
        n = int(.16 * sr)
        envelope = np.hanning(n).astype(np.float32)
        a[begin:begin+n] = .35 * np.sin(2*np.pi*660*np.arange(n)/sr) * envelope
    return a


@torch.no_grad()
def observe(question, image=None, audio=None):
    content = []
    if image is not None: content.append({'type':'image', 'image':'synthetic-image'})
    if audio is not None: content.append({'type':'audio', 'audio':'synthetic-audio'})
    content.append({'type':'text','text':question})
    messages = [{'role':'system','content':[{'type':'text','text':'You are a helpful assistant.'}]}, {'role':'user','content':content}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    kwargs = dict(text=text, return_tensors='pt', padding=True)
    if image is not None: kwargs['images'] = [image]
    if audio is not None: kwargs.update(audio=[audio], sampling_rate=sr)
    batch = processor(**kwargs).to('cuda')
    for k in batch:
        if torch.is_floating_point(batch[k]): batch[k] = batch[k].to(torch.float16)
    res = model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
    hidden = res.hidden_states[read_index][0, -16:].float().cpu().numpy()
    logits = res.logits[0, -1].float()
    if not np.isfinite(hidden).all() or not torch.isfinite(logits).all():
        raise RuntimeError('Non-finite native inference')
    colour_probs = {}
    lp = logits.log_softmax(-1)
    for colour in sum(([a,b] for a,b in palette), []):
        token_ids = {processor.tokenizer.encode(v, add_special_tokens=False)[0] for v in (colour, colour.capitalize(), ' '+colour, ' '+colour.capitalize())}
        colour_probs[colour] = float(lp[list(token_ids)].logsumexp(0).exp())
    ids = model.generate(**batch, max_new_tokens=10, do_sample=False)
    answer = processor.tokenizer.decode(ids[0, batch['input_ids'].shape[1]:], skip_special_tokens=True)
    top = torch.topk(logits, 5).indices.tolist()
    detail = {'answer':answer, 'colour_probability_mass':colour_probs,
              'tokens':int(batch['input_ids'].shape[1]), 'top_next_tokens':[processor.tokenizer.decode([i]) for i in top]}
    del res, batch, logits, ids
    return hidden, detail


rule = 'The image has a left coloured panel and a right coloured panel. If you hear one beep, choose the left panel. If you hear two beeps, choose the right panel. What is the selected colour? Answer with only the colour name.'
try:
    for pi, pair in enumerate(palette):
        for reverse in (False, True):
            im = picture(pair, reverse)
            for count in (1, 2):
                key = f'p{pi}_r{int(reverse)}_n{count}'
                expected = (pair[::-1] if reverse else pair)[count-1]
                for condition in ('joint', 'silent', 'grey'):
                    x, detail = observe(rule, picture(pair, reverse, grey=condition=='grey'), sound(0 if condition=='silent' else count))
                    features[f'{key}_{condition}'] = x
                    score_choice = max(pair, key=lambda c:detail['colour_probability_mass'][c])
                    records.append(dict(trial=key, condition=condition, pair=pi, reversed=reverse, beeps=count, expected=expected,
                                        probability_choice=score_choice, probability_correct=score_choice==expected, **detail))
                # Symbolic audio control preserves the image while replacing sound by a truthful text count.
                _, detail = observe(rule + f' The sound contains {count} beep'+('s.' if count==2 else '.'), im, None)
                records.append(dict(trial=key, condition='text_audio_control', expected=expected, **detail))
                print('TRIAL_COMPLETE', key, flush=True)
            _, detail = observe('What is the colour of the left panel? Answer with only the colour name.', im, None)
            records.append(dict(trial=f'p{pi}_r{int(reverse)}', condition='image_control', expected=(pair[::-1] if reverse else pair)[0], **detail))
    for count in (1,2):
        _, detail = observe('How many short beeps do you hear? Answer with only one or two.', None, sound(count))
        records.append(dict(trial=f'beeps_{count}', condition='audio_control', expected=str(count), **detail))
    summary = {'complete':True, 'model':MODEL_ID, 'revision':REV, 'read_hidden_state_index':read_index,
               'model_parameters':sum(p.numel() for p in model.parameters()), 'base_dtype':'float16', 'attention':'eager',
               'gpu':torch.cuda.get_device_name(0), 'torch':torch.__version__, 'transformers':__import__('transformers').__version__,
               'elapsed_seconds':time.time()-started, 'peak_gpu_memory':torch.cuda.max_memory_allocated(), 'records':records}
except Exception:
    summary = {'complete':False, 'model':MODEL_ID, 'revision':REV, 'records':records, 'error':traceback.format_exc()}
    print(summary['error'], flush=True)
(output/'perception.json').write_text(json.dumps(summary, indent=2))
np.savez_compressed(output/'native_features.npz', **features)
print('PILOT_RESULT_JSON_BEGIN\n'+json.dumps(summary)+'\nPILOT_RESULT_JSON_END', flush=True)
buf = io.BytesIO()
np.savez_compressed(buf, **features)
print('PERCEPTION_STAGE_FINISHED', len(records), len(features), flush=True)
