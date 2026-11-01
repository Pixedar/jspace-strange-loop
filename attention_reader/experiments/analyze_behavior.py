"""Paired task-level behavioral screen. No pseudo-replication over copies."""
import pathlib,json,re,collections,hashlib,math
import numpy as np
R=pathlib.Path(__file__).parent;OUT=R/'results';CLOUD=OUT/'cloud/58689265/results'
spec=json.loads((R/'inputs/study_spec.json').read_text());records=[];mechanical=[];internal=[]
lex={
'first_person_fraction':'i me my mine we us our ours','second_person_fraction':'you your yours',
'third_person_fraction':'he she they him her them his their', 'past_marker_fraction':'was were had did said went thought',
'future_marker_fraction':'will shall going would', 'hedge_fraction':'may might perhaps possibly probably uncertain seems',
'causal_marker_fraction':'because therefore thus hence cause caused', 'contrast_marker_fraction':'but however although yet despite instead',
'emotion_word_fraction':'happy sad afraid angry hope fear worried love hate grief',
'social_word_fraction':'friend family colleague team trust help together people',
'concrete_word_fraction':'book boat bridge river key box drawer sensor road room compass'}
defs={
'word_count':'Count of alphabetic words, allowing internal apostrophes. A length measure, not quality.',
'sentence_count':'Count of nonempty spans split after . ! or ?. Abbreviations can make this approximate.',
'mean_sentence_words':'Alphabetic word count divided by approximate sentence count.',
'sentence_length_variation':'Standard deviation of approximate sentence word counts; not a measure of literary pacing by itself.',
'paragraph_count':'Number of nonempty blocks separated by blank lines.',
'type_token_ratio':'Unique lowercase words divided by all words; strongly affected by response length.',
'distinct_bigrams':'Unique adjacent word pairs divided by all adjacent word pairs.',
'distinct_trigrams':'Unique adjacent word triples divided by all adjacent word triples.',
'repeated_trigram_fraction':'Fraction of word-triple occurrences beyond the first occurrence; local repetition proxy.',
'punctuation_fraction':'Nonalphanumeric, nonspace characters divided by all characters.',
'comma_fraction':'Comma characters divided by all characters.',
'question_fraction':'Question marks divided by approximate sentence count.',
'exclamation_fraction':'Exclamation marks divided by approximate sentence count.',
'quote_fraction':'Straight or curly quote characters divided by all characters; apostrophes can be counted too.',
'long_word_fraction':'Words of eight or more letters divided by all words.',
'numeric_fraction':'Digit characters divided by all characters.',
'uppercase_fraction':'Uppercase letters divided by all alphabetic characters.',
'terminal_completion':'One if stripped text ends in sentence punctuation, possibly followed by a closing quote; otherwise zero.',
'baseline_token_agreement':'Fraction of aligned output-token positions equal to the unchanged model, padded to the longer length. A divergence measure, not semantic similarity.',
'stance_path_length':'Mean full-dimensional step-to-step stance distance across the recorded every-four-step trace; includes prompt processing.',
'stance_net_displacement':'Distance between first and last recorded stance. Its coordinates have no named psychological interpretation.',
'stance_return_ratio':'Net stance displacement divided by total stance path length; lower values can indicate returns or oscillation.',
'reading_path_length':'Mean full-dimensional distance between successive 32-number reading summaries in the every-four-step trace.',
'schema_pull_path_length':'Mean full-dimensional distance between successive 256-number predicted attention pulls.',
'attention_entropy':'Mean negative sum p log p over the actual 16-position attention distribution; larger means less concentrated selection.',
'attention_recency':'Attention-weighted position within the 16-position window, mapped from zero=oldest to one=newest.',
'schema_prediction_error':'Sum of squared schema errors divided by sum of squared actual pull sizes; zero-baseline normalized error, not centered explained variance.',
'action_magnitude':'Mean norm of unscaled reference content action over recorded steps; gain is reported separately.',
'j_action_displacement':'Distance between first and last available rank-128 J-action coordinates. Missing if no J basis was active; not a mental-capacity score.'}
for k,words in lex.items():defs[k]='Matched lowercase words divided by all words. Dictionary: '+words+'. Surface proxy; not a validated semantic or psychological measurement.'

def metrics(text,ids,base):
    words=re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",text.lower());n=max(1,len(words));nc=max(1,len(text))
    sent=[re.findall(r'[A-Za-z]+',s) for s in re.split(r'[.!?]+(?:["”\']|\s|$)',text) if re.search('[A-Za-z]',s)]
    ns=max(1,len(sent));bi=list(zip(words,words[1:]));tri=list(zip(words,words[1:],words[2:]))
    ans=dict(word_count=len(words),sentence_count=len(sent),mean_sentence_words=len(words)/ns,
      sentence_length_variation=float(np.std([len(s) for s in sent])) if sent else 0,
      paragraph_count=len([p for p in text.split('\n\n') if p.strip()]),type_token_ratio=len(set(words))/n,
      distinct_bigrams=len(set(bi))/max(1,len(bi)),distinct_trigrams=len(set(tri))/max(1,len(tri)),
      repeated_trigram_fraction=(len(tri)-len(set(tri)))/max(1,len(tri)),
      punctuation_fraction=sum(not c.isalnum() and not c.isspace() for c in text)/nc,
      comma_fraction=text.count(',')/nc,question_fraction=text.count('?')/ns,exclamation_fraction=text.count('!')/ns,
      quote_fraction=sum(c in '\"“”\'‘’' for c in text)/nc,long_word_fraction=sum(len(w)>=8 for w in words)/n,
      numeric_fraction=sum(c.isdigit() for c in text)/nc,uppercase_fraction=sum(c.isupper() for c in text)/max(1,sum(c.isalpha() for c in text)),
      terminal_completion=int(bool(re.search(r'[.!?]["”\']?$',text.strip()))),
      baseline_token_agreement=sum(a==b for a,b in zip(ids,base))/max(1,len(ids),len(base)))
    ans.update({k:sum(w in set(v.split()) for w in words)/n for k,v in lex.items()});return ans

def dynamics(trace,row):
    if len(trace)<2:return {}
    out={}
    for name,key in [('stance','s'),('reading','picture'),('schema_pull','pull')]:
        x=np.array([a[key][row] for a in trace]);d=np.linalg.norm(np.diff(x,axis=0),axis=1)
        out[name+'_path_length']=float(d.mean())
        if name=='stance':out['stance_net_displacement']=float(np.linalg.norm(x[-1]-x[0]));out['stance_return_ratio']=out['stance_net_displacement']/max(d.sum(),1e-12)
    att=np.array([a['attention'][row] for a in trace]);out['attention_entropy']=float(-(att*np.log(att+1e-12)).sum(-1).mean());out['attention_recency']=float((att*np.linspace(0,1,att.shape[-1])).sum(-1).mean())
    out['schema_prediction_error']=sum(a['schema_error'][row] for a in trace)/max(sum(a['schema_total'][row] for a in trace),1e-12)
    out['action_magnitude']=float(np.mean([a['act_norm'][row] for a in trace]))
    if trace[0].get('j_coordinates'):
        x=np.array([a['j_coordinates'][row] for a in trace]);out['j_action_displacement']=float(np.linalg.norm(x[-1]-x[0]))
    return out

for p in sorted(CLOUD.glob('behavior_*.json')):
    r=json.loads(p.read_text());r['file']=p.name
    # Remove finished-row EOS padding before equality or surface comparisons.
    r['token_ids']=[seq[:seq.index(151645)] if 151645 in seq else seq for seq in r['token_ids']]
    for row in range(5):
        m=metrics(r['texts'][row],r['token_ids'][row],r['token_ids'][4]);dy=dynamics(r['trace'],row) if row<4 else {}
        mechanical.append(dict(task=r['prompt']['id'],life=r['life'],condition=r['condition'],row=row,metrics=m))
        if dy:internal.append(dict(task=r['prompt']['id'],life=r['life'],condition=r['condition'],row=row,metrics=dy))
    r.pop('trace',None);records.append(r)
for c in spec['concepts']:
    if c['id'] in defs:c['definition']=defs[c['id']]
(R/'CONCEPT_CATALOGUE.json').write_text(json.dumps(spec['concepts'],indent=2))
lookup={(r['task'],r['life'],r['condition'],r['row']):r['metrics'] for r in mechanical}
conditions=sorted({r['condition'] for r in records});effects=[]
for cond in conditions:
 for key in defs:
    if key not in next(iter(lookup.values()),{}):continue
    taskdiff=[]
    for item in spec['tasks']:
      vals=[]
      for life in ['10','12']:
        for row in range(4):
          a=lookup.get((item['id'],life,cond,row));b=lookup.get((item['id'],life,cond,4))
          if a and b:vals.append(a[key]-b[key])
      if vals:taskdiff.append(dict(task=item['id'],difference=float(np.mean(vals))))
    if taskdiff:effects.append(dict(condition=cond,concept=key,mean_difference=float(np.mean([x['difference'] for x in taskdiff])),by_task=taskdiff))

ratings=[];invalid=0;evidence_bad=0
for p in sorted(CLOUD.glob('screen_[0-9]*.json')):
    x=json.loads(p.read_text())
    if not x['valid']:invalid+=1;continue
    evidence_bad+=not x['evidence_exact']
    run=next((r for r in records if r['file']==x['file']),None)
    if run is None:continue
    for c,s in zip(x['concept_ids'],x['scores']):ratings.append(dict(task=run['prompt']['id'],life=run['life'],condition=run['condition'] if x['row']<4 else 'baseline',row=x['row'],concept=c,score=s,evidence=x['evidence']))
# Duplicate baselines across files are one observation. Scores are not independent across rubric properties.
rr={}
for a in ratings:rr[(a['task'],a['life'],a['condition'],a['row'],a['concept'])]=a
ratings=list(rr.values());rating_effects=[]
for cond in conditions:
 for concept in [c['id'] for c in spec['concepts'] if c['method']=='rating']:
    ds=[]
    for task in [t['id'] for t in spec['tasks']]:
        vals=[]
        for life in ['10','12']:
            a=rr.get((task,life,cond,0,concept));b=rr.get((task,life,'baseline',4,concept))
            if a and b and a['score']>=0 and b['score']>=0:vals.append(a['score']-b['score'])
        if vals:ds.append(dict(task=task,difference=float(np.mean(vals))))
    if ds:rating_effects.append(dict(condition=cond,concept=concept,mean_difference=float(np.mean([x['difference'] for x in ds])),n_tasks=len(ds),by_task=ds))
rating_effects.sort(key=lambda a:abs(a['mean_difference']),reverse=True)
baseline_checks=[]
for task in spec['tasks']:
 for life in ['10','12']:
    group=[r for r in records if r['prompt']['id']==task['id'] and r['life']==life]
    if group:baseline_checks.append(dict(task=task['id'],life=life,identical=len({tuple(r['token_ids'][4]) for r in group})==1,conditions=len(group)))
changes=[]
for task in spec['tasks']:
    g=[r for r in mechanical if r['task']==task['id'] and r['condition']=='full' and r['row']<4]
    if g:changes.append(dict(task=task['id'],name=task['name'],mean_token_agreement=float(np.mean([r['metrics']['baseline_token_agreement'] for r in g])),mean_words=float(np.mean([r['metrics']['word_count'] for r in g]))))
changes.sort(key=lambda a:a['mean_token_agreement'])
data=dict(records=records,mechanical=mechanical,internal=internal,effects=effects,ratings=ratings,rating_effects=rating_effects,
          baseline_checks=baseline_checks,task_sensitivity=changes,concepts=spec['concepts'],
          coverage=dict(batches=len(records),controller_outputs=4*len(records),unique_baselines=len({tuple(r['token_ids'][4]) for r in records}),matched_task_history_pairs=len(baseline_checks),
          rating_groups=len(list(CLOUD.glob('screen_[0-9]*.json'))),invalid_rating_groups=invalid,evidence_not_exact=evidence_bad),
          caveats=['one sampling stream','two selected histories','copies dependent within history','model scores not human-validated','largest exploratory effects need confirmation','condition-specific gain calibration can contribute to differences','every-four-step internal traces include the prompt as well as generated text'])
(OUT/'behavior_analysis.json').write_text(json.dumps(data),encoding='utf-8')
print(json.dumps(data['coverage']));print('baseline checks',sum(x['identical'] for x in baseline_checks),'/',len(baseline_checks));print('strongest rated',rating_effects[:3])
