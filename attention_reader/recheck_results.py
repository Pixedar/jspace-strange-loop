"""Recompute core published aggregates from the recorded contrasts, without a GPU."""
import pathlib,json,collections
P=pathlib.Path(__file__).resolve().parent
d=json.loads((P/'evidence/revision/extension/information_choice.json').read_text());groups=collections.defaultdict(list)
for record in d['records']:
 for row in record['rows']:groups[(record['scene']['wording'],row['condition'],record['scene']['family'])].append(row['action_change_in_correct_log_odds'])
rows=[]
for wording in ['original','paraphrase']:
 for condition in ['full','recurrent_forecast','recurrent_selector','neighbor_forecast']:
  means=[sum(v)/len(v) for (w,c,f),v in groups.items() if w==wording and c==condition];mean=sum(means)/len(means);expected=next(r['mean'] for r in d['rows'] if r['wording']==wording and r['condition']==condition);assert abs(mean-expected)<1e-10;rows.append(dict(wording=wording,condition=condition,mean=mean))
print(json.dumps(dict(recomputed=rows,contexts=len(d['records']),format_qualified=sum(r['baseline_format_qualified'] for r in d['records']),scope='Checks the aggregate calculation from recorded model contrasts; does not rerun the model.'),indent=2))
