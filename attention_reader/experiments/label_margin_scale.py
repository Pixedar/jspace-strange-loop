"""Held-family scale control for the observed binary answer margins."""
import pathlib,json,numpy as np
R=pathlib.Path(__file__).parent;d=json.loads((R/'results/extension/information_choice.json').read_text());rr=d['records'];results=[]
for cond in ['full','recurrent_forecast','recurrent_selector','neighbor_forecast']:
 residual=[];raw=[];scale=[]
 for family in d['families']:
  train=[x for a in rr if a['scene']['family']!=family for x in a['rows'] if x['condition']==cond];test=[x for a in rr if a['scene']['family']==family for x in a['rows'] if x['condition']==cond];b=np.array([x['baseline_correct_label_log_odds'] for x in train]);y=np.array([x['correct_label_log_odds'] for x in train]);beta=float(b@y/(b@b));scale.append(beta)
  for x in test:residual.append(x['correct_label_log_odds']-beta*x['baseline_correct_label_log_odds']);raw.append(x['action_change_in_correct_log_odds'])
 results.append(dict(condition=cond,mean_cross_family_label_margin_scale=float(np.mean(scale)),heldout_fraction_of_squared_changes_explained=float(1-np.sum(np.square(residual))/np.sum(np.square(raw))),mean_heldout_residual=float(np.mean(residual)),scope='One scalar fitted on other families predicts the acted binary-label margin; not a full-vocabulary temperature-equivalence claim.'))
p=R/'results/extension/label_margin_scale.json';q=p.with_suffix('.new');q.write_text(json.dumps(dict(rows=results),indent=2));q.replace(p)
