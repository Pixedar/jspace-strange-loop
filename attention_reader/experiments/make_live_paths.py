"""Rebuild measured paths without repeating completed behavior analysis."""
import pathlib,json,numpy as np
R=pathlib.Path(__file__).parent;C=R/'specificity/results/cloud/58704070/results';O=R/'results/extension'
records=[json.loads(p.read_text()) for p in sorted(C.glob('pattern_generation_*.json'))]
for d in records:
 d['file']='pattern_generation_'+d['scene']['id']+'_'+str(d['seed'])+'_'+d['condition']+'.json'
 d['token_ids']=[s[:s.index(151645)] if 151645 in s else s for s in d['token_ids']]
def save(p,x):
 q=p.with_suffix('.new');q.write_text(json.dumps(x),encoding='utf-8');q.replace(p)
s=(R/'analyze_pattern_generation.py').read_text(encoding='utf-8');code=s[s.index('# Fixed axes per scene'):s.index('np.savez_compressed(O/')]
exec(compile(code,'canonical_live_projection','exec'));print('Projected',len(paths),'scene/seed groups')
