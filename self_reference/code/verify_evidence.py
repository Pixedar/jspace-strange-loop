"""Check saved evidence, recompute field comparisons and audit displayed loss arithmetic."""
from pathlib import Path
import hashlib
import json
import numpy as np

root=Path(__file__).resolve().parents[2]
results=root/'self_reference/results'
summary=json.loads((results/'verified_summary.json').read_text())
checks=0
for row in summary['runs']:
    run=results/'v24'/row['id']
    residual=json.loads((run/'self_model_residual_3000.json').read_text())
    for key,source in [('dependence','remainder_share_of_field'),('correlation','predicted_remainder_correlation'),('unscaled_explained','predicted_remainder_explained')]:
        assert np.isclose(row[key],residual[source],rtol=1e-7), (row['id'],key)
        checks+=1
    report=json.loads((run/'spotlight_check_3000.json').read_text())[row['id']]
    own=report['against_no_channel']['own']
    assert np.isclose(row['changed'],own['top_prediction_changed'])
    assert np.isclose(row['loss_pct'],100*(own['loss']/own['loss_without']-1),atol=1e-9)
    ratio=report['field_routes']['spotlight']/max(report['field_routes']['both'],1e-12)
    assert np.isclose(row['model_route_ratio'],ratio)
    checks+=3

def corr(a,b):
    a=a.ravel()-a.mean(); b=b.ravel()-b.mean()
    return float((a*b).sum()/max(np.sqrt((a*a).sum()*(b*b).sum()),1e-12))

pairs=json.loads((results/'v24/between_lives.json').read_text())
for row in pairs:
    def record(name):
        run,checkpoint=name.split('@')
        return np.load(results/'v24'/run/f'fields_iface_{checkpoint}.npz',allow_pickle=False)
    a,b=record(row['a']),record(row['b'])
    n=min(len(a['dest']),len(b['dest']))
    da,db=a['dest'][:n],b['dest'][:n]
    assert np.isclose(corr(da-da.mean(2,keepdims=True),db-db.mean(2,keepdims=True)),row['dependence_corr'],atol=2e-6)
    assert np.isclose(corr(da,db),row['destination_corr'],atol=2e-6)
    pa,pb=a['path'],b['path']; k=min(len(pa),len(pb))
    assert np.isclose(np.linalg.norm(pa[k//3:k]-pb[k//3:k],axis=-1).mean(),row['path_distance'],atol=2e-6)
    checks+=3

for path in results.rglob('*.npz'):
    with np.load(path,allow_pickle=False) as z:
        for key in z.files:
            if np.issubdtype(z[key].dtype,np.number):
                assert np.isfinite(z[key]).all(),(str(path),key)
                checks+=1

manifest=json.loads((root/'self_reference/source_manifest.json').read_text())
for relative,entry in manifest['evidence'].items():
    path=root/relative
    assert path.exists(),f'Download missing evidence: {relative}'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['export_sha256'],relative
    checks+=1
print(f'Passed {checks} evidence checks, including {len(pairs)} recomputed field comparisons.')
print('Field-head correlations remain recorded summaries: their full predicted arrays were not saved.')
