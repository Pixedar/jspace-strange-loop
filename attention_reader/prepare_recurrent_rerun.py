"""Assemble a local GPU rerun layout. Does not rent, install, run or upload anything."""
import pathlib,json,time,argparse
P=pathlib.Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--out',type=pathlib.Path,required=True);parser.add_argument('--minutes',type=int,default=90);args=parser.parse_args();assert 10<=args.minutes<=240
target=args.out.resolve();assert target!=P.resolve(),'Choose a separate output directory'
mapping={
 'src/reader.py':'src/reader.py','src/c8_reference.pt':'src/c8_reference.pt',
 'experiments/cloud_recurrent.py':'cloud_queue.py','experiments/cloud_specificity.py':'inputs/specificity_reference.py',
 'experiments/base_cloud.py':'inputs/base_cloud.py','experiments/base_revision.py':'inputs/base_revision.py',
 'experiments/semantic_assay.py':'inputs/semantic_assay.py','experiments/information_choice.py':'inputs/information_choice.py',
 'inputs/selected_long.pt':'inputs/selected_long.pt','inputs/recurrent_state.pt':'inputs/recurrent_state.pt',
 'inputs/study_spec.json':'inputs/study_spec.json','inputs/semantic_context_variants.json':'inputs/semantic_context_variants.json'}
for source,destination in mapping.items():
 p=target/destination;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.new');q.write_bytes((P/source).read_bytes());q.replace(p)
spec=json.loads((P/'inputs/specificity_spec.json').read_text());spec['lease_deadline']=time.time()+args.minutes*60;p=target/'inputs/specificity_spec.json';q=p.with_suffix('.new');q.write_text(json.dumps(spec,indent=2));q.replace(p)
(target/'results').mkdir(exist_ok=True)
print('Prepared',target,'with a',args.minutes,'minute run deadline; no compute started.')
print('Mount this directory at /workspace/consolidation in a qualified CUDA environment. Start cloud_queue.py and then inputs/information_choice.py. Saved histories contain corpus-derived vectors: publication review still applies.')
