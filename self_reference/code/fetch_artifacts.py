"""Download only the selected immutable release artifacts, verifying each SHA-256."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
from huggingface_hub import hf_hub_download

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',default='.')
args=parser.parse_args()
root=Path(__file__).resolve().parents[2]
manifest=json.loads((root/'ARTIFACTS.json').read_text())
output=Path(args.output).resolve()
for entry in manifest['files']:
    destination=(output/entry['path']).resolve()
    if not destination.is_relative_to(output):
        raise ValueError('Artifact path leaves the selected output directory')
    cached=Path(hf_hub_download(manifest['dataset'],entry['path'],repo_type='dataset',revision=manifest['revision']))
    digest=hashlib.sha256(cached.read_bytes()).hexdigest()
    if digest!=entry['sha256']:
        raise RuntimeError('SHA-256 mismatch: '+entry['path'])
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.partial')
    shutil.copyfile(cached,temporary)
    temporary.replace(destination)
print('Verified',len(manifest['files']),'artifacts at dataset revision',manifest['revision'])
