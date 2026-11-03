"""Verify the signed publication record. Requires OpenSSH ssh-keygen."""
from pathlib import Path
import subprocess,json,argparse,hashlib
root=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--evidence-dir',type=Path);args=p.parse_args();payload=(root/'PUBLICATION_RECORD.json').read_bytes();record=json.loads(payload)
verified=subprocess.run(['ssh-keygen','-Y','verify','-f',str(root/'PUBLICATION_ALLOWED_SIGNERS'),'-I','pixedar@gmail.com','-n','file','-s',str(root/'PUBLICATION_RECORD.json.sig')],input=payload,capture_output=True)
if verified.returncode:raise SystemExit('Signature verification failed. Compare the signer with the public account key list and the version tag.')
print('Publication signature valid under the included key. Public account key list: '+record['signature']['public_key_association']);print('Pinned scientific source: '+record['source_commit']);print('Pinned evidence data: '+record['evidence_data_commit'])
if args.evidence_dir:
 for item in record['archives']:
  path=args.evidence_dir/item['path'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
  if digest!=item['sha256']:raise SystemExit('Archive checksum mismatch: '+item['path'])
 print('Both local numerical archives match the signed record.')
else:print('Archive bytes were not checked; use --evidence-dir after downloading the pinned files.')
