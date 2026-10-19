"""Verify the exact combined input corpus used by the saved experiments."""
from pathlib import Path
import hashlib
import json
import sys

sources=Path(__file__).resolve().parents[1]/'corpus_sources.json'
expected=json.loads(sources.read_text())['original_corpus_sha256']
path=Path(sys.argv[1])
actual=hashlib.sha256(path.read_bytes()).hexdigest()
print('corpus SHA-256:',actual)
if actual!=expected:
    raise SystemExit('Corpus mismatch: this reconstruction does not reproduce the original inputs.')
print('Exact corpus identity verified.')
