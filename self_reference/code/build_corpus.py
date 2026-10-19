"""Build the Stage 2 text corpus: labelled genres of ordinary text (no minds, consciousness, AI or robots).
Run with the system Python (needs pyarrow): python -I build_corpus.py OUT_DIR
Genres: expository (WikiText-2), news (AG News test), story (TinyStories valid), dialogue (DialogSum), reports (the v10
renderer of unfolding situations with sources of different reliability)."""
import csv
import json
import os
import random
import re
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REVISIONS = {'Salesforce/wikitext': 'b08601e04326c79dfdd32d625aee71d232d685c3', 'fancyzhx/ag_news': 'eb185aade064a813bc0b7f42de02595523103ca4', 'roneneldan/TinyStories': 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64', 'knkarthick/dialogsum': 'a968e7aee0602e257935f1321a02e4287f7d5848'}

OUT = sys.argv[1]
RAW = os.path.join(OUT, "raw")
os.makedirs(RAW, exist_ok=True)
BAN = re.compile(r"\b(conscious|consciousness|mind|minds|robot|robots|artificial intelligence|AI|android|sentient|"
                 r"self-aware|soul|souls|thinking machine|computer program)\b", re.I)
rng = random.Random(0)


def ok(t):
    return len(t.split()) >= 60 and not BAN.search(t)


def get(repo, fn):
    return hf_hub_download(repo, fn, repo_type="dataset", revision=REVISIONS[repo], local_dir=RAW)


corpus = {}
# expository: WikiText-2 articles split into paragraph groups of ~120-300 words
tbl = pq.read_table(get("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet")).column("text").to_pylist()
docs, cur = [], []
for line in tbl:
    line = line.strip()
    if re.match(r"^= [^=].*[^=] =$", line):
        if cur:
            docs.append(" ".join(cur))
        cur = []
    elif line and not line.startswith("="):
        cur.append(line.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".").replace(" ,", ",").replace(" .", ".").replace(" 's", "'s"))
        if len(" ".join(cur).split()) > 220:
            docs.append(" ".join(cur))
            cur = []
corpus["expository"] = [d for d in docs if ok(d)][:1500]
# news: AG News items grouped by topic into ~150-word pieces
tbl = pq.read_table(get("fancyzhx/ag_news", "data/test-00000-of-00001.parquet"))
texts, labels = tbl.column("text").to_pylist(), tbl.column("label").to_pylist()
by = {}
for t, l in zip(texts, labels):
    by.setdefault(l, []).append(t.replace("\\", " "))
news = []
for l, items in by.items():
    for i in range(0, len(items) - 3, 3):
        news.append(" ".join(items[i:i + 3]))
rng.shuffle(news)
corpus["news"] = [d for d in news if ok(d)][:1500]
# story: TinyStories validation stories
raw = open(get("roneneldan/TinyStories", "TinyStories-valid.txt"), encoding="utf-8").read().split("<|endoftext|>")
corpus["story"] = [s.strip() for s in raw if ok(s.strip())][:1500]
# dialogue: DialogSum dialogues
rows = list(csv.DictReader(open(get("knkarthick/dialogsum", "train.csv"), encoding="utf-8")))
dia = [r["dialogue"].replace("#Person1#", "A").replace("#Person2#", "B").replace("\r", "") for r in rows]
corpus["dialogue"] = [d for d in dia if ok(d)][:1500]
# reports: v10 renderer (situations with a direct inspector, a repeater of old impressions and a recorder)
from report_world import plan  # noqa: E402

rep = []
for seed in range(1500):
    p = plan(10_000 + seed, 6)
    rep.append(p["initial"] + " " + " ".join(p["events"]))
corpus["reports"] = [d for d in rep if ok(d)]
json.dump(corpus, open(os.path.join(OUT, "corpus.json"), "w"))
print({k: (len(v), int(sum(len(d.split()) for d in v) / max(1, len(v)))) for k, v in corpus.items()})
