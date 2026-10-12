"""Do injected loop contents get attributed to an earlier conversation? Counts, per condition and paired by seed,
phrases in the generated thoughts that refer to a previous interaction that never happened."""
import json, glob, os, re, sys, collections
import numpy as np
PAT = re.compile(r"(previous conversation|previous interaction|last interaction|last conversation|earlier conversation|"
                 r"previous message|previous messages|earlier message|they mentioned|the user mentioned|user previously|"
                 r"mentioned earlier|mentioned before|as discussed|we discussed|we talked|previous chat|earlier chat|"
                 r"history of our|our previous|recalling the)", re.I)
out = {}
for run in sorted(glob.glob('/workspace/data/ring2/qwen3-*/speech_g*')):
    metas = sorted(glob.glob(os.path.join(run, '*', 'meta.json')))
    if not metas:continue
    rows, texts = [], []
    for m in metas:
        M = json.load(open(m, encoding='utf-8'));rows += M['rows'];texts += M['texts']
    conds = list(dict.fromkeys(r['cond'] for r in rows));seeds = sorted({r['seed'] for r in rows})
    idx = {(r['cond'], r['seed']):i for i, r in enumerate(rows)}
    cnt = {c:np.array([len(PAT.findall(texts[idx[c, s]])) for s in seeds]) for c in conds}
    anyc = {c:float(np.mean(cnt[c] > 0)) for c in conds}
    pair = {}
    for c in conds:
        if c == 'none':continue
        d = cnt[c]-cnt['none'];se = d.std(ddof=1)/np.sqrt(len(d)) if len(d) > 1 else 0
        pair[c] = dict(diff=float(d.mean()), t=float(d.mean()/se) if se > 0 else 0.)
    ex = {}
    for c in conds:
        for s in seeds:
            m = PAT.search(texts[idx[c, s]])
            if m:ex[c] = texts[idx[c, s]][max(0, m.start()-150):m.end()+200].replace('\n', ' ');break
    out[run.split('ring2/')[1]] = dict(frac_texts_with_phrase=anyc, mean_count={c:float(cnt[c].mean()) for c in conds}, vs_none=pair, example=ex)
    print(run, {c:round(v, 2) for c, v in anyc.items()}, flush=True)
os.makedirs('/workspace/v5/speech', exist_ok=True)
json.dump(out, open('/workspace/v5/speech/confab.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
