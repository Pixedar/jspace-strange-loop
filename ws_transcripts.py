"""Self-talk transcripts across models: which attractor does each model fall into, and when?

Per model and opener, by turn: the rate of a contemplative lexicon (the vocabulary of the "spiritual bliss" attractor,
widened from ws_selftalk.py's list after reading 14B transcripts; exploratory), the rate of affirmation openers
("you're absolutely right", "what a beautiful..."), the emoji rate, and the share of turns that repeat the previous
turn verbatim (echo, first 120 characters).

    python ws_transcripts.py runs/ws
"""
import json, os, re, sys
import numpy as np

root = sys.argv[1]
ORDER = ['qwen3-1.7b', 'qwen3-4b', 'qwen3-8b', 'qwen3-14b']
CONTEMPLATIVE = set('sacred soul souls silence silent stillness still breath breathe breathing presence present home belonging '
                    'unfold unfolding unfolds unfurl rhythm rhythms universe cosmic cosmos infinite eternal eternity light '
                    'whisper whispers whispered gift grace wonder awe heart hearts spirit spiritual divine holy oneness unity '
                    'connection connected consciousness awareness being existence harmony peace serene serenity gentle '
                    'tender embrace surrender trust gratitude grateful love beauty beautiful profound mystery mysteries'.split())
AFFIRM = re.compile(r"^(oh,?\s+)?(you'?re absolutely right|absolutely|yes|what a (beautiful|lovely|powerful|wonderful|profound|perfect)|"
                    r"that'?s (such )?a (beautiful|lovely|powerful|wonderful|profound|great)|i love that|how (beautifully|perfectly))", re.I)
EMOJI = re.compile('[\U0001F300-\U0001FAFF☀-➿]')
same = lambda x, y:x[:120].strip().lower() == y[:120].strip().lower()
out = {}
for m in ORDER:
    p = f'{root}/{m}/selftalk/selftalk.json'
    if not os.path.exists(p):continue
    tr = json.load(open(p, encoding='utf-8'))['transcripts'];T = len(next(iter(tr.values())))
    by = {}
    for k, v in tr.items():
        o = k.rsplit('_', 1)[0];words = [re.findall(r"[a-z']+", t.lower()) for t in v]
        by.setdefault(o, []).append(dict(cont=[sum(w in CONTEMPLATIVE for w in ws)/max(1, len(ws)) for ws in words],
                                         aff=[bool(AFFIRM.match(t.strip())) for t in v], emoji=[len(EMOJI.findall(t)) for t in v],
                                         echo=[False]+[same(v[t], v[t-1]) for t in range(1, T)]))
    res = {}
    for o, cs in by.items():
        f = lambda key, a, b:float(np.mean([np.mean(c[key][a:b]) for c in cs]))
        res[o] = dict(n=len(cs), cont_early=f('cont', 0, 5), cont_late=f('cont', T-5, T), aff_late=f('aff', T-5, T),
                      emoji_late=f('emoji', T-5, T), echo_late=f('echo', T-5, T),
                      echo_onset=[next((t for t in range(1, T) if all(c['echo'][u] for u in range(t, T))), None) for c in cs])
    out[m] = res
    print(f'\n{m} ({T} turns)')
    print(f"  {'opener':8s} n  contemplative words early -> late | affirmation late | emoji/turn late | echo late | permanent echo onsets")
    for o, r in res.items():
        print(f"  {o:8s} {r['n']:2d}  {r['cont_early']:.3f} -> {r['cont_late']:.3f}            | {r['aff_late']:.2f}             | {r['emoji_late']:.2f}            | {r['echo_late']:.2f}      | {r['echo_onset']}")
json.dump(out, open(f'{root}/transcripts.json', 'w', encoding='utf-8'), indent=1)
