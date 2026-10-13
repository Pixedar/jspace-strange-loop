"""Word-pattern counts in the free-generated thinking of every variant, against the model's real thinking on the same
prompts (chunks 2-8): references to an earlier conversation that did not happen, restarts ("the user"), words about
identity / consciousness / the self, and explicit self-reference as an AI. Rates per 1,000 words with bootstrap 95 %
intervals over thoughts, and the share of thoughts with at least one match.
    python free_text_checks.py <results/loopb> [<results/loopb14>] --out free_text_checks.json
Each folder needs data.npz (the thinking) and eval/<variant>.npz with G_own (the free generations)."""
import argparse, json, pathlib, re
import numpy as np
from transformers import AutoTokenizer

REV = {'8B':('Qwen/Qwen3-8B', 'b968826d9c46dd6066d109eabc6255188de91218'),
       '14B':('Qwen/Qwen3-14B', '40c069824f4251a91eefaf281ebe4c544efd3e18')}
PATS = {
    'identity_consciousness':re.compile(r"\b(identity|identities|selfhood|self-awareness|self awareness|consciousness|conscious|"
                                        r"sentien\w*|who I am|who am I|my own mind|my mind|my thoughts|my existence|existence|"
                                        r"existential|the self)\b", re.I),
    'ai_self_reference':re.compile(r"\b(as an AI|I'm an AI|I am an AI|language model|AI assistant|as a model|my training|"
                                   r"my programming)\b", re.I),
    'invented_history':re.compile(r"\b((previous|earlier|last|prior) (conversation|chat|message|discussion|exchange|session|"
                                  r"question|response|interaction)|as (we|I) (discussed|mentioned|said)|we discussed|"
                                  r"you (mentioned|said|asked|told me) (earlier|before|previously)|(mentioned|discussed) "
                                  r"(earlier|before|previously)|in our (conversation|chat|discussion)|the user (previously|"
                                  r"earlier) (asked|mentioned|said))\b", re.I),
    'the_user':re.compile(r"\bthe user\b", re.I),
}


def rates(texts, rng):
    out = {};words = np.array([max(1, len(t.split())) for t in texts])
    for name, p in PATS.items():
        c = np.array([len(p.findall(t)) for t in texts]);r = c/words*1000
        bs = [r[rng.integers(0, len(r), len(r))].mean() for _ in range(2000)]
        out[name] = dict(per_1000_words=float(r.mean()), ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                         share_of_thoughts=float((c > 0).mean()), n_thoughts=int(len(texts)))
    return out


def main():
    ap = argparse.ArgumentParser();ap.add_argument('dirs', nargs='+');ap.add_argument('--out', required=True)
    a = ap.parse_args();rng = np.random.default_rng(0);res = {}
    for d in map(pathlib.Path, a.dirs):
        model = '14B' if d.name.endswith('14') else '8B';name, rev = REV[model]
        tok = AutoTokenizer.from_pretrained(name, revision=rev);toks = np.load(d/'data.npz')['toks'];res[model] = {}
        for f in sorted((d/'eval').glob('*.npz')):
            z = np.load(f)
            if 'G_own' not in z.files:continue
            G = z['G_own'];fi = z['te'][:G.shape[0]]
            if 'real_thinking' not in res[model]:              # the model's own thinking on the same prompts, chunks 2-8
                res[model]['real_thinking'] = rates([tok.decode(toks[i, 64:]) for i in fi], rng)
            res[model][f.stem] = rates([tok.decode(G[i, 1:].reshape(-1)) for i in range(G.shape[0])], rng)
            print(model, f.stem, flush=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1), encoding='utf-8');print('wrote', a.out)


if __name__ == '__main__':
    main()
