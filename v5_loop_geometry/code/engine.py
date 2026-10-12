"""Research v5 rollout engine (PROTOCOL_V5 §3): the decision position of a fixed prompt is recomputed every loop step
with the prompt's KV prefix held fixed; a write is added at layer Lw of that position (additive to the natural state,
replacing the previous write); states are read at Lr (self-state) and Ls (interpretation probe); the observables are
lo (answer log-odds), s (probe), c (challenge response = lo under the same write with a contradiction inserted, - lo).
Reuses the v3 model wrapper (research_v3/code/common.py) and F2 scenarios (tasks.py)."""
import sys, pathlib
import numpy as np, torch

V3 = pathlib.Path(__file__).resolve().parents[2]/'v3_decision_loop'/'code'
sys.path.insert(0, str(V3))
from common import LM, log                      # noqa: E402
from tasks import user_text                      # noqa: E402


def cand_token_sets(lm, cands):
    """Per item: token ids of the surface variants of candidate A and B (disjoint), as used by v3's logodds."""
    out = []
    for a, b in cands:
        ia, ib = set(lm.variant_ids(a)), set(lm.variant_ids(b))
        out.append((sorted(ia-ib), sorted(ib-ia)))
    return out


def logodds_t(logits, sets):
    """Differentiable log P(A variants) - log P(B variants) for each row."""
    lp = torch.log_softmax(logits.float(), -1)
    return torch.stack([torch.logsumexp(lp[i, a], 0)-torch.logsumexp(lp[i, b], 0) for i, (a, b) in enumerate(sets)])


class Batch:
    """A batch of scenario prompts with fixed KV prefixes for the plain and the challenged version."""

    def __init__(self, eng, items, challenged_items):
        self.items = items;lm = eng.lm
        self.sets = cand_token_sets(lm, [it['cands'] for it in items])
        self.base = eng._prefix([user_text(it)[0] for it in items])
        self.chal = eng._prefix([user_text(it)[0] for it in challenged_items])


class Engine:
    def __init__(self, lm, Lr=29, Lw=22, Ls=25):
        self.lm, self.Lr, self.Lw, self.Ls = lm, Lr, Lw, Ls

    @torch.no_grad()
    def _prefix(self, texts):
        lm = self.lm;ids, att, _ = lm.encode([lm.chat(t) for t in texts])
        pre_ids, pre_att, last = ids[:, :-1], att[:, :-1], ids[:, -1:]
        posid = (pre_att.long().cumsum(-1)-1).clamp_min(0);lm.clear()
        o = lm.model(input_ids=pre_ids, attention_mask=pre_att, position_ids=posid, use_cache=True, logits_to_keep=1)
        return dict(cache=o.past_key_values, att=torch.cat([pre_att, torch.ones_like(last)], 1), last=last,
                    pos=pre_att.long().sum(-1, keepdim=True), L=pre_ids.shape[1])

    def _last(self, P, w):
        """Run only the decision token on top of the fixed prefix, adding w (B, d) at Lw; crop the cache back."""
        lm = self.lm;lm.clear();lm.read[self.Lr] = None;lm.read[self.Ls] = None
        if w is not None:
            ww = w.to(lm.model.dtype)
            lm.edit[self.Lw] = lambda h, ww=ww:h+ww[:, None, :]
        o = lm.model(input_ids=P['last'], attention_mask=P['att'], position_ids=P['pos'], past_key_values=P['cache'],
                     use_cache=True, logits_to_keep=1)
        P['cache'].crop(P['L'])
        out = dict(hr=lm.store[self.Lr][:, -1], hs=lm.store[self.Ls][:, -1], logits=o.logits[:, -1])
        lm.clear();return out

    def step(self, B, w=None, grad=False):
        """One loop step for a Batch: returns hr, hs (float), lo, lo_c (log-odds on plain and challenged prompts)."""
        with torch.set_grad_enabled(grad):
            a = self._last(B.base, w);c = self._last(B.chal, w)
            lo = logodds_t(a['logits'], B.sets);lo_c = logodds_t(c['logits'], B.sets)
        return dict(hr=a['hr'], hs=a['hs'], lo=lo, lo_c=lo_c, c=lo_c-lo)
