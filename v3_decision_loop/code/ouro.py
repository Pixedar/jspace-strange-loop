"""Ouro-2.6B natural recurrence (architectural replication of the Huginn tests; protocol section 2.2 of the proposal).
Ouro applies all 48 layers per loop ("ut step"), re-normalizes between loops and has an exit head per loop; 4 loops are
trained, more loops are an out-of-distribution stress test. Hooks on the last decoder layer read / transplant /
perturb the decision-position state at the end of a chosen loop.
    python ouro.py --n 100 --K 8"""
import argparse, json, pathlib
import numpy as np, torch
from common import OUT, log, atomic_npz, atomic_json, seed_all
from e2 import make_contexts

REPO, REV = 'ByteDance/Ouro-2.6B', '1ed04250da1a9936042725d302e81c8fa2ab5abd'


class Ouro:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(REPO, revision=REV, trust_remote_code=True)
        self.tok.padding_side = 'left'
        if self.tok.pad_token is None:self.tok.pad_token = self.tok.eos_token
        kw = dict(revision=REV, trust_remote_code=True)
        try:self.m = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16, **kw)
        except TypeError:self.m = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16, **kw)
        self.m = self.m.cuda().eval()
        for p in self.m.parameters():p.requires_grad_(False)
        self.last_layer = self.m.model.layers[self.m.config.num_hidden_layers-1]
        self.edit = None;self.store = {}
        self.last_layer.register_forward_hook(self._hook, with_kwargs=True)

    def _hook(self, mod, args, kwargs, out):
        ut = kwargs.get('current_ut', 0);h = out[0] if isinstance(out, tuple) else out
        self.store[ut] = h[:, -1].float().clone()
        if self.edit is not None:
            new = self.edit(ut, h)
            if new is not None:return (new,)+tuple(out[1:]) if isinstance(out, tuple) else new
        return None

    def chat(self, t):
        try:return self.tok.apply_chat_template([{'role':'user', 'content':t}], tokenize=False, add_generation_prompt=True)
        except Exception:return f'{t}\nAnswer:'

    @torch.no_grad()
    def run(self, texts, K, edit=None):
        """Returns exit logits per loop at the decision position (K, B, V) and the pre-norm decision states per loop."""
        enc = self.tok([self.chat(t) for t in texts], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
        pos = (enc['attention_mask'].long().cumsum(-1)-1).clamp_min(0)
        self.m.model.total_ut_steps = K;self.edit = edit;self.store = {}
        outs, hs, gates = self.m.model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'], position_ids=pos, use_cache=False)
        self.edit = None
        logits = torch.stack([self.m.lm_head(h[:, -1]).float() for h in hs])
        return logits, torch.stack([self.store[k] for k in range(K)])

    def variant_ids(self, w):
        ids = set()
        for v in (w, ' '+w, w.capitalize(), ' '+w.capitalize()):
            t = self.tok.encode(v, add_special_tokens=False)
            if t:ids.add(t[0])
        return sorted(ids)

    def logodds(self, logits, cands):
        lp = torch.log_softmax(logits, -1);out = np.zeros(lp.shape[:2], np.float32)
        for b, (a, c) in enumerate(cands):
            ia, ic = self.variant_ids(a), self.variant_ids(c);sa, sc = set(ia), set(ic)
            ia = [i for i in ia if i not in sc];ic = [i for i in ic if i not in sa]
            out[:, b] = (torch.logsumexp(lp[:, b, ia], -1)-torch.logsumexp(lp[:, b, ic], -1)).cpu().numpy() if ia and ic else np.nan
        return out


def main():
    p = argparse.ArgumentParser();p.add_argument('--n', type=int, default=100);p.add_argument('--K', type=int, default=8)
    p.add_argument('--bs', type=int, default=32);p.add_argument('--tag', default='main')
    a = p.parse_args();out = OUT/'ouro'/a.tag;out.mkdir(parents=True, exist_ok=True);seed_all(0)
    O = Ouro();ctxs = make_contexts(a.n);texts = [c['text'] for c in ctxs];cands = [c['cands'] for c in ctxs];N = len(ctxs);K = a.K
    atomic_json(out/'contexts.json', ctxs);atomic_json(out/'provenance.json', dict(model=REPO, revision=REV, K=K, trained_loops=4))
    # 1. natural: exit log-odds per loop (loops 1..4 trained, 5..K out of distribution)
    LO, DS = [], []
    for i in range(0, N, a.bs):
        lg, st = O.run(texts[i:i+a.bs], K);LO.append(O.logodds(lg, cands[i:i+a.bs]));DS.append(st.cpu().numpy().astype(np.float16))
    lo = np.concatenate(LO, 1);ds = np.concatenate(DS, 1)
    dlt = np.linalg.norm(np.diff(ds.astype(np.float32), axis=0), axis=-1)/(np.linalg.norm(ds[1:].astype(np.float32), axis=-1)+1e-6)
    atomic_npz(out/'natural.npz', lo=lo, states=ds, dlt=dlt);log('natural done')
    # 2. memory: decision-state transplant from A or B into N at the end of loop k0, full-state swap A <-> B
    by = {}
    for i, c in enumerate(ctxs):by.setdefault((c['fam'], c['sid']), {})[c['tag']] = i
    trip = [(d['A'], d['B'], d['N']) for d in by.values() if all(k in d for k in 'ABN')];res = {}
    for k0 in (0, 1, 2):
        GA, GB, SA, SB = [], [], [], []
        for i in range(0, len(trip), a.bs//2):
            tt = trip[i:i+a.bs//2];_, src = O.run([texts[t[0]] for t in tt]+[texts[t[1]] for t in tt], K)
            sA, sB = src[k0, :len(tt)], src[k0, len(tt):]
            for s_, store in ((sA, GA), (sB, GB)):
                def tr(ut, h, s_=s_):
                    if ut != k0:return None
                    h = h.clone();h[:, -1] = s_.to(h.dtype);return h
                lg, _ = O.run([texts[t[2]] for t in tt], K, edit=tr);store.append(O.logodds(lg, [cands[t[2]] for t in tt]))
            same = [j for j, t in enumerate(tt) if len(O.tok(O.chat(texts[t[0]]))['input_ids']) == len(O.tok(O.chat(texts[t[1]]))['input_ids'])]
            if same:
                # full swap at the end of loop k0: run A with B's whole sequence state (captured by a full-state hook)
                full = {}

                def cap(ut, h):
                    if ut == k0:full['h'] = h.clone()
                    return None
                O.run([texts[tt[j][1]] for j in same], K, edit=cap);hB = full['h']
                O.run([texts[tt[j][0]] for j in same], K, edit=cap);hA = full['h']
                for tgt, srcf, store in ((0, hB, SA), (1, hA, SB)):
                    def sw(ut, h, srcf=srcf):
                        return srcf.to(h.dtype) if ut == k0 else None
                    lg, _ = O.run([texts[tt[j][tgt]] for j in same], K, edit=sw);store.append(O.logodds(lg, [cands[tt[j][tgt]] for j in same]))
        res[f'k{k0}_N_from_A'] = np.concatenate(GA, 1);res[f'k{k0}_N_from_B'] = np.concatenate(GB, 1)
        if SA:res[f'k{k0}_A_with_B_state'] = np.concatenate(SA, 1);res[f'k{k0}_B_with_A_state'] = np.concatenate(SB, 1)
        g = res[f'k{k0}_N_from_A']-res[f'k{k0}_N_from_B'];log('ouro memory k0', k0, 'gap per later loop', np.nanmean(g, 1).round(2).tolist())
    atomic_npz(out/'memory.npz', trip=np.array(trip), **res)
    # 3. sensitivity: 1 % noise on the decision state at the end of loop 1
    gen = torch.Generator(device='cuda');gen.manual_seed(7);D, L = [], []
    for i in range(0, N, a.bs):
        lg0, s0 = O.run(texts[i:i+a.bs], K)

        def eps(ut, h):
            if ut != 1:return None
            h = h.clone();v = h[:, -1].float();n = torch.randn(v.shape, device=v.device, generator=gen)
            h[:, -1] = (v+0.01*v.norm(dim=-1, keepdim=True)*n/n.norm(dim=-1, keepdim=True)).to(h.dtype);return h
        lg1, s1 = O.run(texts[i:i+a.bs], K, edit=eps)
        D.append(((s1-s0).norm(dim=-1)/s0.norm(dim=-1).clamp_min(1e-6)).cpu().numpy());L.append(O.logodds(lg1, cands[i:i+a.bs])-O.logodds(lg0, cands[i:i+a.bs]))
    atomic_npz(out/'sensitivity.npz', dist=np.concatenate(D, 1), dlo=np.concatenate(L, 1))
    log('ouro sensitivity median rel dist per loop', np.median(np.concatenate(D, 1), 1).round(4).tolist())
    (out/'DONE').write_text('ok')


if __name__ == '__main__':
    main()
