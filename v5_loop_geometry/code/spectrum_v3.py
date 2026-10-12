"""The spectrum of the J-space strange loop (research_v5 geometry, causal follow-up; GPU, frozen Qwen3-8B).

Re-runs v3's J loop exactly (research_v3/code/e2.py: same calibration, contexts, read L29 / write L22, constant-norm
broadcast of the 25-word code) and asks what the loop's self-reproducing states are, dynamically:
  settle   run the loop T0 = 32 steps from the natural state, then 32 more (late window), per gain
  jac      local linear map of the loop at its late state, in coefficient space over the state's own 25 atoms:
           c -> least-squares coefficients of the next read on the same atoms; central differences -> 25 x 25 matrix;
           eigenvalues (stable |l| < 1, neutral l ~ 1, flip l ~ -1, rotation = complex) and the words of the eigenvectors
  pert     M random coefficient perturbations at T0, free loop for 32 steps: which part of a perturbation persists, in how
           many directions (local dimension of the set of self-reproducing states), carried by which words
  fp32     (separate process) the settle stage in an fp32 model: does the late jitter survive without bf16 rounding?
    python spectrum_v3.py --stage settle|jac|pert|fp32 --out /workspace/v5/spec [--gains 0.25,1,4]"""
import argparse, json, math, os, pathlib, sys, time
import numpy as np
import torch

V3 = pathlib.Path(os.environ.get('V3_CODE', pathlib.Path(__file__).resolve().parents[2]/'v3_decision_loop'/'code'))
sys.path.insert(0, str(V3))
import common                                          # noqa: E402
from common import LM, log, atomic_npz, atomic_json   # noqa: E402

MODEL = os.environ.get('S_MODEL', '8b')                    # '8b' (main) or '14b' (replication)
KEY = {'8b':'qwen3-8b', '14b':'qwen3-14b'}[MODEL];V3TAG = {'8b':'main', '14b':'rep'}[MODEL]
LR, LW, K = {'8b':(29, 22, 25), '14b':(32, 24, 25)}[MODEL]
common.MODELS[KEY]['read'] = LR;common.MODELS[KEY]['write'] = LW
DEV = 'cuda';BS = int(os.environ.get('S_BS', 48))


def v3_files(root):
    from huggingface_hub import hf_hub_download
    out = {}
    for f in ('contexts.json', 'calib/calib.npz', 'provenance.json'):
        out[f] = hf_hub_download('Pixedar/jspace-v3', f'v3/e2/{KEY}/{V3TAG}/{f}', repo_type='dataset', local_dir=root)
    return out


def pick_contexts(ctxs, n_per=24):
    """24 per family (F2 A/B/N, F3 A/B/N, F6 A/B, F5) in file order; F6 N excluded (identical duplicate prompts)."""
    sel = {'F2':[], 'F3':[], 'F6':[], 'F5':[]}
    for i, c in enumerate(ctxs):
        f = c['fam']
        if f == 'F6' and c['tag'] == 'N':continue
        if f in sel and len(sel[f]) < n_per:sel[f].append(i)
    return [i for f in ('F2', 'F3', 'F6', 'F5') for i in sel[f]]


class JLoop:
    def __init__(self, lm, C):
        self.lm = lm;self.C = C;self.D = lm.lens.D[LR];self.Dw = lm.lens.D[LW]

    @torch.no_grad()
    def prefix(self, texts):
        lm = self.lm;ids, att, _ = lm.encode([lm.chat(t) for t in texts])
        pre_ids, pre_att, last = ids[:, :-1], att[:, :-1], ids[:, -1:]
        posid = (pre_att.long().cumsum(-1)-1).clamp_min(0);lm.clear()
        o = lm.model(input_ids=pre_ids, attention_mask=pre_att, position_ids=posid, use_cache=True, logits_to_keep=1)
        return dict(cache=o.past_key_values, att=torch.cat([pre_att, torch.ones_like(last)], 1), last=last,
                    pos=pre_att.long().sum(-1, keepdim=True), L=pre_ids.shape[1])

    @torch.no_grad()
    def read(self, P, W):
        lm = self.lm;lm.clear();lm.read[LR] = None
        if W is not None:
            ww = W.to(lm.model.dtype);lm.edit[LW] = lambda h, ww=ww:h+ww[:, None, :]
        o = lm.model(input_ids=P['last'], attention_mask=P['att'], position_ids=P['pos'], past_key_values=P['cache'],
                     use_cache=True, logits_to_keep=1)
        P['cache'].crop(P['L']);h = lm.store[LR][:, -1].float();lm.clear()
        return h, o.logits[:, -1].float()

    def code(self, h):
        idx, coef, rec = self.lm.lens.decompose(LR, h-self.C.mu_r, K);return idx, coef, rec

    def write(self, idx, coef, g):
        u = self.lm.lens.synth(LW, idx, coef);return g*self.C.n_ref*u/u.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def z(self, rec):
        return ((rec-self.C.zmu)@self.C.Z.T)/self.C.zs

    def ls_on(self, h, idx):
        """Least-squares coefficients of the centred read on fixed atoms idx (B, K)."""
        A = self.D[idx].float();x = (h-self.C.mu_r)[:, :, None]
        G = A@A.transpose(1, 2)+1e-4*torch.eye(K, device=h.device);return torch.linalg.solve(G, A@x)[:, :, 0]


def load(fp32=False):
    if MODEL == '14b':fp32 = False                            # fp32 14B (59 GB) does not fit on a 48 GB card
    lm = LM(KEY, dtype=torch.float32 if fp32 else torch.bfloat16);return lm


def calib(lm, files):
    from e2 import load_calib
    return load_calib(lm, files['calib/calib.npz'])


@torch.no_grad()
def settle(L, P, g, T):
    """Free J loop from the natural state; returns per-step z, codes, the final (idx, coef), residual and logits."""
    h, lg = L.read(P, None);idx, coef, rec = L.code(h)
    Z, I, Cf = [L.z(rec).cpu().numpy()], [idx.cpu().numpy()], [coef.cpu().numpy()]
    for t in range(T):
        h, lg = L.read(P, L.write(idx, coef, g));idx, coef, rec = L.code(h)
        Z.append(L.z(rec).cpu().numpy());I.append(idx.cpu().numpy());Cf.append(coef.cpu().numpy())
    return np.stack(Z), np.stack(I), np.stack(Cf), (idx, coef, h, lg)


def stage_settle(a, out, fp32=False):
    files = v3_files(a.cache);ctxs = json.loads(open(files['contexts.json']).read());sel = pick_contexts(ctxs)
    lm = load(fp32);C = calib(lm, files);L = JLoop(lm, C);res = {};arr = {}
    for g in a.gains:
        Zs, Is, Cs = [], [], []
        for s in range(0, len(sel), BS):
            ii = sel[s:s+BS];P = L.prefix([ctxs[i]['text'] for i in ii]);Z, I, Cf, _ = settle(L, P, g, 64)
            Zs.append(Z);Is.append(I);Cs.append(Cf);del P;torch.cuda.empty_cache()
        Z, I, Cf = np.concatenate(Zs, 1), np.concatenate(Is, 1), np.concatenate(Cs, 1)
        late = Z[33:].astype(np.float64);step = np.abs(np.diff(late, axis=0)).max(axis=(0, 2))
        exact = (Z[-1] == Z[-2]).all(-1)&(Z[-2] == Z[-3]).all(-1)
        turn = np.mean([[1-len(set(I[t, i])&set(I[t+1, i]))/K for t in range(33, 64)] for i in range(Z.shape[1])], 1)
        d1 = np.linalg.norm(np.diff(late, axis=0), axis=-1);d2 = np.linalg.norm(late[2:]-late[:-2], axis=-1)
        r2 = np.median(d2/np.maximum(d1[1:], 1e-12), 0)
        res[str(g)] = dict(n=int(Z.shape[1]), exact_fixed=float(exact.mean()), late_maxstep_median=float(np.median(step)),
                           frac_late_step_lt_1e_3=float(np.mean(step < 1e-3)), atom_turnover_mean=float(turn.mean()),
                           period2_ratio_median=float(np.median(r2)))
        arr[f'z_{g}'] = Z.astype(np.float32);arr[f'idx_{g}'] = I.astype(np.int32);arr[f'coef_{g}'] = Cf.astype(np.float32)
        log('settle', 'fp32' if fp32 else 'bf16', g, res[str(g)])
    tag = 'fp32' if fp32 else 'bf16'
    atomic_npz(out/f'settle_{tag}.npz', sel=np.array(sel), **arr);atomic_json(out/f'settle_{tag}.json', dict(res=res, sel=sel))


@torch.no_grad()
def stage_jac(a, out, fp32=False, rels=(.02, .06)):
    """Local linear map at the state reached after T0 steps, coefficient space over its own atoms (central differences).
    fp32: the whole loop in an fp32 model (bf16 rounding makes small-step differences noisy); rels: step sizes."""
    files = v3_files(a.cache);ctxs = json.loads(open(files['contexts.json']).read());sel = pick_contexts(ctxs)
    lm = load(fp32);C = calib(lm, files);L = JLoop(lm, C);res = {};arr = {};words = lm.lens.words
    for g in a.gains:
        EV, VEC, IDX, C0, FID, R2 = [], [], [], [], [], []
        for s in range(0, len(sel), BS):
            ii = sel[s:s+BS];P = L.prefix([ctxs[i]['text'] for i in ii])
            _, _, _, (idx, coef, h, _) = settle(L, P, g, 32)
            c0 = L.ls_on(h, idx)                                   # the state's own coefficients on its atoms
            hb, _ = L.read(P, L.write(idx, c0.clamp_min(0), g));base = L.ls_on(hb, idx)
            Js = []
            for rel in rels:                                       # step sizes: linearity / rounding check
                eps = rel*c0.abs().mean(1, keepdim=True).clamp_min(1e-6);Jm = torch.zeros(len(ii), K, K, device=DEV)
                for k in range(K):
                    e = torch.zeros_like(c0);e[:, k] = 1.
                    hp, _ = L.read(P, L.write(idx, (c0+eps*e).clamp_min(0), g));hm, _ = L.read(P, L.write(idx, (c0-eps*e).clamp_min(0), g))
                    Jm[:, :, k] = (L.ls_on(hp, idx)-L.ls_on(hm, idx))/(2*eps)
                Js.append(Jm)
            Jm = Js[len(Js)//2];R2.append(np.stack([torch.linalg.eigvals(J_.double()).abs().max(1).values.cpu().numpy() for J_ in Js], 1))
            ev, vec = torch.linalg.eig(Jm.double())
            EV.append(ev.cpu().numpy());VEC.append(vec.cpu().numpy());IDX.append(idx.cpu().numpy());C0.append(c0.cpu().numpy())
            FID.append(((base-c0).norm(dim=1)/c0.norm(dim=1)).cpu().numpy());del P;torch.cuda.empty_cache()
        EV, VEC, IDX, C0, FID, R2 = map(np.concatenate, (EV, VEC, IDX, C0, FID, R2))
        mag = np.abs(EV);o = np.argsort(-mag, 1);EVs = np.take_along_axis(EV, o, 1)
        lead = EVs[:, 0];sr = np.abs(lead)
        cls = np.where(np.abs(lead.imag) > 1e-3, 'complex', np.where(lead.real < 0, 'negative', 'positive'))
        # words of the leading eigenvector per context (largest |loading|)
        lead_words = {}
        for i in range(len(sel)):
            v = VEC[i][:, o[i, 0]];top = np.argsort(-np.abs(v))[:4]
            for j in top:lead_words[words[int(IDX[i, j])]] = lead_words.get(words[int(IDX[i, j])], 0)+1
        res[str(g)] = dict(n=len(sel), spectral_radius_median=float(np.median(sr)), spectral_radius_q90=float(np.quantile(sr, .9)),
                           frac_radius_gt_0_9=float(np.mean(sr > .9)), frac_radius_gt_1=float(np.mean(sr > 1)),
                           lead_class={c:float(np.mean(cls == c)) for c in ('positive', 'negative', 'complex')},
                           lead_rotation_deg_median=float(np.median(np.abs(np.degrees(np.angle(lead))))),
                           n_modes_abs_gt_0_5_median=float(np.median((mag > .5).sum(1))),
                           self_reproduction_residual_median=float(np.median(FID)),
                           radius_median_by_step={str(r):float(np.median(R2[:, j])) for j, r in enumerate(rels)},
                           radius_step_agreement=[float(np.corrcoef(R2[:, j], R2[:, j+1])[0, 1]) for j in range(len(rels)-1)],
                           lead_words=sorted(lead_words.items(), key=lambda x:-x[1])[:30])
        arr[f'ev_{g}'] = EV;arr[f'idx_{g}'] = IDX;arr[f'c0_{g}'] = C0
        log('jac', g, {k:v for k, v in res[str(g)].items() if k != 'lead_words'})
    tag = 'jac32' if fp32 else 'jac'
    atomic_npz(out/f'{tag}.npz', sel=np.array(sel), **arr);atomic_json(out/f'{tag}.json', res)


@torch.no_grad()
def stage_pert(a, out, M=16, chunk=24):
    """Random coefficient perturbations at T0 = 32, free loop to T0 + 32; offsets vs the unperturbed continuation.
    Batched: chunk contexts x (M perturbed + 1 base) copies run together."""
    files = v3_files(a.cache);ctxs = json.loads(open(files['contexts.json']).read());sel = pick_contexts(ctxs)[::2]
    lm = load();C = calib(lm, files);L = JLoop(lm, C);res = {};arr = {};words = lm.lens.words
    gen = torch.Generator(device=DEV);gen.manual_seed(5)
    for g in a.gains:
        for eps in (.1, .3):
            OFF0, OFF, PERS, PR, WD = [], [], [], [], {}
            for s0 in range(0, len(sel), chunk):
                cs = sel[s0:s0+chunk];nb = len(cs)
                P1 = L.prefix([ctxs[i]['text'] for i in cs]);_, _, _, (idx, coef, h, _) = settle(L, P1, g, 32);del P1
                ii = idx.repeat_interleave(M+1, 0).clone();cc = coef.repeat_interleave(M+1, 0).clone()
                d = torch.randn(nb*(M+1), K, device=DEV, generator=gen);d = d/d.norm(dim=1, keepdim=True)
                d = d*eps*coef.norm(dim=1).repeat_interleave(M+1)[:, None];d[::M+1] = 0       # row 0 of each block: base
                cc = (cc+d).clamp_min(0)
                P = L.prefix([ctxs[i]['text'] for i in cs for _ in range(M+1)]);z0 = None
                for t in range(32):
                    hh, _ = L.read(P, L.write(ii, cc, g));ii, cc, rec = L.code(hh);zz = L.z(rec)
                    if t == 0:z0 = zz.clone()
                del P;torch.cuda.empty_cache()
                z0, zz = z0.view(nb, M+1, -1).cpu().numpy(), zz.view(nb, M+1, -1).cpu().numpy()
                I_, C_ = ii.view(nb, M+1, K).cpu().numpy(), cc.view(nb, M+1, K).cpu().numpy()
                for b in range(nb):
                    off0 = z0[b, 1:]-z0[b, :1];off = zz[b, 1:]-zz[b, :1]
                    pers = np.linalg.norm(off, axis=1)/np.maximum(np.linalg.norm(off0, axis=1), 1e-9)
                    s_ = np.linalg.svd(off-off.mean(0), compute_uv=False)**2;PR.append(float(s_.sum()**2/max((s_**2).sum(), 1e-30)))
                    OFF0.append(off0);OFF.append(off);PERS.append(pers)
                    base = dict(zip(I_[b, 0].tolist(), C_[b, 0].tolist()))
                    for m in range(1, M+1):
                        cur = dict(zip(I_[b, m].tolist(), C_[b, m].tolist()));ks = set(base)|set(cur)
                        for k in sorted(ks, key=lambda k:-abs(cur.get(k, 0)-base.get(k, 0)))[:2]:WD[words[int(k)]] = WD.get(words[int(k)], 0)+1
            PERS = np.stack(PERS)
            res[f'{g}|{eps}'] = dict(n_ctx=len(sel), M=M, persistence_median=float(np.median(PERS)), persistence_q90=float(np.quantile(PERS, .9)),
                                     frac_persist_gt_0_5=float(np.mean(PERS > .5)), offset_pr_median=float(np.median(PR)),
                                     changed_words=sorted(WD.items(), key=lambda x:-x[1])[:30])
            arr[f'off_{g}_{eps}'] = np.stack(OFF);arr[f'off0_{g}_{eps}'] = np.stack(OFF0)
            log('pert', g, eps, {k:v for k, v in res[f'{g}|{eps}'].items() if k != 'changed_words'})
    atomic_npz(out/'pert.npz', sel=np.array(sel), **arr);atomic_json(out/'pert.json', res)


def main():
    p = argparse.ArgumentParser();p.add_argument('--stage', required=True);p.add_argument('--out', required=True)
    p.add_argument('--gains', default='0.25,1,4');p.add_argument('--cache', default='/workspace/data')
    a = p.parse_args();a.gains = [float(x) for x in a.gains.split(',')];out = pathlib.Path(a.out);out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.stage == 'settle':stage_settle(a, out)
    elif a.stage == 'fp32':stage_settle(a, out, fp32=True)
    elif a.stage == 'jac':stage_jac(a, out)
    elif a.stage == 'jac32':stage_jac(a, out, fp32=True, rels=(.01, .03, .1))
    elif a.stage == 'pert':stage_pert(a, out)
    log('done', a.stage, f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
