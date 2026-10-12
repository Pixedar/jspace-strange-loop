"""Phase 0: mechanism of the J-adapter action switching in the historical 64-pass loops (self-workspace study,
Qwen3-4B @ 1cfa9a72, layer 25, K = 25 English dictionary). Uses the ORIGINAL JSpace implementation (strange_loop.py)
and the saved adapter checkpoint, re-decomposes the saved incoming residuals and asks:
 1. Does recomputation reproduce the saved predictions (pipeline check)?
 2. Do action switches coincide with changes of the selected atom support, while the residual changes smoothly?
 3. Open-loop counterfactual: with the atom support frozen at pass 0 (coefficients still refit), do the head's action
    choices along the same state trajectory still switch?
 4. Same for the residual adapter (continuous features) as a reference.
    python phase0.py /workspace/phase0"""
import sys, json, pathlib
import numpy as np, torch

P = pathlib.Path(sys.argv[1]);sys.path.insert(0, str(P))
from strange_loop import JSpace
from analyze_train import Predictor
from common import OUT, atomic_json, log

REV = '1cfa9a7208912126459214e8b04321603b3df60c'


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import hf_hub_download
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B', revision=REV)
    try:model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B', revision=REV, dtype=torch.float32, device_map='cuda')
    except TypeError:model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B', revision=REV, torch_dtype=torch.float32, device_map='cuda')
    lens = hf_hub_download('neuronpedia/jacobian-lens', 'qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt',
                           revision='9fa267f841d62aa468dab5b9b38c3dc45a82d863')
    J = torch.load(lens, map_location='cpu', weights_only=True)['J'][25]
    js = JSpace(model, tok, J, k=25, vocab='english', center=True)
    log('dictionary', len(js.words))
    out = {}
    for tag, prefix in (('causal', ''), ('residual_causal', 'res_')):
        ck = torch.load(P/'full_future'/f'{tag}_cycle1.pt', map_location='cuda', weights_only=False)['state']
        net = Predictor(2560).cuda();net.load_state_dict({k[4:]:v for k, v in ck.items() if k.startswith('net.')})
        fmu, sd, ym, ys, center = (ck[k].float().cuda() for k in ('feature_mu', 'sd', 'ym', 'ys', 'center'))
        rows = [];per_item = []
        for i in range(12):
            z = np.load(P/'full_future'/'dynamics'/'causal'/f'{prefix}{i:05d}_own0.npz')
            H = torch.tensor(z['h'], device='cuda').float();T = H.shape[0];ref = torch.tensor(z['reference'], device='cuda').float()
            belief_sign = np.sign(np.tanh(z['belief']/2))
            for k in range(3):
                h = H[:, k]
                if tag == 'causal':
                    idx, coef, j = js.decompose(h-center)
                    idx = idx.cpu().numpy();sup0 = torch.tensor(idx[0], device='cuda')
                    # fixed support: atoms of pass 0, non-negative least squares on that support only
                    A = js.D[sup0].float();G = A@A.T+1e-4*torch.eye(25, device='cuda');b = (h-center)@A.T
                    from common import nnls_fista
                    Gb = G[None].expand(b.shape[0], -1, -1).contiguous()
                    cf = nnls_fista(Gb, b, torch.linalg.solve(G, b.T).T, iters=2000);jfix = cf@A   # true NNLS on the frozen support
                else:
                    j = h;idx = None;jfix = h
                with torch.no_grad():
                    p, _ = net((j-fmu)/sd, ref[k].expand(T)/10);p = p*ys+ym;p[:, 0] = 0
                    pf, _ = net((jfix-fmu)/sd, ref[k].expand(T)/10);pf = pf*ys+ym;pf[:, 0] = 0
                p = p.cpu().numpy();pf = pf.cpu().numpy();saved = z['prediction'][:, k]
                bs = np.sign(z['memory'][:, k, 9])[:, None]          # belief sign at each pass (memory dim 9 = tanh(b/5))
                act_re = (bs*p).argmax(1);act_fix = (bs*pf).argmax(1);act_saved = z['actions'][:, k]
                dh = np.r_[np.nan, (np.linalg.norm(np.diff(z['h'][:, k], axis=0), axis=1)/np.linalg.norm(z['h'][1:, k], axis=1))]
                if idx is not None:
                    jac = np.r_[np.nan, [1-len(set(idx[t])&set(idx[t-1]))/25 for t in range(1, T)]]
                else:jac = np.full(T, np.nan)
                sw = np.r_[False, act_saved[1:] != act_saved[:-1]]
                rows.append(dict(item=i, state=k, pred_max_abs_err=float(np.abs(p-saved).max()),
                                 action_reproduced=float((act_re == act_saved).mean()),
                                 switches_saved=int(sw[32:].sum()), switches_fixed_support=int((act_fix[33:] != act_fix[32:-1]).sum()),
                                 atom_turnover_at_switch=float(np.nanmean(jac[sw])) if sw.any() else np.nan,
                                 atom_turnover_no_switch=float(np.nanmean(jac[~sw&(np.arange(T) > 0)])),
                                 dh_at_switch=float(np.nanmean(dh[sw])) if sw.any() else np.nan,
                                 dh_no_switch=float(np.nanmean(dh[~sw&(np.arange(T) > 0)]))))
                if tag == 'causal' and i == 0 and k == 2:
                    per_item.append(dict(actions=act_saved[:24].tolist(), atom_turnover=np.round(jac[:24], 2).tolist(),
                                         dh=np.round(dh[:24], 4).tolist(),
                                         words_pass=[js.top(idx[t], coef[t].cpu().numpy(), 8) for t in range(3, 6)]))
        R = {k:float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k not in ('item', 'state')}
        R['states_switching_saved'] = int(sum(r['switches_saved'] > 0 for r in rows))
        R['states_switching_fixed_support'] = int(sum(r['switches_fixed_support'] > 0 for r in rows))
        out[tag] = dict(summary=R, rows=rows, example=per_item)
        log(tag, R)
    atomic_json(OUT/'phase0'/'results.json', out)


if __name__ == '__main__':
    main()
