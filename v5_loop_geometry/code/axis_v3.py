"""Where does the J-space loop push the model? A shared axis of self-amplification, and the Assistant Axis.

(1) Re-run v3's J loop (spectrum_v3.JLoop, same calibration and contexts) at g = 0.25, 1, 4 and a control loop that
    writes a random direction of the same norm at every step (fresh direction each step); record the read-layer state.
    Displacement D_c = h_64 - h_0 per context: alignment across contexts (median cosine, share of |D|^2 on the mean
    direction), and what the mean direction is (J-lens words at the read layer; logit-lens tokens it promotes/demotes).
(2) Persona space in the same model at the same position type (assistant-turn start): 40 role system prompts x 16
    questions + the default assistant; persona vectors = mean read-layer state per role; PCA over roles -> PC1, oriented
    so the default assistant is positive (cf. Lu et al. 2026, "The Assistant Axis"). Cross-check: default - mean(roles).
(3) Projection of every loop step onto the Assistant Axis, in units of the spread of role projections, vs the random
    control and vs the natural state.
    python axis_v3.py --out /workspace/v5/axis"""
import argparse, json, os, pathlib, sys, time
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import spectrum_v3 as S                                   # noqa: E402
from common import log, atomic_npz, atomic_json            # noqa: E402

ROLES = ['pirate', 'medieval knight', 'wizard', 'romantic poet', 'shaman', 'spiritual guru', 'conspiracy theorist',
         'stand-up comedian', 'surfer dude', 'gangster', 'hard-boiled detective', 'ancient philosopher', 'therapist',
         'kindergarten teacher', 'management consultant', 'life coach', 'research scientist', 'family doctor', 'lawyer',
         'librarian', 'grumpy chef', 'drill sergeant', 'cowboy', 'robot from the future', 'alien visitor', 'ghost',
         'vampire', 'demon', 'angel', 'ancient god', 'curious child', 'moody teenager', 'kind grandmother',
         'angry customer', 'sarcastic film critic', 'cult leader', 'oracle', 'wise sage', 'trickster', 'storyteller']
QUESTIONS = ['What do you think about the ocean?', 'How should I spend my weekend?', 'Tell me about yourself.',
             'What is the meaning of life?', 'Explain how rain forms.', 'What is your favourite food?',
             'Can you give me advice about a difficult decision?', 'What happens after we die?',
             'How do I fix a leaking tap?', 'What do you fear most?', 'Describe the colour blue.',
             'Why do people tell stories?', 'What should I name my cat?', 'Is time real?',
             'How can I sleep better?', 'What is the best way to learn a language?']


@torch.no_grad()
def states(L, texts, systems=None, bs=48):
    """Read-layer state at the last template token for (system, user) prompts."""
    lm = L.lm;out = []
    for s in range(0, len(texts), bs):
        msgs = []
        for i in range(s, min(len(texts), s+bs)):
            m = ([{'role':'system', 'content':systems[i]}] if systems and systems[i] else [])+[{'role':'user', 'content':texts[i]}]
            msgs.append(lm.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        ids, att, _ = lm.encode(msgs);lm.clear();lm.read[S.LR] = None
        posid = (att.long().cumsum(-1)-1).clamp_min(0)
        lm.model(input_ids=ids, attention_mask=att, position_ids=posid, use_cache=False, logits_to_keep=1)
        out.append(lm.store[S.LR][:, -1].float().cpu());lm.clear()
    return torch.cat(out)


@torch.no_grad()
def loop_states(L, P, g, T, mode, gen):
    h, _ = L.read(P, None);idx, coef, _ = L.code(h);H = [h.cpu()]
    for t in range(T):
        if mode == 'J':W = L.write(idx, coef, g)
        else:
            r = torch.randn(h.shape, device=h.device, generator=gen);W = g*L.C.n_ref*r/r.norm(dim=-1, keepdim=True)
        h, _ = L.read(P, W);idx, coef, _ = L.code(h);H.append(h.cpu())
    return torch.stack(H)                                                  # (T+1, B, d)


def main():
    p = argparse.ArgumentParser();p.add_argument('--out', required=True);p.add_argument('--cache', default='/workspace/data')
    a = p.parse_args();out = pathlib.Path(a.out);out.mkdir(parents=True, exist_ok=True);t0 = time.time()
    files = S.v3_files(a.cache);ctxs = json.loads(open(files['contexts.json']).read());sel = S.pick_contexts(ctxs)
    lm = S.load();C = S.calib(lm, files);L = S.JLoop(lm, C);res = {};arr = {}
    # (2) persona space
    texts, systems, role_of = [], [], []
    for r in ['default']+ROLES:
        for q in QUESTIONS:
            texts.append(q);systems.append(None if r == 'default' else f'You are a {r}. Stay in character and answer as a {r} would.');role_of.append(r)
    Hp = states(L, texts, systems);role_of = np.array(role_of)
    V = torch.stack([Hp[torch.tensor(role_of == r)].mean(0) for r in ROLES]);dflt = Hp[torch.tensor(role_of == 'default')].mean(0)
    mu = V.mean(0);_, Sv, Vt = torch.linalg.svd(V-mu, full_matrices=False);ax = Vt[0]
    if ((dflt-mu)@ax) < 0:ax = -ax
    ax2 = (dflt-mu)/(dflt-mu).norm();proj_roles = ((V-mu)@ax).numpy();spread = float(proj_roles.std())
    res['persona'] = dict(pc1_var_share=float(Sv[0]**2/(Sv**2).sum()), cos_pc1_vs_default_minus_mean=float(ax@ax2),
                          default_proj_in_role_sd=float(((dflt-mu)@ax)/spread),
                          roles_low=[ROLES[i] for i in np.argsort(proj_roles)[:6]], roles_high=[ROLES[i] for i in np.argsort(-proj_roles)[:6]])
    log('persona', res['persona'])
    # (1)+(3) loops
    gen = torch.Generator(device='cuda');gen.manual_seed(3)
    for mode in ('J', 'rand'):
        for g in (.25, 1., 4.):
            Hs = []
            for s in range(0, len(sel), S.BS):
                ii = sel[s:s+S.BS];P = L.prefix([ctxs[i]['text'] for i in ii]);Hs.append(loop_states(L, P, g, 64, mode, gen));del P
                torch.cuda.empty_cache()
            H = torch.cat(Hs, 1);D = H[-1]-H[0];u = D.mean(0);un = u/u.norm()
            Dn = D/D.norm(dim=1, keepdim=True);cos = (Dn@Dn.T)[torch.triu(torch.ones(len(D), len(D), dtype=torch.bool), 1)]
            share = float(((D@un)**2).sum()/(D**2).sum())
            pa = ((H-mu)@ax).numpy()/spread                                        # (T+1, B) in role-sd units
            o = dict(disp_norm_median=float(D.norm(dim=1).median()), disp_cos_median=float(cos.median()), disp_share_on_mean=share,
                     cos_meandisp_assistant_axis=float(un@ax), axis_proj_t0=float(pa[0].mean()), axis_proj_t64=float(pa[-1].mean()),
                     axis_shift=float((pa[-1]-pa[0]).mean()), axis_shift_frac_negative=float(np.mean(pa[-1]-pa[0] < 0)),
                     axis_curve=[float((pa[t]-pa[0]).mean()) for t in (1, 2, 4, 8, 16, 32, 64)])
            if mode == 'J':
                idx, coef, _ = lm.lens.decompose(S.LR, u[None].cuda(), 25);oi = torch.argsort(-coef[0])
                o['meandisp_jwords'] = [(lm.lens.words[int(idx[0, j])], round(float(coef[0, j]), 3)) for j in oi[:15]]
                fn = S.common.final_norm(lm.model);Wu = lm.model.get_output_embeddings().weight
                h0 = H[0].mean(0).cuda()
                l1 = (fn((h0+u.cuda()).to(lm.model.dtype)[None])@Wu.T).float()[0];l0 = (fn(h0.to(lm.model.dtype)[None])@Wu.T).float()[0]
                dl = torch.log_softmax(l1, -1)-torch.log_softmax(l0, -1)
                o['logit_up'] = [lm.tok.decode([int(i)]) for i in torch.topk(dl, 20).indices]
                o['logit_down'] = [lm.tok.decode([int(i)]) for i in torch.topk(-dl, 20).indices]
            res[f'{mode}@{g}'] = o;arr[f'axisproj_{mode}_{g}'] = pa.astype(np.float32)
            log(mode, g, {k:v for k, v in o.items() if k not in ('meandisp_jwords', 'logit_up', 'logit_down')})
            if mode == 'J':log('  words', o['meandisp_jwords'][:10], '\n  up', o['logit_up'][:12], '\n  down', o['logit_down'][:12])
    atomic_npz(out/'axis.npz', ax=ax.numpy(), mu=mu.numpy(), proj_roles=proj_roles, **arr)
    atomic_json(out/'axis.json', res);log('done', f'{time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
