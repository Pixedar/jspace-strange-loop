"""Pilot read-out: is the free-running recurrence wandering (not frozen, not memoryless), and can the model read its
own trajectory (does a self-model match its own past window better than the other rows' windows)?"""
import json, sys
import numpy as np

d = sys.argv[1]
z = np.load(f'{d}/traj.npz');m = json.load(open(f'{d}/meta.json', encoding='utf-8'))
rows = m['rows'];a = m['args'];K = a['K'];B = len(rows)
j = z['j'].astype(np.float32);N = j.shape[0]
unit = lambda x:x/np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)
ju = unit(j)
alpha = a['alpha'];zz = np.zeros_like(j);s = np.zeros_like(j[0])
for t in range(N):
    s = (1-alpha)*s+alpha*j[t];zz[t] = s
print(f"Rn {m['Rn']:.1f}  rj {m['rj']:.2f}  rc {m['rc']:.2f}  rj_x {m['rj_x']:.2f}  |z|/rc {np.linalg.norm(zz[20:], axis=-1).mean()/m['rc']:.2f}")
import re
wordy = lambda s:np.mean([bool(re.fullmatch(r"[A-Za-z',.!?;:-]+", w)) for w in s.split()] or [0])

print('\n== wandering: cos(j_t, j_t+lag), and cos to another row at the same t (chance)')
groups = {}
for b, r in enumerate(rows):groups.setdefault((r['cond'], r['g'], r['beta']), []).append(b)
for key, bs in groups.items():
    out = []
    for lag in (1, 2, 5, 10, 20, 40):
        out.append(np.mean([(ju[:-lag, b]*ju[lag:, b]).sum(-1)[10:].mean() for b in bs]))
    other = [b2 for b2 in range(B) if b2 not in bs]
    ch = np.mean([(ju[10:, b]*ju[10:, b2]).sum(-1).mean() for b in bs for b2 in other[:6]])
    texts = [m['texts'][t][bs[0]] for t in (N-3, N-2, N-1)]
    distinct = np.mean([len(set(zip(w[:-1], w[1:])))/max(1, len(w)-1) for t in range(N) for b in bs
                        for w in [m['texts'][t][b].split()]])
    wd = np.mean([wordy(m['texts'][t][b]) for t in range(N) for b in bs])
    print(f'{key}: lag1..40 ' + ' '.join(f'{x:.2f}' for x in out) + f' | chance {ch:.2f} | distinct-2 {distinct:.2f} | words {wd:.2f}')
    for tx in texts:print('     ', tx.replace('\n', ' ')[:150])

print('\n== reflection fidelity: self-model vs the mean recurrent state of its own window vs other rows\' windows')
NR = z['m'].shape[0]
win = np.stack([zz[r*K:(r+1)*K].mean(0) for r in range(NR)])               # (NR, B, d)
winu = unit(win)
for name in ['m']+[k for k in z.files if k.startswith('fid_')]:
    M = unit(z[name].astype(np.float32))
    own, oth, hit, n = [], [], 0, 0
    for r in range(NR):
        S = M[r]@winu[r].T                                                     # (B self-models, B windows)
        W = winu[r]@winu[r].T
        for b in range(B):
            cand = [b2 for b2 in range(B) if b2 == b or W[b, b2] < .999]
            own.append(S[b, b]);oth.append(np.mean([S[b, c] for c in cand if c != b]))
            hit += int(max(cand, key=lambda c:S[b, c]) == b);n += 1
    print(f'{name:12s} cos own {np.mean(own):.3f}  others {np.mean(oth):.3f}  retrieval {hit/n:.2f} (chance ~{1/B:.2f})')
desc = m['desc']
print('\n== answers ("Over these last moments, my mind has been ...")')
for b in range(0, B, max(1, B//8)):
    print(f"{rows[b]['cond']} g={rows[b]['g']} b={rows[b]['beta']} s={rows[b]['seed']}:")
    for r in (0, NR//2, NR-1):print('     ', desc[r][b].replace('\n', ' ')[:140])
