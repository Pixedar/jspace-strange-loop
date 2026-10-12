"""Quick look at an E2 run directory: per condition and gain, how much the workspace changes, whether the memory test
retains the initial evidence, accuracy over steps, and late-window trajectory classes (protocol section 4 definitions).
    python an_e2_quick.py /workspace/v3/e2/qwen3-8b/pilot [A1|A2]"""
import sys, json, pathlib
import numpy as np

root = pathlib.Path(sys.argv[1]);stage = sys.argv[2] if len(sys.argv) > 2 else 'A1'
ctxs = json.loads((root/'contexts.json').read_text())


def classes(z, rec):
    """z (T+1, n, 32) compact state; rec unused. Late window = last 16 steps."""
    T = z.shape[0]-1;w = z[T-15:].astype(np.float32)
    out = []
    for i in range(z.shape[1]):
        x = w[:, i];nx = np.linalg.norm(x, axis=1)+1e-9
        cons = 1-(x[1:]*x[:-1]).sum(1)/(nx[1:]*nx[:-1])
        if cons.max() < 0.01:out.append('fixed');continue
        per = None
        for p in (2, 3, 4):
            c = (x[p:]*x[:-p]).sum(1)/(nx[p:]*nx[:-p])
            if c.min() > 0.99:per = p;break
        out.append(f'cycle{per}' if per else 'aperiodic')
    return out


agg = {}
for f in (sorted((root/stage).glob(f'{stage}_*.npz')) if stage in ('A1', 'A2') else []):
    kind = f.name.split('_')[1];m = json.loads(f.with_suffix('.json').read_text())
    zz = np.load(f);z = {k:zz[k] for k in zz.files if k != 'H'};rows = m['rows'];units = m['units'];T = z['idx'].shape[0]-1
    cl = classes(z['z'], None)
    nat = {r['unit']:set(z['idx'][1, i].tolist()) for i, r in enumerate(rows) if r['cond'] == 'nofb'}
    for i, r in enumerate(rows):
        key = (kind, r['cond'], r['g']);a = agg.setdefault(key, dict(ov=[], step=[], lo=[], truth=[], cls=[], tag=[], fam=[], sid=[], wn=[]))
        ii = z['idx'][:, i]
        a['ov'].append([len(set(ii[t].tolist())&nat.get(r['unit'], set()))/25 for t in (1, 4, 16, T)])
        a['step'].append(np.mean([1-len(set(ii[t].tolist())&set(ii[t-1].tolist()))/25 for t in range(T-15, T+1)]))
        a['lo'].append(z['lo'][:, i]);u = units[r['unit']];a['truth'].append(ctxs[u['init']]['truth'])
        a['cls'].append(cl[i]);a['tag'].append(u['tag']);a['fam'].append(u['fam']);a['sid'].append(u['sid'])
        a['wn'].append(float(z['wnorm'][:, i].mean()))
if stage in ('A1', 'A2'):print(f'{"kind":4s} {"cond":8s} {"g":>5s} | ovl_nat t1 t4 t16 tT | late turnover | wnorm | classes | acc(t0,t4,t16,tT) | memory gap lo(A)-lo(B) t0 t1 t4 t16 tT')
for key in sorted(agg):
    a = agg[key];kind, c, g = key;ov = np.array(a['ov']).mean(0);lo = np.array(a['lo']);truth = np.array(a['truth'])
    T = lo.shape[1]-1;cls = {k:a['cls'].count(k) for k in sorted(set(a['cls']))}
    ok = truth >= 0
    acc = [float(((lo[ok, t] > 0) == (truth[ok] == 0)).mean()) if ok.any() else np.nan for t in (0, 4, 16, T)]
    mem = ''
    if kind == 'mem':
        tags = np.array(a['tag']);sids = np.array(a['sid'])
        gaps = []
        for s in sorted(set(sids)):
            ia = np.where((sids == s)&(tags == 'A'))[0];ib = np.where((sids == s)&(tags == 'B'))[0]
            if len(ia) and len(ib):gaps.append(lo[ia[0]]-lo[ib[0]])
        gaps = np.array(gaps);mem = ' '.join(f'{gaps[:, t].mean():6.2f}' for t in (0, 1, 4, 16, T))
    print(f'{kind:4s} {c:8s} {g:5.2f} | {" ".join(f"{v:.2f}" for v in ov)} | {np.mean(a["step"]):.3f} | {np.mean(a["wn"]):6.1f} | {cls} | '
          f'{" ".join(f"{v:.2f}" for v in acc)} | {mem}')


def bc_summary(root):
    rows_all = {}
    for f in sorted((root/'BC').glob('BC*_*.npz')):
        m = json.loads(f.with_suffix('.json').read_text());zz = np.load(f);z = {k:zz[k] for k in zz.files if k != 'H'};kind = m['kind']
        for i, r in enumerate(m['rows']):
            k = (kind, r['cond'], r['g']);a = rows_all.setdefault(k, dict(pe=[], act=[], turn=[], lo=[], z=[]))
            a['pe'].append(z['perr'][:, i]);a['act'].append(z['act'][:, i])
            ii = z['idx'][:, i];T = ii.shape[0]-1
            a['turn'].append([1-len(set(ii[t].tolist())&set(ii[t-1].tolist()))/25 for t in range(1, T+1)])
            a['z'].append(z['z'][:, i].astype(np.float32))
    print(f'{"kind":4s} {"cond":4s} {"g":>5s} | perr t0-8 t8-16 t16-32 t48-64 | turnover early late | action entropy late | z step late')
    for k in sorted(rows_all):
        a = rows_all[k];pe = np.array(a['pe']);act = np.array(a['act']);tu = np.array(a['turn']);zz = np.array(a['z'])
        T = pe.shape[1];w = lambda x, s, e:float(np.nanmean(x[:, s:e]))
        late = act[:, T//2:].ravel();late = late[late >= 0];p = np.bincount(late, minlength=9)/max(1, len(late))
        H = float(-(p[p > 0]*np.log(p[p > 0])).sum())
        dz = np.linalg.norm(zz[:, 1:]-zz[:, :-1], axis=-1)
        print(f'{k[0]:4s} {k[1]:4s} {k[2]:5.2f} | {w(pe,0,8):.3f} {w(pe,8,16):.3f} {w(pe,16,32):.3f} {w(pe,48,64):.3f} | '
              f'{w(tu,0,8):.3f} {w(tu,48,64):.3f} | {H:.2f} | {float(dz[:, T//2:].mean()):.3f}')


if stage == 'BC':
    bc_summary(root)
