"""Figures from the compact result files (run where the results live). Writes PNGs to <root>/figs.
    python figs.py /workspace/v3"""
import sys, json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path(sys.argv[1]);OUTF = ROOT/'figs';OUTF.mkdir(exist_ok=True)
C = dict(full='#3b5b92', J='#d1495b', rem='#66a182', rand='#9a9a9a', none='#cccccc', residual='#3b5b92', remainder='#66a182',
         J25='#d1495b', pca25='#edae49', rp25='#9a9a9a', dense='#3b5b92', pca='#edae49', shuf='#8e7dbe', foreign='#00798c',
         tshuf='#b56576', Jpb='#e07a5f', nofb='#555555', B='#d1495b', C1='#00798c', C2='#edae49', R='#9a9a9a', F='#3b5b92', Y='#8e7dbe')
plt.rcParams.update({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False, 'figure.dpi':130})


def save(fig, name):
    fig.tight_layout();fig.savefig(OUTF/name);plt.close(fig);print('wrote', OUTF/name)


def fig_e1(model, tag):
    f = ROOT/'e1'/model/tag/'results_e1.json'
    if not f.exists():return
    r = json.loads(f.read_text())
    if 'H1' in r:
        P = r['H1']['patch'];layers = sorted(P['decision'], key=int)
        fig, ax = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
        for a, name, title in zip(ax, ('context', 'scenario_end', 'decision'),
                                  ('identical-token context positions', 'scenario end (one position)', 'decision position (one position)')):
            for c in ('full', 'rem', 'J', 'rand'):
                y = [P[name][l][c]['shift'] for l in layers];lo = [P[name][l][c]['ci'][0] for l in layers];hi = [P[name][l][c]['ci'][1] for l in layers]
                a.plot([int(l) for l in layers], y, 'o-', color=C[c], label={'full':'full residual', 'rem':'remainder (non-J)', 'J':'J component', 'rand':'random J'}[c])
                a.fill_between([int(l) for l in layers], lo, hi, color=C[c], alpha=.15)
            a.set_title(title, fontsize=9);a.set_xlabel('layer');a.axhline(0, color='k', lw=.5)
        ax[0].set_ylabel('shift toward the swapped answer\n(fraction of the way)');ax[2].legend(frameon=False, fontsize=8)
        fig.suptitle(f'H1 - where is the binding / answer carried? ({model})', fontsize=10);save(fig, f'e1_h1_patch_{model}_{tag}.png')
    if 'H2' in r:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
        for a, fam in zip(ax, ('F2', 'F3')):
            q = r['H2'].get(fam)
            if not q:continue
            keys = sorted(q['by_representation']);reps = ['J25', 'residual', 'remainder', 'pca25', 'rp25'];x = np.arange(len(keys));w = .16
            for i, rep in enumerate(reps):
                y = [q['by_representation'][k][rep].get('gain_eligible_mean', np.nan) for k in keys]
                e = [q['by_representation'][k][rep].get('ci', [np.nan, np.nan]) for k in keys]
                a.bar(x+(i-2)*w, y, w, color=C[rep], label=rep, yerr=np.abs(np.array(e).T-np.array(y)), ecolor='#444', capsize=1.5, lw=0)
            a.set_xticks(x);a.set_xticklabels([k.replace('_', '\n') for k in keys], fontsize=7);a.axhline(0, color='k', lw=.5)
            a.set_title(f'{fam}: disposition prediction beyond the current answer', fontsize=9)
            a.set_ylabel('held-out within-scenario R$^2$ gain')
        ax[0].legend(frameon=False, fontsize=7, ncol=5);save(fig, f'e1_h2_gain_{model}_{tag}.png')
        fig, ax = plt.subplots(1, 2, figsize=(9, 3))
        for a, fam in zip(ax, ('F2', 'F3')):
            q = r['H2'].get(fam)
            if not q:continue
            k = max(q['by_representation'], key=lambda kk:q['by_representation'][kk]['residual'].get('gain_eligible_mean', -9))
            for fam_rep, col in (('J', C['J']), ('pca', C['pca25']), ('rp', C['rp25'])):
                ks = [8, 16, 25, 32, 64] if fam_rep == 'J' else [8, 16, 25, 32, 64]
                y = [q['by_representation'][k].get(f'{fam_rep}{kk}', {}).get('gain_eligible_mean', np.nan) for kk in ks]
                a.plot(ks, y, 'o-', color=col, label={'J':'J (K atoms)', 'pca':'PCA (k dims)', 'rp':'random proj. (k)'}[fam_rep])
            a.axhline(q['by_representation'][k]['residual'].get('gain_eligible_mean', np.nan), color=C['residual'], ls='--', label='full residual')
            a.set_xscale('log', base=2);a.set_xlabel('capacity');a.set_title(f'{fam} capacity sweep at {k}', fontsize=9);a.legend(frameon=False, fontsize=7)
        save(fig, f'e1_h2_capacity_{model}_{tag}.png')


def fig_e2(model, tag):
    f = ROOT/'e2'/model/tag/'results_e2.json'
    if not f.exists():return
    r = json.loads(f.read_text())
    A = {**r.get('A1', {}), **r.get('A2', {})}
    fams = sorted({fm for k, v in A.items() if k.startswith('mem') for fm in v.get('memory_gap', {})})
    if fams:
        fig, ax = plt.subplots(1, len(fams), figsize=(3.6*len(fams), 3.1), sharey=False)
        ax = np.atleast_1d(ax)
        for a, fm in zip(ax, fams):
            for k, v in sorted(A.items()):
                if not k.startswith('mem') or fm not in v.get('memory_gap', {}):continue
                _, c, g = k.split('|');g = float(g)
                if c == 'J' or (c in ('dense', 'pca', 'rand', 'shuf', 'foreign') and g == max(float(kk.split('|')[2]) for kk in A if kk.split('|')[1] == c)):
                    mg = v['memory_gap'][fm];ts = sorted(mg, key=int);y = [mg[t][0] for t in ts]
                    a.plot([max(int(t), .5) for t in ts], y, 'o-', ms=3, color=C.get(c, '#000'), alpha=.4+.6*(g/4 if c == 'J' else 1),
                           label=f'{c} g={g:g}')
            a.set_xscale('log');a.axhline(0, color='k', lw=.5);a.set_title(f'{fm}: memory of evidence carried only by the loop', fontsize=8)
            a.set_xlabel('loop step');a.set_ylabel('log-odds(A init) - log-odds(B init)')
        ax[-1].legend(frameon=False, fontsize=6, ncol=2);save(fig, f'e2_memory_{model}_{tag}.png')
    # change vs structure map
    pts = [(k, v) for k, v in A.items() if k.startswith('mem') and 'evidence_decodability' in v]
    if pts:
        fig, a = plt.subplots(figsize=(5.6, 4))
        for k, v in pts:
            _, c, g = k.split('|');x = v['late_z_step'];dd = v['evidence_decodability'];y = dd[max(dd, key=int)]
            s = 20+80*min(v.get('context_contraction', 1), 2)
            a.scatter(x, y, s=s, color=C.get(c, '#000'), alpha=.75, edgecolor='k', lw=.3);a.annotate(f'{c} {float(g):g}', (x, y), fontsize=6)
        for coll in a.collections:                                   # pin no-feedback (step 0) at the left edge of a log axis
            off = coll.get_offsets();off[:, 0] = np.maximum(off[:, 0], 1e-3);coll.set_offsets(off)
        for t in a.texts:
            x, y = t.get_position();t.set_position((max(x, 1e-3), y))
        a.axhline(.5, color='k', lw=.6, ls=':');a.text(1.1e-3, .505, 'chance', fontsize=7, color='#555')
        a.set_xscale('log');a.set_xlim(7e-4, 3);a.set_ylim(.4, 1.0)
        a.set_xlabel('late change per step (log; no feedback at left)')
        a.set_ylabel('how much of the initial evidence survives\n(held-out decoding of A vs B at step 64)')
        a.set_title('Between collapse and noise (marker size = spread between contexts)', fontsize=9)
        save(fig, f'e2_change_structure_{model}_{tag}.png')
    if 'BC' in r or 'BC1' in r:
        B = r.get('BC') or r['BC1'];fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
        for c in ('B', 'C1', 'C2', 'R', 'F', 'Y'):
            ks = [k for k in B if k.startswith('std|'+c+'|')]
            for k in ks:
                g = float(k.split('|')[2]);pe = B[k]['perr'];zs = B[k]['z_step']
                ax[0].plot([0, 1, 2], [pe[w][0] for w in ('early', 'mid', 'late')], 'o-', color=C[c], alpha=.5 if g == min(float(kk.split('|')[2]) for kk in ks) else 1, label=f'{c} g={g:g}')
                ax[1].plot([0, 1, 2], [zs[w][0] for w in ('early', 'mid', 'late')], 'o-', color=C[c], alpha=.5 if g == min(float(kk.split('|')[2]) for kk in ks) else 1)
        for a, t in zip(ax, ('self-model prediction error', 'workspace change per step')):
            a.set_xticks([0, 1, 2]);a.set_xticklabels(['steps 0-8', '16-32', '48-64']);a.set_title(t, fontsize=9)
        ax[0].legend(frameon=False, fontsize=6, ncol=2);save(fig, f'e2_selfmodel_{model}_{tag}.png')
    if 'pert' in r:
        P = r['pert'];fig, ax = plt.subplots(1, 2, figsize=(9, 3))
        for k, v in P.items():
            kind, g = k.split('|');ts = sorted(v['dist'], key=int)
            ax[0].plot([int(t) for t in ts], [v['dist'][t] for t in ts], 'o-', label=f'{kind} g={float(g):g}')
            ax[1].plot([int(t) for t in ts], [v['abs_dlo'][t] for t in ts], 'o-')
        ax[0].set_yscale('log');ax[0].set_title('distance from unperturbed twin (z)', fontsize=9);ax[1].set_title('|answer log-odds difference|', fontsize=9)
        ax[0].legend(frameon=False, fontsize=6);save(fig, f'e2_perturb_{model}_{tag}.png')


def fig_traj(model, tag):
    """Workspace trajectories (PCA-2 of the compact state z) of A- and B-initialized loops run in the same neutral
    context, for the J loop at g1 / g2 and the dense control; one panel per condition, a few scenarios per family."""
    d = ROOT/'e2'/model/tag
    if not (d/'A2').exists():return
    import glob
    fs = sorted((d/'A2').glob('A2_mem_*.npz'))[:2]
    if not fs:return
    zs, meta = [], []
    for f in fs:
        z = np.load(f);m = json.loads(f.with_suffix('.json').read_text());zs.append(z['z'].astype(np.float32));meta.append(m)
    gains = sorted({r['g'] for m in meta for r in m['rows'] if r['cond'] == 'J'})
    panels = [('J', g) for g in gains]+[('dense', gains[-1])]
    allz = np.concatenate([z.reshape(-1, z.shape[-1]) for z in zs]);mu = allz.mean(0)
    U, S, Vt = np.linalg.svd(allz[::7]-mu, full_matrices=False);P = Vt[:2]
    fig, ax = plt.subplots(1, len(panels), figsize=(3.6*len(panels), 3.3))
    for a, (c, g) in zip(np.atleast_1d(ax), panels):
        k = 0
        for z, m in zip(zs, meta):
            for i, r in enumerate(m['rows']):
                if r['cond'] != c or r['g'] != g:continue
                u = m['units'][r['unit']]
                if u['fam'] not in ('F2', 'F6') or int(u['sid'].split('-')[-1]) % 8:continue
                y = (z[:, i]-mu)@P.T;col = '#d1495b' if u['tag'] == 'A' else '#3b5b92'
                a.plot(y[:, 0], y[:, 1], '-', color=col, lw=.6, alpha=.6);a.plot(y[0, 0], y[0, 1], 'o', color=col, ms=3)
                a.plot(y[-1, 0], y[-1, 1], 'x', color='k', ms=4);k += 1
        a.set_title(f'{c} g={g:g} (red A-init, blue B-init; x = step 64)', fontsize=7);a.set_xticks([]);a.set_yticks([])
    save(fig, f'e2_trajectories_{model}_{tag}.png')


def fig_huginn(tag='main'):
    f = ROOT/'huginn'/tag/'results_huginn.json'
    if not f.exists():return
    r = json.loads(f.read_text());fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    for fm, v in r['accuracy_by_recurrence'].items():
        ks = sorted(v, key=int);ax[0].plot([max(int(k), .5) for k in ks], [v[k][0] for k in ks], 'o-', label=fm)
    ax[0].set_xscale('log');ax[0].set_title('accuracy (exit lens) vs recurrence', fontsize=9);ax[0].legend(frameon=False, fontsize=7)
    d = r['evidence_decodability_from_decision_state'];ks = sorted(d, key=int)
    ax[1].plot([max(int(k), .5) for k in ks], [d[k] for k in ks], 'ko-');ax[1].set_xscale('log');ax[1].set_title('evidence decodable from decision state', fontsize=9)
    m = r['memory'].get('k16_transplant_gap', {})
    for fm, v in m.items():
        ts = sorted(v, key=int);ax[2].plot([int(t) for t in ts], [v[t][0] for t in ts], 'o-', label=fm)
    ax[2].axhline(0, color='k', lw=.5);ax[2].set_title('decision-state transplant at k=16: gap A-B', fontsize=9);ax[2].legend(frameon=False, fontsize=7)
    save(fig, f'huginn_{tag}.png')


def fig_ouro(tag='main'):
    f = ROOT/'ouro'/tag/'results_ouro.json'
    if not f.exists():return
    r = json.loads(f.read_text());fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for fm, v in r['accuracy_by_loop'].items():
        ks = sorted(v, key=int);ax[0].plot([int(k) for k in ks], [v[k][0] for k in ks], 'o-', label=fm)
    ax[0].axvline(4.5, color='k', ls=':', lw=.7);ax[0].set_title('Ouro accuracy per loop (dotted: trained depth)', fontsize=9)
    ax[0].legend(frameon=False, fontsize=7);ax[0].set_xlabel('loop')
    for k0, ls in (('k0', '-'), ('k1', '--'), ('k2', ':')):
        m = r['memory'].get(f'{k0}_transplant_gap', {})
        for fm, v in m.items():
            ks = sorted(v, key=int);ax[1].plot([int(k) for k in ks], [v[k][0] for k in ks], ls, marker='o', ms=3, label=f'{fm} at end of loop {int(k0[1])+1}')
    ax[1].axhline(0, color='k', lw=.5);ax[1].set_title('decision-state transplant: gap A-B per loop', fontsize=9);ax[1].legend(frameon=False, fontsize=6)
    save(fig, f'ouro_{tag}.png')


def fig_x(model, tag):
    """Exploratory arm X: memory carried by the loop and late change, for the self-forecast feedbacks."""
    f = ROOT/'e2'/model/tag/'results_e2.json'
    if not f.exists():return
    r = json.loads(f.read_text());X = r.get('X')
    if not X:return
    cols = dict(nofb='#555555', J='#d1495b', inc='#00798c', out='#edae49', lens='#8e7dbe')
    gains = sorted({float(k.split('|')[2]) for k in X if k.split('|')[1] != 'nofb'})
    fams = [fm for fm in ('F2', 'F6', 'F3') if any(fm in v.get('memory_gap', {}) for k, v in X.items() if k.startswith('mem'))]
    fig, ax = plt.subplots(1, len(fams)+1, figsize=(3.4*(len(fams)+1), 3.1))
    for a, fm in zip(ax, fams):
        for k, v in sorted(X.items()):
            kind, c, g = k.split('|');g = float(g)
            if kind != 'mem' or fm not in v.get('memory_gap', {}) or (c != 'nofb' and g != gains[-1]):continue
            mg = v['memory_gap'][fm];ts = sorted(mg, key=int)
            a.plot([max(int(t), .5) for t in ts], [mg[t][0] for t in ts], 'o-', ms=3, color=cols.get(c, '#000'), label=f'{c}' + ('' if c == 'nofb' else f' g={g:g}'))
        a.set_xscale('log');a.axhline(0, color='k', lw=.5);a.set_title(f'{fm}: memory carried by the loop', fontsize=8);a.set_xlabel('loop step')
    ax[0].set_ylabel('log-odds(A init) - log-odds(B init)');ax[0].legend(frameon=False, fontsize=6)
    a = ax[-1];xs = [];lab = []
    for c in ('nofb', 'J', 'inc', 'out', 'lens'):
        for g in ([0.0] if c == 'nofb' else gains):
            v = X.get(f'std|{c}|{g}') or X.get(f'std|{c}|{g:g}')
            if not v:continue
            xs.append(v['late_z_step']);lab.append((c, g, v.get('late_turnover', np.nan)))
    a.bar(range(len(xs)), [max(x, 1e-4) for x in xs], color=[cols.get(c, '#000') for c, g, _ in lab])
    a.set_xticks(range(len(xs)));a.set_xticklabels([c if c == 'nofb' else f'{c} {g:g}' for c, g, _ in lab], fontsize=6, rotation=60)
    a.set_yscale('log');a.set_title('late change per step (standard contexts)', fontsize=8)
    save(fig, f'e2_xarm_{model}_{tag}.png')


def fig_bcs(model, tag):
    """Exploratory strong coupling (stage BCs, g = 1 / 2) next to the pre-declared gains."""
    f = ROOT/'e2'/model/tag/'results_e2.json'
    if not f.exists():return
    r = json.loads(f.read_text())
    if 'BCs' not in r:return
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for st, ls in (('BC', ':'), ('BCs', '-')):
        B = r.get(st, {})
        for c in ('B', 'C1', 'C2', 'R', 'F', 'Y'):
            ks = sorted([k for k in B if k.startswith('std|'+c+'|')], key=lambda k:float(k.split('|')[2]))
            if not ks:continue
            gs = [float(k.split('|')[2]) for k in ks]
            ax[0].plot(gs, [B[k]['perr']['late'][0] for k in ks], ls, marker='o', ms=3, color=C[c], label=c if st == 'BCs' else None)
            ax[1].plot(gs, [B[k]['z_step']['late'][0] for k in ks], ls, marker='o', ms=3, color=C[c])
    for a, t in zip(ax, ('late self-model prediction error', 'late workspace change per step')):
        a.set_xscale('log', base=2);a.set_xlabel('broadcast gain g (dotted: pre-declared, solid: strong)');a.set_title(t, fontsize=9)
    ax[0].legend(frameon=False, fontsize=7, ncol=3);save(fig, f'e2_selfmodel_strong_{model}_{tag}.png')


def safe(fn, *a):
    try:fn(*a)
    except Exception as e:print('figure failed', fn.__name__, a, repr(e)[:200])


if __name__ == '__main__':
    for m, t in (('qwen3-8b', 'pilot2'), ('qwen3-8b', 'main'), ('qwen3-14b', 'rep')):safe(fig_e1, m, t)
    for m, t in (('qwen3-8b', 'main'), ('qwen3-14b', 'rep')):safe(fig_e2, m, t);safe(fig_traj, m, t)
    safe(fig_huginn, 'main');safe(fig_ouro, 'main')
    for m, t in (('qwen3-8b', 'main'),):safe(fig_x, m, t);safe(fig_bcs, m, t)
