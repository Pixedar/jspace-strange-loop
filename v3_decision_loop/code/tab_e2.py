"""Quick tables from a pulled results_e2.json: regime-A style stages (A1, A2, X) and self-model stages (BC, BCs).
    python tab_e2.py results/e2_qwen3-8b_main_results_e2.json [X|BCs|...]"""
import sys, json

d = json.load(open(sys.argv[1]));want = sys.argv[2:] or list(d)


def regime(st):
    for k, v in sorted(d[st].items()):
        cl = v['classes'];tot = sum(cl.values())
        s = (f"{st} {k:18s} n{v['n']:5d} fixed {cl.get('fixed', 0)/tot:.2f} turn {v['late_turnover']:.3f} "
             f"zstep {v['late_z_step']:.3f} ov64 {v['overlap_with_natural'].get('64', float('nan')):.2f} contr {v['context_contraction']:.2f}")
        if 'memory_gap' in v:
            mg = v['memory_gap']
            s += ' | '+' '.join(f"{f}:"+'/'.join(f"{mg[f][t][0]:.2f}" for t in ('1', '4', '16', '64') if t in mg[f]) for f in mg)
        if 'evidence_decodability' in v:s += ' | dec '+'/'.join(f"{x:.2f}" for x in v['evidence_decodability'].values())
        if 'accuracy' in v:
            ac = v['accuracy'];s += ' | acc '+' '.join(f"{f}:{ac[f]['64'][0]:.2f}" for f in ac if '64' in ac[f])
        print(s)
        if k.startswith('std') and ('late_write_words' in v or k.split('|')[1] in ('inc', 'out', 'lens', 'J', 'nofb')):
            print('     late words :', ', '.join(w for w, _ in v.get('late_top_words', [])[:12]))
            if v.get('late_write_words'):print('     write words:', ', '.join(w for w, _ in v['late_write_words'][:12]))


def selfmodel(st):
    B = d[st]
    for k, v in sorted(B.items()):
        if k.startswith('contrasts'):continue
        cl = v['classes'];tot = sum(cl.values());ae = v.get('action_entropy_late', [float('nan')])[0]
        s = (f"{st} {k:12s} n{v['n']:5d} fixed {cl.get('fixed', 0)/tot:.3f} ent {ae:.2f} perr {v['perr']['late'][0]:.3f} "
             f"zstep {v['z_step']['late'][0]:.3f}")
        if 'memory_gap' in v:
            mg = v['memory_gap'];s += ' | '+' '.join(f"{f}:"+'/'.join(f"{mg[f][t][0]:.2f}" for t in ('1', '4', '16', '64') if t in mg[f]) for f in mg)
        print(s)
    for k, v in B.get('contrasts_vs_random_actions', {}).items():
        print(f"  {k:28s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] n={v[3]}")


for st in want:
    if st not in d:continue
    if st in ('BC', 'BCs', 'BC1'):selfmodel(st)
    elif st in ('A1', 'A2', 'X'):regime(st)
