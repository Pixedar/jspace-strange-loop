"""Table of the edit controls (edits_<tag>.json): per condition, answers and attention at the edited chunk and agreement
at the next chunk, averaged over chunks 2-4 (and per chunk on request).
    python edits_table.py <edits_*.json> [...]"""
import json, sys
import numpy as np

COND = [('own', 'none'), ('focus_edit_x1', 'focus code A→B'), ('focus_edit_query_only', 'same edit, query only'),
        ('focus_edit_answer_only', 'same edit, answer prompt only'), ('focus_edit_x0.5', 'focus edit × 0.5'),
        ('focus_edit_x2', 'focus edit × 2'), ('focus_edit_x-1', 'focus edit reversed'), ('random_edit_same_size', 'random edit, same size'),
        ('focus_edit_toward_present_C', 'focus code A→C (C in the chunk)'), ('focus_edit_toward_absent_D', 'focus code A→D (D not in the chunk)')]
KEYS = [('answers_A', "A's new colour"), ('answers_B', "B's new colour"), ('answers_C', "C's colour"), ('answers_A_stored', "A's stored colour"),
        ('att_A', 'attention A'), ('att_B', 'attention B'), ('att_C', 'attention C'), ('att_nothing_new', 'attention nothing new'),
        ('next_answers_A', 'next chunk: A'), ('next_answers_B', 'next chunk: B'), ('displacement_1_minus_cos', 'state moved (1-cos)')]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    for f in sys.argv[1:]:
        r = json.load(open(f, encoding='utf-8'));bc = r['by_chunk']
        print(f"\n### {r['tag']} ({f})\n\nAveraged over the edit at chunks 2, 3, 4 (256 twin streams each).\n")
        print('| condition | ' + ' | '.join(k[1] for k in KEYS) + ' |');print('|---|' + '---|'*len(KEYS))
        for c, name in COND:
            if c not in bc['2']:continue
            cells = []
            for k, _ in KEYS:
                v = [bc[t][c].get(k) for t in bc if bc[t][c].get(k) is not None]
                cells.append(f'{np.mean(v):.3f}' if v else '–')
            print(f'| {name} | ' + ' | '.join(cells) + ' |')
        ce = r.get('colour_edit', {})
        print('\nColour edit (tracked animal not updated at the edited chunk): ' + ', '.join(f'{k} {v:.3f}' for k, v in ce.items() if isinstance(v, float)))


if __name__ == '__main__':
    main()
