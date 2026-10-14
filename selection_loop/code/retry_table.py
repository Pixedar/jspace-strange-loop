"""Tables for the retry experiment (evalr_<tag>.json from focus_retry.py eval).
    python retry_table.py <run folder> [--out retry_summary.md]"""
import argparse, json, pathlib, sys

CATS = [('present_first_failed', 'first read failed'), ('present_first_delivered', 'first read delivered'), ('absent', 'tracked animal absent')]
EDITS = [('none', 'no edit'), ('toward_failed', 'toward "failed"'), ('toward_failed_x0.5', 'toward "failed" × 0.5'), ('toward_failed_x2', 'toward "failed" × 2'),
         ('toward_failed_query_only', 'toward "failed", second query only'), ('toward_failed_update_only', 'toward "failed", second update only'),
         ('random_same_size', 'random, same size (mean of 3)'), ('toward_delivered', 'toward "delivered"'),
         ('toward_delivered_query_only', 'toward "delivered", second query only'), ('toward_delivered_update_only', 'toward "delivered", second update only')]


def f(x, nd=3):
    return '–' if x is None else (f'{x:.{nd}f}' if isinstance(x, float) else str(x))


def main():
    p = argparse.ArgumentParser();p.add_argument('dir');p.add_argument('--out', default=None);a = p.parse_args()
    sys.stdout.reconfigure(encoding='utf-8');d = pathlib.Path(a.dir)
    R = {x.stem[6:]:json.load(open(x, encoding='utf-8')) for x in sorted(d.glob('evalr_*.json'))}
    L = [f'# Retry experiment: {d}', '', '| arm | overall | tracked updated | colour kept | reads per chunk | first read on tracked (present) | first read "nothing new" (absent) |',
         '|---|---|---|---|---|---|---|']
    for t, r in R.items():
        L.append(f"| {t} | {f(r['acc'])} | {f(r['acc_updated'])} | {f(r['acc_kept'])} | {f(r['reads_per_chunk'], 2)} | "
                 f"{f(r['first_read_on_tracked_when_present'])} | {f(r['first_read_nothing_new_when_absent'])} |")
    L += ['', 'Second read, by what happened to the first read (attention on the tracked record / on "nothing new"; accuracy at that question):', '',
          '| arm | ' + ' | '.join(n for _, n in CATS) + ' |', '|---|' + '---|'*len(CATS)]
    for t, r in R.items():
        b = r['by_first_read'];cells = []
        for c, _ in CATS:
            x = b.get(c);cells.append(f"{f(x['second_read_on_tracked'])} / {f(x['second_read_nothing_new'])}; acc {f(x['acc'])} (n {x['n']})" if x else '–')
        L.append(f'| {t} | ' + ' | '.join(cells) + ' |')
    L += ['', 'Knowing that it does not know (same tracked animal as before; the stored colour is then the old one, always wrong if the update was missed):', '',
          '| arm | update missed: acc / max prob / prob on previous true colour | delivered by first read | delivered by second read |', '|---|---|---|---|']
    for t, r in R.items():
        c = r['confidence'];g = lambda x:(f"{f(x['acc'])} / {f(x['max_prob'])} / {f(x.get('prob_previous_true_colour', x.get('prob_stale_colour')))} (n {x['n']})" if x else '–')
        L.append(f"| {t} | {g(c.get('update_not_acquired'))} | {g(c.get('acquired_by_first_read'))} | {g(c.get('acquired_by_second_read'))} |")
    L += ['', '| arm | intermediate state: "first read failed" (balanced acc., grouped by stream) | "tried and failed" vs "chose nothing" |', '|---|---|---|']
    for t, r in R.items():
        dm = r.get('decode_mid_state') or {}
        L.append(f"| {t} | {f(dm.get('first_read_failed'))} | {f(dm.get('tried_and_failed_vs_chose_nothing'))} |")
    L += ['', 'Paired counterfactual (same stream, history, first attention and noise; only the first delivery at chunk t differs; pooled over t = 2-4):', '',
          '| arm | delivered | failed | delivered, second query from the failed state | failed, second query from the delivered state |', '|---|---|---|---|---|']
    for t, r in R.items():
        cf = r.get('counterfactual_first_delivery')
        if not cf:continue
        g = lambda x:(f"second read on tracked {f(x['second_read_on_tracked'])}; acc {f(x['acc'])} (n {x['n']})" if x else '–')
        L.append(f"| {t} | {g(cf.get('delivered'))} | {g(cf.get('failed'))} | {g(cf.get('delivered_query_from_failed_state'))} | {g(cf.get('failed_query_from_delivered_state'))} |")
    L += ['', 'Other failure rates at test time (trained at 0.3):', '', '| arm | rate | acc | reads per chunk | second read on tracked after failed / after delivered |', '|---|---|---|---|---|']
    for t, r in R.items():
        for pt, x in (r.get('test_failure_rate') or {}).items():
            L.append(f"| {t} | {pt} | {f(x['acc'])} | {f(x['reads_per_chunk'], 2)} | {f(x['second_read_on_tracked_after_failed'])} / {f(x['second_read_on_tracked_after_delivered'])} |")
    for t, r in R.items():
        E = r.get('mid_state_edits')
        if not E:continue
        L += ['', f"### Edits of the intermediate state: {t} (direction norm {f(r.get('edit_direction_norm'))}; calibration {r.get('calibration_counts')}; one chunk at a time, chunks 2-4, mean of the per-chunk means)", '',
              '| edit | moved (1-cos) | first read failed: second read on tracked / acc | first read delivered: second read on tracked / acc | absent: second read on tracked / "nothing new" |',
              '|---|---|---|---|---|']
        for c, name in EDITS:
            e = E.get(c)
            if not e:continue
            pf, pd, ab = e.get('present_first_failed'), e.get('present_first_delivered'), e.get('absent')
            L.append(f"| {name} | {f(e.get('state_moved_1_minus_cos'))} | {f(pf['second_read_on_tracked']) + ' / ' + f(pf['acc']) if pf else '–'} | "
                     f"{f(pd['second_read_on_tracked']) + ' / ' + f(pd['acc']) if pd else '–'} | "
                     f"{f(ab['second_read_on_tracked']) + ' / ' + f(ab['second_read_nothing_new']) if ab else '–'} |")
    txt = '\n'.join(L);print(txt)
    if a.out:pathlib.Path(a.out).write_text(txt+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
