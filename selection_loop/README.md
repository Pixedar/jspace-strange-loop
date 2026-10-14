# A state that decides what the model reads

Code and results for section 5 of the [report](../REPORT.md): a compact learned state (32 numbers) that lives in a
frozen language model's processing, carried from one piece of text to the next, trained only to give the right answer.
The question is whether it comes to represent which distinction matters and to steer what the model takes in through
that representation.

![the system](../figures/selection_loop.png)

## The task
Ten animals, ten colours; a stream has six chunks of four updates ("The fox is now red."). Chunk 0 says which animal to
keep track of (in 30 % of streams a later chunk switches it); after every chunk the model is asked, with only the
loop's state in view, "what color is the animal you are keeping track of now?"

## Code (`code/`)

| file | what it does |
|---|---|
| `focus_sel.py` | restricted access: each update read on its own, one attention read per chunk whose query comes from the state; stages `prep` (encode the updates once), `train`, `eval` (accuracy, query swaps, replay, edits), `edits` (the edit controls of section 5.3) |
| `focus_retry.py` | reads that sometimes fail, a second optional read decided after or before the first (section 5.4); paired counterfactual, edits, other failure rates at test time |
| `focus_open.py` | open access: the model reads the whole chunk as text, the state is written into it and read back from the last word (section 5.5); selectivity, edits, and the model's own attention |
| `focus_loop.py` | the frozen model with write and read hooks, and the stream generator used by `focus_open.py` |
| `analyze_v7.py`, `edits_table.py` | tables for restricted access: accuracy, paired intervals per stream, decoding (whole streams held out), edit controls |
| `retry_table.py`, `analyze_open.py` | tables for sections 5.4 and 5.5 |
| `plot_v7.py`, `plot_open.py`, `make_selection_figure.py` | the figures |

Designs written down before their results: [`DESIGN_retry.md`](DESIGN_retry.md), [`DESIGN_open.md`](DESIGN_open.md).

## Reproduce

```bash
pip install torch transformers accelerate scikit-learn matplotlib
cd selection_loop/code
# restricted access, Qwen2.5-0.5B-Instruct (about 30 minutes per run on a 4 GB laptop GPU)
python focus_sel.py prep  --out out/sel05 --model Qwen/Qwen2.5-0.5B-Instruct
python focus_sel.py train --out out/sel05 --query state --steps 1200 --bs 16 --seed 0
python focus_sel.py train --out out/sel05 --query const --steps 1200 --bs 16 --seed 0
python focus_sel.py eval  --out out/sel05 --tag sel-state_l0
python focus_sel.py edits --out out/sel05 --tag sel-state_l0
python analyze_v7.py --dir out/sel05
# reads that can fail: start from the trained loop
python focus_retry.py train --out out/sel05 --arm retry --steps 600 --bs 16 --cost_start 100 --cost_ramp 100 --init out/sel05/ckpt/sel-state_l0.pt
python focus_retry.py eval  --out out/sel05 --tag retry_c0.1_ws
# open access
python focus_open.py train --out out/open05 --model Qwen/Qwen2.5-0.5B-Instruct --lw 8 --lr 16 --steps 1500 --bs 16
python focus_open.py eval  --out out/open05 --tag open_z32
python analyze_open.py out/open05
```

Arguments used for the other models: Qwen3-1.7B restricted access `--lw 10 --lr 18 --norm 1` (its activations need
per-dimension standardisation), Qwen2.5-3B-Instruct open access `--lw 12 --lr 24`.

## Results (`results/`)
* `restricted_0.5B.md` — every Qwen2.5-0.5B arm: accuracy, paired differences per stream, query swaps, decoding;
* `restricted_1.7B.md` — Qwen3-1.7B;
* `edit_controls.md` — the edit controls for every run (focus and colour edits, read-only and answer-only edits, dose,
  random directions, present and absent animals);
* `own_read.md` — reads that can fail (section 5.4);
* `open_access.md` — open access at 0.5B and 3B (section 5.5);
* `eval/` — the evaluation summaries (JSON) behind these tables. Arrays and trained weights are in the dataset
  [Pixedar/jspace-strange-loop](https://huggingface.co/datasets/Pixedar/jspace-strange-loop) under `selection_loop/`.
