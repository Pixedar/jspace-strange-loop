# Phase B: a trained self-describing loop around the model's own thinking

Qwen3-8B (main) and Qwen3-14B (replication). The model's own thinking traces are cut into 8 segments of 64 tokens; a
segment never sees earlier tokens; the only link is a 32-number state read from the model at the end of the previous
segment and written into the next one. A 2-position channel carries a constant, the previous state, or the self-model's
forecast of the state the segment will end in. The self-model learns only from its own prediction loss; λ is the
pressure on the system to be predictable to it. The language model stays frozen; reader, writers and self-model are
trained on next-token loss (+ λ × self-prediction).

Results and their interpretation: [../REPORT.md](../REPORT.md) §3.6 and §5. Every arm in one table:
[results/phaseB_arms_summary.md](results/phaseB_arms_summary.md) (also `.csv`). Design and the predictions written
before any result: [PLAN.md](PLAN.md).

## Files

* `code/loopb.py` — the whole experiment: `gen` (the model's own thinking traces, readers, reference NLLs), `train` (one
  arm), `eval` (teacher-forced interventions, self-prediction baselines, word readouts, free generation), `gate`,
  `words` (reader axes and the clock plane in J-lens words)
* `code/loopb_analyze.py` — probes, geometry (rotation fits, clock coherence), collapse-proof predictability
* `code/summarize_arms.py` — the summary table with the strict rotation null (scrambled segment order) and the
  held-out single-trace test
* `code/figs_loopb.py` — figures
* `results/loopb/` (8B), `results/loopb14/` (14B): `eval/<arm>.json` (every number of an arm), `ckpt/<arm>/` (config and
  training curves), `reader_words.json`, `j_axes.json`, `prompts.json`. The arrays (`eval/<arm>.npz`, traces `data.npz`,
  trained weights `loop.pt`) are in the Hugging Face dataset under `phaseB/`.

## Arm names

`<channel>_<reader>_l<λ>[_c|_cx][_s<seed>]`

* channel: `const` (no information), `state` (the previous state again: matched memory control), `fcst` (the
  self-model's forecast: the strange loop), `cut` (the state is never passed on: the floor)
* reader: `learned` (linear, initialised at the principal axes of natural states) or `J` (fixed: the top-32 principal
  axes of the J-lens word projections)
* λ: pressure on the system to be predictable to its own self-model
* loss: no suffix = cosine on the sphere (degenerate: collapse), `_n` = variance-normalised (degenerate), `_c` =
  contrastive with the system's own other moments among the candidates, `_cx` = contrastive with other traces at the
  same moment only

## Run

```bash
pip install torch transformers==4.57.1 accelerate scikit-learn wordfreq huggingface_hub
cd code
python loopb.py gen   --out out/loopb
python loopb.py train --out out/loopb --chan const --lam 0 --steps 400
python loopb.py eval  --out out/loopb --tag const_learned_l0
python loopb.py train --out out/loopb --chan fcst --lam 1 --ploss nce --steps 400
python loopb.py eval  --out out/loopb --tag fcst_learned_l1_c
python loopb.py words --out out/loopb
python summarize_arms.py out/loopb --out out
```

8B: one 48 GB GPU, ~14 min per arm (400 steps, batch 10) and ~5 min per evaluation; trace generation ~26 min.
14B: one 80 GB GPU, ~20 min per arm.
