# The trained self-describing loop

Qwen3-8B (main) and Qwen3-14B (replication). The model's own thinking is cut into 8 chunks of 64 tokens; a chunk never
sees earlier chunks; the only memory is a 32-number state read from the model at the end of a chunk and written into
the next one. Two extra positions (the channel) carry a constant, the previous state, or the self-model's forecast of
the next state. The self-model learns only from its own prediction error; the pressure λ trains the reader and writers
to make the states predictable to it. The language model stays frozen.

Results and their interpretation: [archived detailed report](../history/REPORT_20261014.md), section 4. Every variant in one table:
[results/arms_summary.md](results/arms_summary.md) (also `.csv`). Design and the predictions written before the
results: [PLAN.md](PLAN.md).

## Files

* `code/loopb.py` — the whole experiment: `gen` (the model's thinking, natural statistics, readers, reference values),
  `train` (one variant), `eval` (teacher-forced interventions, self-prediction baselines, word readouts, free
  generation), `gate`, `words` (the state's axes and the plane of its rotation in J-space words)
* `code/loopb_analyze.py` — probes, geometry, self-predictability measures that cannot be met by collapse
* `code/summarize_arms.py` — the summary table, with the rotation fit against scrambled chunk orders and the held-out
  single-thought test
* `code/free_text_checks.py` — word-pattern counts in the free-generated thinking (invented past, restarts, identity
  words, AI self-reference) against the model's real thinking
* `code/figs_loopb.py` — figures
* `results/loopb/` (8B), `results/loopb14/` (14B): `eval/<variant>.json` (every number of a variant),
  `ckpt/<variant>/` (configuration and training curves), `reader_words.json`, `j_axes.json`, `prompts.json`
* `results/rotation_plane_positions.json` — the chunk averages in the plane of the rotation with their J-space words
  (figure `stages_of_a_thought.png`); `results/free_text_checks.json` — output of `free_text_checks.py`

The arrays (`eval/<variant>.npz`, the thinking `data.npz`, trained weights `loop.pt`) are in the Hugging Face dataset
under `phaseB/`.

## Variant names

`<channel>_<reader>_l<λ>[_c|_cx|_n][_s<seed>]`

* channel: `const` (plain loop: the channel carries a learned constant), `state` (the previous state again: matched
  memory control), `fcst` (the self-model's forecast: the strange loop), `cut` (no state passed on: the floor)
* reader: `learned` (linear, initialised at the main directions of natural states) or `J` (fixed: the 32 main
  directions of the J-space word readings)
* λ: pressure on the system to be predictable to its own self-model
* loss: no suffix = cosine (degenerate: collapse), `_n` = variance-normalised (degenerate), `_c` = contrastive with the
  thought's own other moments among the candidates, `_cx` = contrastive with other thoughts at the same moment only
* `_s1`: second seed

In PLAN.md this system is called "Phase B"; "G2" there refers to the private-state test of the frozen loop.

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
python free_text_checks.py out/loopb --out out/free_text_checks.json
```

8B: one 48 GB GPU, about 14 minutes per variant (400 steps of 10 thoughts) and 5 minutes per evaluation; generating the
thinking about 26 minutes. 14B: one 80 GB GPU, about 20 minutes per variant.
