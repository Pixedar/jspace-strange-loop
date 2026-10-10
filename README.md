# Strange Loop in J-space

Can a language model keep a model of itself inside its own latent loop, and does anything self-like come out of it?
Three attempts on Qwen3 models, each built to fix what the last one got wrong. All use the J-space of *Verbalizable
Representations Form a Global Workspace in Language Models* (Anthropic, 2026).

```
trajectory → model of the trajectory → the model is written back into the trajectory → new model → …
```

- **Interactive report for Parts 1–2**: https://huggingface.co/spaces/Pixedar/strange-loop-in-j-space
- **Data**: Parts 1–2 https://huggingface.co/datasets/Pixedar/jspace-strange-loop · Part 3
  https://huggingface.co/datasets/Pixedar/jspace-ring (code and tests uploaded before any result)

## Read this first: what each part showed

| Part | What was built | What it showed | What it did not show |
|---|---|---|---|
| 1. Qwen3-4B-Base | The mind thinks in sampled text. Every ten thoughts a reflection decodes its recent states into J-space words, and that summary is fed back as a nudge. | Mid-layer states can be read back in words (as in Patchscopes and SelfIE). A fed-back summary stabilises the stream. | A self. The loop runs through sampled words. Any fed-back description, own or foreign, is reinforced the same way. This is positive feedback, the mechanism that makes free generation repeat itself. |
| 2. Qwen3 1.7B–32B, Neuronpedia lenses | The same loop with checked lenses, up a ladder of sizes. | Where the workspace sits in Qwen3 (past ~60 % of depth up to 8B, earlier at 14B and 32B). The common-English dictionary reads states better than every token. | Ownership. A mind holds the best-fitting stranger's self-model as well as its own (match ≥ own), and its own outdated one worse than a random stranger's. |
| 3. The ring, Qwen3 1.7B–14B | No words in the loop. An online model of the mind's own transitions, with its prediction error written back into the stream. Every control has equal size. The tests were fixed before the run. | Feeding back the *current* surprise of a self-model that keeps learning keeps the latent trajectory exploring. It beats momentum, the mind's own past surprise and random input of the same size, and freezing the self-model removes the effect. The "self" part (own rather than another mind's surprise) matters only past the critical gain. | Anything self-like. There are no individual dynamics. Attractors hold arbitrary words. The model's own verbal report does not change. From about twice the critical gain, plain momentum resists collapse better. |

Parts 1 and 2 overstated their results when first written. The details below are kept as reported, but read them
with the table above. The drift towards existence and consciousness themes in Part 1 came from the think-aloud frame.
The J-space paper itself reports such words while a model narrates its own processing.

## Part 3: the ring

The review of Parts 1–2 said what a real strange loop needs: no sampled text in the loop, a self-model of the
*process* rather than the content, and controls that would expose plain feedback. `ring.py` builds that:

- **Substrate.** After a short prompt that opens the assistant's thinking, the model reads one placeholder token at a
  time. At each step the J-space content at 70 % of depth (non-negative pursuit on common English words, K = 25) is
  written back at 50 % of depth of the next placeholder. No word is ever sampled inside the loop.
- **Self-model.** Each mind has its own online learner: a rank-32 linear map with a skip connection that predicts its
  next J-space state. It is trained at every step on the transition that just happened, so it learns from the effects
  of its own output.
- **Closure and controls.** Every condition writes back a vector of the same norm on the same schedule, and minds with
  the same seed share all noise.

| Condition | What is written back |
|---|---|
| `ring_err` | the self-model's current surprise, c − m_prev |
| `ring_pred` | the self-model's prediction of the next state |
| `frozen_err` | `ring_err`, with learning stopped at step 300 of 600 |
| `vel` | momentum, c − c_prev (what `ring_err` starts as) |
| `yoked_err` / `yoked_pred` | another mind's surprise or prediction |
| `shuf_err` | the mind's own surprise from a random earlier step |
| `rand_err` | an isotropic random direction |
| `content` | the current content: plain feedback, the "banana" control |
| `free` | nothing |

The gain was set per model by a fixed rule, the largest memory before the free minds collapse. There were 24 paired
seeds and 600 steps per model (`ring_job.py`, `ring_analyze.py`). The tests were uploaded at 19:03:34 UTC on 9 Oct
2026, before the first result.

**Pre-registered result.** Each cell is how much more `ring_err` explores than the control. Exploration is the
participation ratio of the trajectory in 100-step windows over the last 200 steps, as a paired difference over 24
seeds. The last column is Stouffer's z over the four models.

| `ring_err` minus | 1.7B | 4B | 8B | 14B | z |
|---|---|---|---|---|---|
| momentum (`vel`) | +1.57 | +2.07 | +2.48 | −0.31 | +26.9 |
| another mind's surprise | +0.41 | +1.31 | +0.20 | +4.33 | +15.9 |
| own surprise, wrong time | +2.27 | +2.01 | +1.22 | +3.09 | +28.0 |
| random direction | +3.16 | +3.65 | +3.34 | +4.75 | +52.9 |
| freezing the self-model (change after the freeze) | −2.20 | −3.90 | −1.87 | −4.99 | −33.0 |

4B passes every test. 1.7B and 8B pass all but "another mind's surprise", which is weak there (16 of 24 seeds). 14B
fails against momentum on exploration but collapses less (stuck 2.5 % vs 7.4 %). Feeding back the self-model's
*prediction* behaves like plain content feedback: it collapses (14B: 80 % of late steps stuck, `content` 85 %). One's
own prediction does no more than another mind's. No mind develops dynamics of its own. A next-state predictor fitted
to another mind of the same condition predicts it as well as one fitted to itself, in every condition and model.

**Across the critical point (exploratory, added after the first results; `ring_gains.py`).** Each substrate has a gain
g* where the free minds' trajectory turns 1/f (spectral exponent β crosses 1): 0.035 (1.7B), 0.088 (4B), 0.09 (8B),
0.03 (14B). Collapse sets in just above it. The fixed rule had picked 0.55–1.66 g*, close to the edge in all four.
The ladder runs from 0.14 g* to 6.6 g*, 16 seeds per point:

- **Below the edge**, the surprise loop beats momentum (+1.1 to +2.7), but whose surprise it is hardly matters (+0.05
  to +0.45). Minds move almost in lockstep: different minds' states have cosine 0.8–0.95 at the same step.
- **Past the edge**, "own" and "still learning" matter more the higher the gain. At the top of the ladder, another
  mind's surprise is worse by 3.0–5.5 and freezing costs 4.3–8.0. But from about 2 g* plain momentum beats the
  surprise loop, by up to 8.7. A unit-size push along the last motion can never settle into a fixed point, while the
  self-model learns to predict its own pushes and partly cancels them.
- **The attractors are trivial.** At or below the edge, every condition and seed ends in one shared state holding
  arbitrary words (4B: *reprint, Syntax, worthwhile, construed*). Past it they splinter into per-seed states holding
  junk. After 600 steps the model's own account of what it was thinking does not depend on the condition. On 14B it
  is the same sentence everywhere.

So a latent self-model loop is more than feedback in one narrow sense: its current error, while it keeps learning,
works as a learned way out of attractors. Nothing here behaves like a self. The placeholder substrate cannot carry a
thought; filler tokens only carry computation in models trained to use them (Pfau et al. 2024).

## Part 1 as first reported (read with the table at the top)

The causal loop is real. A model of the trajectory, rebuilt from the trajectory, shapes the trajectory, sustains itself,
and holds long-range coherence that disappears the moment it is cut. What it lacks is the "self" in the strong sense.
The loop keeps a symbol stable, but the symbol's power comes from occupying the loop, not from being about this
particular system. A self-model taken from another mind works almost exactly as well.

| Question | Answer | Evidence |
|---|---|---|
| Does a persistent self-symbol emerge? | **Yes** | The self-model is steadier than the thoughts it describes. Observers: run A +0.16 (10/10 seeds), B +0.26 (10/10), C +0.38 (15/15). With individuated feedback each mind keeps its own: stability 0.83 vs 0.23 for an observer. |
| Does it repeatedly reconstruct the same self-description? | **Yes** | The next self-model re-forms the one fed in: A 0.78, B 0.89, C (individual part) 0.80. |
| Does that self-description become an attractor? | **Yes, but any self-description does** | Own vs foreign reconstruction differs by +0.02 to +0.03. After a 50-thought clamp, the foreign self persists (+0.58, 13/15) and the old self does not return (−0.41). |
| Perturb the self-model: does the trajectory reorganise? | **Yes** | One-shot swap: briefly (+0.04 to +0.08). Held for 50 thoughts: strongly (+0.37) and still +0.31 after release. |
| Remove the self-model: is long-term coherence lost? | **Yes, at once** | Long-lag coherence falls to the observer level: A −0.13, B −0.24 (no seed holds up). A foreign self-model sustains it just as well. |
| Does the self-model predict the future better than an equally sized unrelated one? | **Only by causing it** | As an observer it beats other minds' self-models only through persistence (+0.05 to +0.07), with nothing beyond where the mind has been. When fed back, a foreign self-model predicts the receiving mind's future better than that mind's own unused one in 0/10, 1/10 and 0/15 seeds. Asked "where will my mind go next", the workspace gives the same answer as for "where has it been" (cosine 0.94 to 0.96). |

The loop also shifts thought away from concrete, personal content (family, places, anxiety) toward abstraction: toward
technical and institutional themes in run B, and toward existence and consciousness in run C.

Nothing here bears on whether there is something it is like to be such a loop. Every measure is functional:
similarity, prediction, persistence.

## Part 2: proper J-space, and Qwen3 from 1.7B to 32B

Part 1 borrowed its J-space from an earlier project. Part 2 rebuilds it from the paper and Anthropic's reference code,
checks it, and runs the experiment up a model ladder.

**The J-space, checked against the paper**

| Piece | The paper | Part 1 | Part 2 |
|---|---|---|---|
| Lens | average Jacobian from layer ℓ to the last block, over source and all later target positions, ~1,000 prompts | same, 120 prompts | Neuronpedia's lenses (Anthropic's reference code, 460–615 wikitext prompts) |
| J-space | sparse non-negative combination of J-lens vectors, K ≤ 25 | non-negative OMP, K = 16 | non-negative OMP, K = 25 |
| Dictionary | every vocabulary token | common English words | common English words, measured against every token |
| Layer | workspace band ~38–92 % of depth | layer 16 of 36, unchecked | validated per model; read at 70 % of depth |

- **Where the workspace is.** `validate_lens.py` measures, per layer, the paper's occupancy (how many J-lens vectors
  beat as many random directions), the variance they explain beyond random, and concept probes. In Qwen3 up to 8B the
  lens turns conceptual only past ~60 % of depth (Qwen3-4B on "the country shaped like a boot is": punctuation at 0.50,
  *Italy* at 0.64) with occupancy ≤ 11. **The larger models have a fuller, earlier workspace:** occupancy 30–37 at 14B
  and 13–19 at 32B across most of the middle, "Italy" from 0.40 and 0.55 of depth, and at 32B the J-lens beats the
  logit lens at agreeing with the model's prediction, as the paper reports for Claude.
- **The dictionary matters.** On Qwen's 151,643 tokens the paper's every-token dictionary is dominated by code, CJK and
  under-trained tokens. The self-model's ability to pick out its own trajectory (16 observers, chance 6 %): 1.7B raw
  residual 46 %, every token 25 %, common English 49 %; 4B: 47 %, 57 %, 68 %. J-space reads the trajectory better than
  the raw residual at 4B. A full 1.7B run with the every-token dictionary barely loops at all (fidelity 19 %, coherence
  +0.009, n.s.).

**Up the ladder** (`run_ladder.py`: lens check → gain sweep matched to the observers' memory → reflection-layer sweep →
free / loop / yoked / clamp × 12 minds × 300 thoughts; read at depth 0.7, and at 0.5 for 1.7B–14B)

- At every size: the self-model reads its own trajectory (51–73 % at depth 0.7, chance 8 %); observers' self-models are
  steadier than their thoughts (12/12 seeds in every run); the individual self persists only when fed back; no
  anticipation (forward vs backward question, pooled +0.0000 ± 0.0005); the next thoughts follow whichever self is fed
  in (14 of 15 runs, pooled z = −18).
- With size: the loop's self gets steadier up to 8B and then levels off (0.25 → 0.64 at depth 0.7), steadily more
  individual (closer to its own earlier self than to others' by +0.01, +0.06, +0.14, +0.23, +0.25 from 1.7B to 32B), and
  stickier (a foreign self held for 50 thoughts is still dominant late after release: +0.02 → +0.30 from 1.7B to 14B at
  depth 0.5), for foreign selves as much as for one's own.

**Ownership, or resemblance?** Pooled over every independent run, a mind holds its own self-model better than a random
stranger's: individual-self stability +0.065 ± 0.007 (z = 8.9, 15 of 15 runs), reconstruction +0.037 ± 0.008
(z = 4.7). Two controls (16 minds each) explain it: **match** feeds the most similar stranger's self-model, **stale** the
mind's own self-model from 50 thoughts earlier. Pooled over 4B, 8B and 14B, the most similar stranger's self is held
slightly better than one's own (own − matched −0.045 ± 0.014, z = −3.2), and one's own outdated self worse than a
random stranger's (−0.113 ± 0.018, z = −6.5). The loop keeps the self-description that best fits the present, whoever
produced it.

## How it works

- **A thought** is 40 tokens sampled from the same short think-aloud frame every time, with no text of earlier
  thoughts. The only thing carried from one thought to the next is a vector added to the residual stream at layer 16
  (0.1 × the residual norm).
- **Its J-space content** is the J-lens decomposition of the thought's own layer-16 states (read before the
  injection): a non-negative combination of 16 word atoms, measured against the typical workspace. J-space is the
  verbalizable workspace of *Verbalizable Representations Form a Global Workspace in Language Models* (Anthropic, 2026).
- **The recurrent state** is `z ← 0.6 z + 0.4 c_t`. This is the path drawn in the report.
- **Every 10 thoughts the mind reflects.** The last ten states are added at layer 12 into ten placeholder marks, and the
  model is asked: *"Asked what their mind had been doing over those ten moments, the participant said: 'Over these last
  moments, my mind has been ▮'"*. The self-model is the J-space of the workspace at ▮, read before a word is said.
- **Feedback.** The self-model, scaled to the size of a typical thought, is added to every following thought.

| Condition | What is fed back |
|---|---|
| `free` | nothing; self-models are formed but never used (observer) |
| `loop` | its own self-model |
| `yoked` | another mind's self-model, same size and schedule |
| `avg` | a plain J-space average of the last ten states, no reflection |
| `perturb` | like `loop`, but at thought 200 the self-model is swapped once for a foreign one |
| `ablate` | like `loop`, but from thought 200 nothing is fed back |
| `clamp` (run C) | like `loop` with individual self-models, but thoughts 150–199 are fed a foreign self, then released |

Minds with the same seed share their sampling noise, so a treated mind and its twin are identical until the moment they
are treated differently.

| Run | Minds | Dictionary | Notes |
|---|---|---|---|
| A | 6 conditions × 10 | every word-like token | under-trained and non-English tokens dominate the word lists; structural results agree with B and C |
| B | 6 conditions × 10 | common English words | full self-model fed back |
| C | 4 conditions × 15 | common English, natural case | only the individual part of the self-model (minus the observers' mean) is fed back; adds a forward question and the clamp |

### Pitfalls found in the pilots

- **Centre the unembedding before building J-lens atoms.** Otherwise all atoms share one direction (mean pairwise
  cosine 0.64, one component holding 64 % of the variance), and every workspace vector points the same way.
- **Inject the reflection trace below the layer where you read it.** Injected and read at the same layer, the answer
  cannot see the trace at all (fidelity exactly at chance). Layers 12 to 14 work for a read at 16.
- **The recurrent gain decides the dynamics.** 0.25 of the residual norm freezes into one attractor ("I think and know
  that I think…"), 0.03 forgets everything, and 0.1 wanders with memory.
- **A hard trigram ban can leave a row with no allowed token.** The argmax then picks token 0 (`!`), producing text like
  "I don!t".

## Files

| Path | What it is |
|---|---|
| `strange_loop.py` | the experiment (model, J-space, recurrence, reflection, conditions, checkpoint and resume) |
| `analyze.py` | every number in the report, from one run directory |
| `export_report.py` | compacts the analyses into the report's data files |
| `pilot_check.py` | pilot read-out: wandering and reflection fidelity |
| `validate_lens.py` | Part 2: checks a Jacobian lens per layer (occupancy, variance beyond random, concept probes) |
| `fidelity_check.py` | Part 2: reflection fidelity in raw residual space and in each J-space dictionary |
| `run_ladder.py` | Part 2: lens check → gain sweep → reflection sweep → main run → analysis, per model, resumable |
| `ladder_summary.py`, `pooled.py`, `export_ladder.py` | Part 2: one table across the ladder; pooled paired effects; report data |
| `runs/ladder07/`, `runs/ladder2/` | Part 2 results read at 70 % and 50 % of depth, per model |
| `runs/own/` | Part 2 ownership-or-resemblance runs (free / loop / yoked / match / stale) |
| `report/` | the interactive report (`template.html` → `index.html`, built by `export_report.py`) |
| `qwen3-4b-base-jlens.npz` | the fitted J-lens used in every run |
| `runs/main{A,B,C}/` | `results.json`, `meta.json` (every thought and answer), `summary.txt` |
| `ring.py` | Part 3: the ring (placeholder substrate, online self-models, all conditions, paired noise) |
| `ring_analyze.py` | Part 3: the pre-registered tests for one run (or a folder of seed chunks) |
| `ring_job.py`, `ring_run.sh`, `ring_run2.sh`, `vast_self.sh` | Part 3: the unattended job on a rented GPU (pilot sweep → pick rule → main run → analysis → upload → the instance destroys itself) |
| `ring_gains.py`, `ring_ladder.py` | Part 3: the exploratory criticality ladder (synchrony, spectral exponent, attractor census); one table across models |
| `ring_words.py` | Part 3: what the J-space words of the recorded trajectories mean, checked with an independent embedding model (coherence, turnover, spread in meaning space, shared vs individual content, abstractness) |
| `docs/research_map.md` | related work for the J-space loop, and what is still open |
| `runs/ring/` | Part 3 results (summaries; the raw arrays are on Hugging Face) |

The raw trajectories (`traj.npz`, 160–175 MB per run) are too large for this repository; they are in the Hugging Face dataset linked above.

## Run it

```bash
pip install torch transformers scipy wordfreq
# qwen3-4b-base-jlens.npz: the J-lens for Qwen3-4B-Base at layer 16 (average Jacobian over 120 prompts), included
python strange_loop.py --grid main --seeds 10 --out runs/mainB --vocab english --n_iter 400 --t_int 200
python strange_loop.py --grid main --seeds 15 --conds free,loopc,yokedc,clampc --out runs/mainC --vocab english --n_iter 400 --t_int 150 --clamp_len 50
python analyze.py runs/mainB
python export_report.py report/data runs/mainA runs/mainB runs/mainC
```

A run of 60 minds × 400 thoughts takes about 15 minutes on one RTX 4090 and needs about 12 GB of GPU memory.

Part 3 on one 48 GB card. It is about 40 minutes for four models with 24 seeds each, plus 45 minutes for the ladder:

```bash
python ring_job.py --models qwen3-4b,qwen3-8b,qwen3-14b,qwen3-1.7b --root runs/ring
python ring_job.py --models qwen3-4b,qwen3-8b,qwen3-1.7b,qwen3-14b --root runs/ring --gains .25,.5,2,4
python ring_ladder.py runs/ring
```

To view the Parts 1–2 report locally, serve the folder (it loads its data files):

```bash
python -m http.server 8000 --directory report
```
