# Strange Loop in J-space

A language model left to wander with no memory of its own words, only a latent state. Every ten thoughts it is asked
what its mind has been doing. It answers from that state alone, and the answer is fed back into everything it thinks
next. The question: does something like a persistent self form inside that loop?

```
thought trajectory → compressed model of the trajectory → model shapes the trajectory → new compressed model → …
```

Qwen3-4B-Base · J-lens at layer 16 · 3 runs · 180 minds · 72,000 thoughts · one RTX 4090 (Vast.ai), about $0.87

## The short answer

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
| `report/` | the interactive report (`template.html` → `index.html`, built by `export_report.py`) |
| `qwen3-4b-base-jlens.npz` | the fitted J-lens used in every run |
| `runs/main{A,B,C}/` | `results.json`, `meta.json` (every thought and answer), `summary.txt` |

The raw trajectories (`traj.npz`, 160–175 MB per run) are too large for this repository and are not included.

## Run it

```bash
pip install torch transformers scipy wordfreq
# qwen3-4b-base-jlens.npz: the J-lens for Qwen3-4B-Base at layer 16 (average Jacobian over 120 prompts), included
python strange_loop.py --grid main --seeds 10 --out runs/mainB --vocab english --n_iter 400 --t_int 200
python strange_loop.py --grid main --seeds 15 --conds free,loopc,yokedc,clampc --out runs/mainC --vocab english --n_iter 400 --t_int 150 --clamp_len 50
python analyze.py runs/mainB
python export_report.py report/data runs/mainA runs/mainB runs/mainC
```

A run of 60 minds × 400 thoughts takes about 15 minutes on one RTX 4090 and needs about 12 GB of GPU memory. To view
the report locally, serve the folder (it loads its data files):

```bash
python -m http.server 8000 --directory report
```
