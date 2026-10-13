# Strange loop in J-space

**What happens inside a language model when a description of its own state is fed back into it — and when it is
trained to predict, and then follow, its own next state?**

Wiktor Tomasik · 2026 · builds on my earlier work on self-models and latent loops ([medium.com/@pixedar](https://medium.com/@pixedar))

The starting idea (2024) is a *strange loop*: a system that holds a compact representation of its own behaviour and uses
it in a way that changes that very behaviour, so the representation must keep up with the changes it causes. Here this
is built inside Qwen3 language models (8B, replicated at 14B), using the model's *J-space* — a way to read an internal
state as a weighted mix of words (Gurnee et al., 2026) — as the first candidate for the compact representation.

Full report with methods, numbers and caveats: **[REPORT.md](REPORT.md)**.

![how it works](figures/how_it_works.png)

## Two systems

1. **The frozen J-space loop.** The model reads a prompt; its internal state at one layer is described as a mix of 25
   words; that description is written back into an earlier layer, and the model reads again. Nothing is trained. The
   question: what does such a loop converge to, and why?
2. **The trained self-describing loop.** The model's own thinking is cut into 8 chunks that cannot see each other's
   text. The only memory between chunks is 32 numbers that the model's state is compressed into at the end of a chunk
   and written into the next one. A small *self-model* learns to predict the next 32 numbers, and its prediction can
   be written back into the model (the strange loop) or not (control). A knob — the *pressure* — pushes the system to
   be predictable to its own self-model. The language model itself stays frozen.

## Main findings

* **The frozen loop turns inward.** It settles into many locally stable states (small nudges die out, large ones switch
  it to another state). What survives is how the model speaks — discourse and stance words, an answering frame, a
  casual voice — while names, places and the evidence of the prompt fade. Words that reproduce themselves through the
  model come back up to ~3× stronger than names; word frequency does not explain it. The loop ends up representing the
  act of answering rather than the world it was asked about.
* **It drifts toward a casual persona.** Every context is pushed along one shared direction (up to 71 % of the change;
  random feedback: 2–20 %), which reads out as "yeah, wanna, gonna, guys" and points toward the casual end of the
  model's persona space (gangster, surfer dude) rather than the wise end (sage, oracle) — in 99 % of contexts, also at 14B.
* **Perturbed while thinking, the model invents a past.** Disturbing its internal state during thinking makes Qwen3-8B
  refer to a previous conversation that never happened (6 % → up to 27 % of thoughts). In the trained loop the same
  happens when the memory is missing: with no state passed between chunks, 14 of 128 free-running thoughts refer to an
  earlier conversation, against 2 of 64 with a working memory state and none of 128 in the model's real thinking.
* **A self-model can keep up with a system it changes.** A forecaster that learns online while its forecasts drive the
  loop tracks the changing process (relative squared error 0.47, against 0.72 for a frozen forecaster) and drifts away
  from what the system used to be — the 2024 tension. In the frozen loop this is not specific to the self, because that loop has no private state.
* **The trained loop's memory reads as modes of thought.** Its main axes are uncertainty (maybe, hmm, perhaps),
  reconsidering vs planning what to say, recalling vs speculating. A reader fixed to J-space works as well as a learned
  one.
* **Asked to be predictable to itself, the system cheats — unless prediction means telling states apart.** It collapses
  its state or hides its information where the self-model does not look. When the self-model must pick the true next
  state out of alternatives, a **clock** emerges: one rotation, at several frequencies at once, carries the state from
  chunk to chunk, and single thoughts follow it (8B two seeds, 14B, also with the J-space reader; with or without the
  forecast fed back). Read in words, it follows the stages of a thought — recalling → hedging → asserting → concluding.
* **What the self-model must tell apart decides the shape.** Telling its own moments apart gives a clock; telling
  itself apart from other trajectories gives a persistent *identity* code instead, which keeps more of the content.
* **The fed-back forecast is enacted.** When the self-model's prediction is written back, the system moves its next
  state toward it: the self-model becomes a cause of what it predicts. This is several times stronger than for the
  same channel carrying a plain copy of the previous state, and absent without pressure. Strong pressure lets the
  self-model take over (the state stops carrying the thought; at 8B the clock turns into a carousel of unrelated
  topics); mild pressure keeps both — enactment as strong, most of the content kept — at 8B and 14B.

![word clock](figures/word_clock.png)

## What this does not show

Every measure is functional (what states carry, predict and cause); nothing here measures experience. The trained loop
learns on fixed recordings of the model's thinking; in free generation the topic still drifts within a few chunks. The
clock is shaped by what the self-model is asked to distinguish, and enactment may be explained by the system learning
to copy the forecast channel — the next experiments are designed to separate these.

## Repository

| folder | content |
|---|---|
| [`frozen_loop/`](frozen_loop) | the frozen J-space loop: fixed points, word transport, persona drift, stance, invented past; the self-model inside it |
| [`trained_loop/`](trained_loop) | the trained self-describing loop: code, every result (`results/arms_summary.md`), plan |
| [`common/`](common) | model, J-space reading and loop code shared by both |
| [`figures/`](figures) | the report's figures |
| [`infra/`](infra) | the unattended GPU runner used for the experiments |

Data (states, forecasts, generated thoughts, trained weights, loop-geometry arrays):
[huggingface.co/datasets/Pixedar/jspace-strange-loop](https://huggingface.co/datasets/Pixedar/jspace-strange-loop).

Code: MIT ([LICENSE](LICENSE)). Text, figures, data: CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA)). Cite: [CITATION.cff](CITATION.cff).
Commits and release tags are signed; `SHA256SUMS` lists a hash of every file.
