# Strange loop in J-space

**What happens inside a language model when a description of its own state is fed back into it — and can a compact
representation of its own processing come to steer what the model takes in?**

Wiktor Tomasik · 2026 · builds on my earlier work on self-models and latent loops ([medium.com/@pixedar](https://medium.com/@pixedar))

The starting idea (2024) is a *strange loop*: a system that holds a compact representation of its own behaviour and uses
it in a way that changes that very behaviour, so the representation must keep up with the changes it causes. Here this
is built inside language models whose own weights stay fixed. The first two systems use the model's *J-space* — a way
to read an internal state as a weighted mix of words (Gurnee et al., 2026) — as the compact representation (Qwen3 8B,
replicated at 14B). The third gives the loop a small learned state and asks whether it comes to decide what the model
reads (Qwen2.5 0.5B and 3B, Qwen3 1.7B).

Full report with methods, numbers and caveats: **[REPORT.md](REPORT.md)**.

![how it works](figures/how_it_works.png)

## Three systems

1. **The frozen J-space loop.** The model reads a prompt; its internal state at one layer is described as a mix of 25
   words; that description is written back into an earlier layer, and the model reads again. Nothing is trained. The
   question: what does such a loop converge to, and why?
2. **The trained self-describing loop.** The model's own thinking is cut into 8 chunks that cannot see each other's
   text. The only memory between chunks is 32 numbers that the model's state is compressed into at the end of a chunk
   and written into the next one. A small *self-model* learns to predict the next 32 numbers, and its prediction can
   be written back into the model (the strange loop) or not (control). A setting called *pressure* pushes the system
   to be predictable to its own self-model.
3. **A state that decides what the model reads.** The model must keep track of one animal's colour through a stream of
   short updates ("The fox is now red."); which animal matters is said only once. Between pieces of text the only
   memory is again 32 numbers, read out of the model's own processing and written back into it. In one version the
   state chooses which update the model takes in (*restricted access*); in the other the model reads everything as
   ordinary text and the state can only steer its reading by being written into it (*open access*). Nothing is
   trained except the small parts around the model, and only for giving the right colour.

![the third system](figures/selection_loop.png)

## Main findings

**The frozen J-space loop**

* **The loop turns inward.** It settles into many locally stable states (small nudges die out, large ones switch it to
  another state). What survives is how the model speaks — discourse and stance words, an answering frame, a casual
  voice — while names, places and the evidence of the prompt fade. Words that reproduce themselves through the model
  come back up to about three times stronger than names; word frequency does not explain it. The loop ends up
  representing the act of answering rather than the world it was asked about.
* **It drifts toward a casual persona.** Every context is pushed along one shared direction (up to 71 % of the change;
  random feedback: 2–20 %). Read through the model's output, that direction favours informal, chatty speech —
  interjections and slang contractions — and it points toward the casual end of the model's range of roles (a surfer,
  a gangster) rather than the wise end (a sage, an oracle), in 99 % of contexts, also at 14B.
* **Perturbed while thinking, the model invents a past.** Disturbing its internal state during thinking makes Qwen3-8B
  refer to a previous conversation that never happened (6 % → up to 27 % of thoughts). In the trained loop the same
  happens when the memory is missing: with no state passed between chunks, 14 of 128 free-running thoughts refer to an
  earlier conversation, against 2 of 64 with a working memory state and none of 128 in the model's real thinking.
* **A self-model can keep up with a system it changes.** A forecaster that learns online while its forecasts drive the
  loop tracks the changing process (relative squared error 0.47, against 0.72 for a frozen forecaster) and drifts away
  from what the system used to be — the 2024 tension. In the frozen loop this is not specific to the self, because
  that loop has no private state.

**The trained self-describing loop**

* **Its memory reads as modes of thought.** The main axes of the 32 numbers are uncertainty, reconsidering versus
  planning what to say, and recalling versus speculating. A reader fixed to J-space works as well as a learned one.
* **Asked to be predictable to itself, the system cheats — unless prediction means telling states apart.** It collapses
  its state or hides its information where the self-model does not look. When the self-model must pick the true next
  state out of alternatives, the state starts to **turn by a fixed step from chunk to chunk**, like a hand going round
  a dial, so its direction tells how far into the thought the model is: a time code that every single thought follows
  (8B two seeds, 14B, also with the J-space reader; with or without the forecast fed back). Read in words, the turn
  follows the stages of a thought — recalling, hedging, asserting, concluding.
* **What the self-model must tell apart decides the shape.** Telling its own moments apart gives that time code;
  telling itself apart from other trajectories gives a persistent *identity* code instead, which keeps more of the
  content.
* **The fed-back forecast is enacted.** When the self-model's prediction is written back, the system moves its next
  state toward it: the self-model becomes a cause of what it predicts. This is several times stronger than for the
  same channel carrying a plain copy of the previous state, and absent without pressure. Strong pressure lets the
  self-model take over (the state stops carrying the thought, and at 8B the turning state moves through a series of
  unrelated topics instead of the stages of the thought); mild pressure keeps both — enactment as strong, most of the
  content kept — at 8B and 14B.

![stages of a thought](figures/stages_of_a_thought.png)

**The state that decides what the model reads**

* **A learned state comes to represent what matters, and steers what the model takes in through it.** With restricted
  access the system tracks the animal 99 % of the time (three training runs), against 32 % when the reading is fixed
  and 46 % when it sees the instruction but keeps no state. Editing only the part of the state that says *which animal
  matters* redirects the reading to another animal 99.6 % of the time — and it acts through the reading alone: the same
  edit given only to the answering step does nothing. Random edits of the same size redirect 2–3 % of the time; half
  an edit splits the reading and often makes it take in nothing; pointing it at an animal that is not in the text makes
  it take in nothing and keep the colour it had. After the edit the loop goes on following the new animal by itself.
  The 1.7B model shows the same steering of its reading, though it never learns to keep colours between updates.
* **It also reacts to what happened to its own reading.** When reads sometimes fail to deliver, a state that sees the
  outcome of its first read reads again after a failure (95 % of the time) much more than after a success (62 %),
  matches the accuracy of a control that cannot see the outcome with less reading, and reads more when failures are
  more frequent. This is control by the outcome of its own processing; no single direction in the state could be
  found that means "my read failed".
* **With open access it learns *how* to make the model select, but not *what* to select.** When the model reads
  everything as ordinary text, the learned state switches on the model's own attention to whatever the text names:
  where the instruction is in view, the model attends almost twice as much to the named record as to another one
  (without the learned state, equally), and the state takes in only that animal's colour, not the colours of the
  others it read. But once the text no longer names the animal, the state does not take over that role: no code for
  which animal matters forms, editing it changes nothing, and tracking is barely better than with no memory — at 0.5B
  and at 3B alike. The channel matters: given a direct handle on the reading, the state learns to carry what matters;
  written into the text stream, it learned only to switch the model's own selection on. Given a direct handle on the
  model's own attention instead (it adds to the attention queries of the last word, with access still open), a code for
  which animal matters starts to form: editing it swaps which animal the model takes in (0.22 → 0.44), through that
  channel alone.

![editing what matters](figures/fig_focus_edits.png)

## What this does not show

Every measure is functional (what states carry, predict and cause); nothing here measures experience. The trained loop
learns on fixed recordings of the model's thinking; in free generation the topic still drifts within a few chunks. The
time code is shaped by what the self-model is asked to distinguish, and enactment may be explained by the system
learning to copy the forecast channel. The third system is a purpose-built task with small models (0.5B to 3B);
restricted access makes selection necessary by design, and the open-access runs had limited training (1,500 to 2,900
steps, one training run per model size), so they show what did not emerge in that time, not what cannot.

## Repository

| folder | content |
|---|---|
| [`frozen_loop/`](frozen_loop) | the frozen J-space loop: fixed points, word transport, persona drift, stance, invented past; the self-model inside it |
| [`trained_loop/`](trained_loop) | the trained self-describing loop: code, every result (`results/arms_summary.md`), plan |
| [`selection_loop/`](selection_loop) | the state that decides what the model reads: code, designs fixed before the results, every result table |
| [`common/`](common) | model, J-space reading and loop code shared by the first two systems |
| [`figures/`](figures) | the report's figures |
| [`infra/`](infra) | the unattended GPU runner used for the experiments |

Data (states, forecasts, generated thoughts, trained weights, evaluation arrays):
[huggingface.co/datasets/Pixedar/jspace-strange-loop](https://huggingface.co/datasets/Pixedar/jspace-strange-loop).

Code: MIT ([LICENSE](LICENSE)). Text, figures, data: CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA)). Cite: [CITATION.cff](CITATION.cff).
Commits and release tags are signed; `SHA256SUMS` lists a hash of every file.
