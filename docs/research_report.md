# The J-space loop: what it is, what it showed, what the words mean, what next

Status on 10 October 2026. This covers only the J-space recurrence ("the ring", Part 3 of the repository) and an
analysis of its recorded trajectories. Code: `ring.py`, `ring_analyze.py`, `ring_gains.py`, `ring_words.py`.
Data: https://huggingface.co/datasets/Pixedar/jspace-ring

---

## 1. Short answers

**What does "+2.07 more dimensions explored" mean?** It is not new dimensions, and not growth over time. It is the
effective number of independent directions the trajectory moves through. It was measured as the participation ratio
of the direction-normalised J-space states, in 100-step windows over steps 400–600, and averaged. Feeding back the
self-model's prediction error gave a value 2.07 higher than feeding back momentum (Qwen3-4B, 24 paired seeds).

The recorded trajectories show what this corresponds to:
- the active words turn over faster,
- the mind uses more distinct words,
- its position in an independent meaning space wanders through more directions,
- it jumps between more topic clusters.

The words active together are no more related to each other than random words, and the content is mostly shared by
all minds. So the evidence points to more diverse but incoherent activity. That is closer to "complicated but not
demonstrably meaningful" than to "more diverse semantic processing". Whether it helps a task is untested.

**Do the arbitrary-looking J-space words mean anything?**
- **At the level of words, no evidence.** It was tested with two independent embedding models (MiniLM and OpenAI
  text-embedding-3-large). In every condition, on 4B and 14B, the 25 words active together are no more related than
  random word sets from the same vocabulary.
- **Most content is a common baseline** that drifts with time and is the same in every condition.
- **The individual part is small and fleeting.** It carries 6–14 % of the weight, and most of it is gone ten steps
  later.
- **Not ruled out.** The decomposition is not unique, so abstract structure could hide behind arbitrary labels. That
  needs causal tests, not word lists.

**Does the loop make the content more abstract or more "meta"?** No, not by a meaningful amount. On an
abstract–concrete axis and a mind/self axis, conditions differ by less than 1–2 % of the axis scale. The small shift
toward mind words comes from every kind of error feedback, including another mind's error, so it is not self-specific.

**What does the loop change in what the model says?** Inside the loop the model says nothing: it only reads
placeholder tokens. The one verbal readout, "Putting it into words, what I was just thinking about was …" after 600
steps, does not depend on the condition. On 14B it is the same sentence everywhere. This is a weak readout, so it does
not prove that nothing changed. It shows no effect that reaches language.

**Is prediction-error steering steering the J-space meaningfully?** Not shown. There are three weaknesses, and they
are the next experiment:
- **The write is not calibrated.** The error is measured at layer 25 (J-space at 70 % of depth) and added directly at
  layer 18 (50 %), with no map between the two layers. Whether the push changes layer 25 in the intended direction
  was never measured.
- **Every error gets the same strength.** The error is normalised before it is written back, so even a tiny or
  unreliable error gets a full-size push.
- **Error vectors share a bias.** The words the error loop adds (*Trout, glucose, Ecology, Mutation, extinction…*) are
  the same for the mind's own error, another mind's error, a shuffled one and a frozen one. That is a property of
  error-like vectors, not of self-modelling.

---

## 2. How the loop works (`ring.py`)

**Substrate.**
- A chat prompt ("Let your mind wander. You don't need to answer anything, just think.") opens the assistant's
  thinking. Then the model reads one placeholder token ('…') per step, for 600 steps.
- At each step, the state at **70 % of depth** is decomposed into J-space content c. The Neuronpedia Jacobian lens is
  used, with non-negative pursuit over about 26,000 common English words, K = 25, relative to the typical placeholder
  state.
- Vectors are added at **50 % of depth** of the next placeholder. No sampled word is ever part of the loop.

**Carried state.**
- `x = 0.5 x + 0.5 c`, written back as `g · Rn · x/rc`, with its size capped.
- Rn is the median residual norm at the write layer, and rc is the typical content norm.
- Plus noise `sigma · Rn · (unit random vector)`. Minds with the same seed share every random draw, so two conditions
  differ only by what they write back.

**Self-model.**
- Every mind has its own online learner `m = c + U V^T c + b` (rank 32, with a skip connection, so it starts as
  "next = current").
- It is trained at every step on the transition that just happened (replaying the last 32, with a cosine loss and
  Adam). It therefore also learns the effects of its own output.

**Ring channel.** One extra vector of fixed size `s · Rn` (s = g), the same for every condition:

| Condition | What is written back |
|---|---|
| `ring_err` | the self-model's surprise e = c − m_prev |
| `ring_pred` | its prediction of the next state |
| `frozen_err` | like `ring_err`, with learning stopped at step 300 |
| `vel` | momentum c − c_prev (what `ring_err` is before the self-model has learned anything) |
| `yoked_err`, `yoked_pred` | another mind's surprise or prediction |
| `shuf_err` | the mind's own surprise from a random earlier step |
| `rand_err` | a random direction |
| `content` | the current content: plain feedback |
| `free` | nothing |

**Regime.** A fixed rule on a pilot sweep picks the largest memory before free minds collapse: g = 0.05–0.1 and
sigma = 0.05–0.15, depending on the model. There were 24 seeds × 600 steps per model.

**Tests**, uploaded before the run:
- exploration (participation ratio, as above),
- collapse (share of late steps with step-to-step cosine > 0.98),
- memory (lag-1 and lag-100 cosine),
- self-model skill,
- the freeze effect,
- privileged self-knowledge.

---

## 3. Results

**Pre-registered.** Exploration of `ring_err` minus each control, paired over 24 seeds. The last column is Stouffer's
z over models.

| `ring_err` minus | 1.7B | 4B | 8B | 14B | z |
|---|---|---|---|---|---|
| momentum | +1.57 | +2.07 | +2.48 | −0.31 | +26.9 |
| another mind's surprise | +0.41 | +1.31 | +0.20 | +4.33 | +15.9 |
| own surprise at a random earlier time | +2.27 | +2.01 | +1.22 | +3.09 | +28.0 |
| random direction | +3.16 | +3.65 | +3.34 | +4.75 | +52.9 |
| freezing the self-model | −2.20 | −3.90 | −1.87 | −4.99 | −33.0 |

- **Collapse.** Plain feedback collapses inside the latent loop: on 14B, `content` is stuck 85 % of the time and
  `ring_pred` 80 %, against 2.5 % for `ring_err` and 20 % for free minds.
- **Own prediction vs another mind's.** Feeding back one's own prediction does no more than feeding back another
  mind's.
- **No individual dynamics.** A predictor fitted to another mind of the same condition predicts a mind as well as
  one fitted to itself.

**Across the critical gain (exploratory).** Each substrate has a gain g\* where free minds turn 1/f: 0.03–0.09.
- **Below g\*:** minds move almost in lockstep (state cosine 0.8–0.95 between minds), and whose surprise it is hardly
  matters.
- **Past g\*:** own surprise and continued learning matter more and more, up to +5.5 and −8.0. From about 2 g\*,
  plain momentum resists collapse better than the self-model's error. The self-model learns to predict its own pushes
  and partly cancels them.

---

## 4. What the recorded trajectories show (`ring_words.py`)

Every step of every mind stores its 25 active J-space words and their coefficients. The analysis used the last 300
steps, 24 seeds × 10 conditions. Each word was also embedded with an independent model, MiniLM and OpenAI
text-embedding-3-large; both give the same conclusions, and the OpenAI numbers are shown.

| Qwen3-4B | free | content | momentum | ring_err | another mind's err | random |
|---|---|---|---|---|---|---|
| Words active together, relatedness above random sets | −0.006 | −0.003 | −0.005 | −0.005 | −0.004 | −0.006 |
| Overlap of the active words 10 steps later | 0.63 | 0.67 | 0.51 | 0.44 | 0.46 | 0.60 |
| Distinct words a mind uses (300 steps) | 294 | 258 | 310 | 356 | 359 | 319 |
| Spread in meaning space (participation ratio of the meaning centroid) | 17.9 | 14.0 | 21.0 | 22.8 | 21.6 | 19.4 |
| Words shared with another mind at the same step | 83 % | 76 % | 75 % | 65 % | 62 % | 78 % |
| Weight carried by the mind's individual words | 6 % | 9 % | 9 % | 13 % | 14 % | 8 % |

- **Same picture on 14B.** It has fewer words in play (70–150), and momentum explores meaning space as much as
  `ring_err` there (−0.35). Own surprise beats another mind's by +4.1 (24/24 seeds).
- **Paired over seeds on 4B, `ring_err` spreads further in meaning space** than momentum (+1.8, 24/24), another
  mind's surprise (+1.2, 24/24), own surprise at a random time (+2.4) and random input (+3.4). So the extra J-space
  exploration is not only numerical: the decoded labels really vary more.
- **But there are no themes.** At no step do the active words cohere. The common baseline is the same few words in
  every condition at the same moment (*Genetics, Syntax, Romero, construed, reprint, worthwhile…*), drifting with time.
  Individual content is small and short-lived.
- **The words the error loop adds belong to all error feedback.** The same words are over-represented for the mind's
  own error, another mind's, a shuffled one and a frozen one (*Trout, glucose, Ecology, Mutation, extinction,
  deputies…*).
- **Abstractness and mind words.** Both axes were checked first: held-out abstract words score +0.195 and concrete
  words −0.146. Between conditions the difference is under 0.003 on the abstract axis and under 0.008 on the mind
  axis. The shift toward mind words comes from all error feedback, and another mind's error gives at least as much.

---

## 5. The next version of the loop (proposal)

The goal is to fix the three weaknesses, and to add the second-order step of your "difference about difference"
ladder: correlation → prediction error → precision (which errors to trust) → valence (how that bears on the system) →
a self-model of that machinery.

1. **Calibrated write.**
   - Measure the causal map A from a push at the write layer to the change in J-space content at the read layer,
     using a few hundred probe pushes at the placeholder position.
   - Write `u = A⁺ δ` so the J-space content actually moves by the intended δ.
   - Report how often the intended change happens, for the current direct write and the calibrated one.
2. **Second-order self-model (prediction about prediction).**
   - A second learner predicts the first one's next error: its size (expected surprise) and its direction.
   - From it come the **precision** π = how much to trust the current error, and the **second-order error**
     e₂ = e − ê, the part of the surprise that was itself unexpected.
   - Conditions:
     - precision-weighted error (π · e, not normalised, so small or unreliable errors get small pushes),
     - second-order error,
     - first-order error as before.
3. **Valence and uncertainty.**
   - **Valence:** a slow signal for whether the first-order errors are shrinking (things are becoming predictable) or
     growing.
   - **Uncertainty:** the model's own next-token entropy at the placeholder, read alongside.
   - One condition lets valence set the gain of the error loop, playing the role of affect as a global precision
     control.
4. **Controls that need no learning.**
   - momentum,
   - a novelty controller (push away from the running average),
   - a low-pass filter (the running average itself),
   - a high-pass filter (the current state minus a fast running average),
   - random,
   - another mind's signal,
   - frozen self-models.
   If the learned second-order loop is doing something these simple dynamical systems cannot, it must beat all of
   them.
5. **A test of whether the loop carries and uses information.**
   - Give a word or a small problem before the loop. Then cut the model's direct view of it: the attention cache
     keeps only the first tokens and a recent window of placeholders, so the loop is the only channel.
   - After 0, 50, 150 and 300 steps, ask a controlled question: recall the word, its category, or the answer.
   - Does accuracy survive or improve with more recurrence, and does shuffling or replacing the feedback destroy it?
6. **Measures.**
   - Everything above, plus the trajectory analysis of section 4 run on the new data.
   - Exploration measured window by window across the whole run, to see whether it grows.

Cost estimate: one RTX 4090 48 GB for about 1–2 hours on Qwen3-4B and 8B, about $1–2.

---

## 6. Caveats

- One substrate (placeholder tokens) and one prompt.
- Filler tokens carry computation only in models trained to use them (Pfau et al. 2024), so this substrate may be
  unable to hold a thought at all. A natively recurrent model (Huginn, Ouro) would be the stronger substrate.
- The J-lens dictionary is overcomplete and the decomposition is not unique, so word labels are an imperfect window.
- The exploratory parts (the gain ladder, the word analysis) were added after the pre-registered tests and are
  labelled as such.

## 7. Files

| File | What it is |
|---|---|
| `ring.py`, `ring_analyze.py` | the loop and its pre-registered tests |
| `ring_job.py`, `ring_run.sh`, `ring_run2.sh`, `vast_self.sh` | unattended runs on a rented GPU, with self-destroy |
| `ring_gains.py`, `ring_ladder.py` | the criticality ladder; one table across models |
| `ring_words.py` | the word and meaning analysis of recorded trajectories (MiniLM, or OpenAI with `EMBEDDER=openai`) |
| `runs/ring/` | results, including `*/words/` for the analysis above |
