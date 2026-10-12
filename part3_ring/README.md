# Part 3: the ring

No text in the loop: placeholder tokens, an online self-model per mind, its current surprise written back, every control of equal size, tests uploaded before the first result. Raw arrays: [Pixedar/jspace-ring](https://huggingface.co/datasets/Pixedar/jspace-ring).


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

## Run it (one 48 GB card)

```bash
python ring_job.py --models qwen3-4b,qwen3-8b,qwen3-14b,qwen3-1.7b --root runs/ring
python ring_job.py --models qwen3-4b,qwen3-8b,qwen3-1.7b,qwen3-14b --root runs/ring --gains .25,.5,2,4
python ring_ladder.py runs/ring
```
