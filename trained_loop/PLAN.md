# Phase B — a trained self-describing loop: the reversed autoencoder inside the language model

Status: revision 3, 12 Oct 2026 (17:30 UTC). Revision 2 was reviewed; the changes below
follow that review where it held up. Ran on a rented RTX 4090 48GB (8B) and A100 80GB (14B) (code/loopb.py).

## Why this system

Every frozen loop so far recomputed its state from the input (no private state: v5 G2, memoryless and memory loops) or
ran where the J space does not see the model's thought (Part 3). The author's 2024 reversed autoencoder was a *trained*
system whose bottleneck described the system itself and was fed back, under a tension between learning its own
behaviour and changing it. Here: the model processes its **own thought** in segments; the only bridge from one segment
to the next is a compact state z that the system reads from itself and writes back; a self-model predicts the next z,
and in the strange-loop arms its prediction is written back too. Only the system's own objectives train it.

## What is already known (so that this does not repeat it)

| work | what it showed | what it did not do |
|---|---|---|
| Premakumar et al. 2024 (arXiv 2407.10188) | an auxiliary loss for predicting one's own hidden activations makes networks simpler (narrower weights, lower RLCT), dose–response | the self-model's output is never fed back into the network |
| NextLat 2025 (arXiv 2511.05963) | a transformer trained to predict its own next latent learns belief-state-like latents | prediction not fed back into processing |
| attention-schema agents (Wilterson & Graziano 2021; Liu 2023; Farrell 2024; ASAC 2025) | a model of one's own attention can be used to control attention | small agents / benchmark modules, not a self-description inside an LLM's thought |
| compressive memory (RMT, AutoCompressors, gist tokens) | a compact learned state can carry context across segments | no self-model; "a state carries context" is not our question |
| Lindsey 2025/26; Lederman & Mahowald 2026 | models detect injected internal content, content-agnostically | no closed loop |

The untested core of the author's idea: **what changes when the self-model's output becomes an input of the process it
models.** Sharp form: under the same self-prediction pressure, does closing the loop turn self-simplification
(Premakumar) into self-control — the forecast becomes something the system follows — and what does the state then
represent?

## System (Qwen3-8B frozen, b968826d; 14B for the decisive contrasts)

* Data: the model's own thinking traces (thinking mode) for ~2 400 prompts (databricks-dolly-15k instructions, stratified
  by category; one trace per prompt; 2 000 train / 400 test), first 512 generated tokens. Closed system: the system
  models its own behaviour.
* Episode: K = 8 segments × 64 tokens. Segment k is processed **without** any earlier tokens (no KV); 6 placeholder
  positions at its start carry the writes, added at the output of block L_w = 22: 4 positions W(z_{k−1}), 2 positions
  W′(ẑ_k) (constant in arms without feedback). Segment 0 gets a learned z_init.
* Read: output of block L_r = 29 at the segment's last token → z (32-d) on a fixed-norm sphere (|z| = √32, so
  self-prediction cannot be satisfied by shrinking scale). Readers: **learned** (linear on per-dimension standardised
  residual), **J** (fixed: the J-lens word projection at L29 followed by its top-32 PCs on natural segment ends), **random**
  (fixed random 32 directions; control).
* Writes: per position n_w · normalise(M_j z + b_j), n_w = median natural residual norm at L22.
* Self-model S: MLP 32 → 256 → 32, output on the same sphere; loss L_S = mean_k (1 − cos(S(z_k), z_{k+1})).
* Training: next-token loss on segments 1–7 + λ L_S, backpropagation through all 8 segments; only R, W, W′, S trained.

| arm | self-model | forecast written | gradient of L_S into the system |
|---|---|---|---|
| A0 | — (loop cut: constant z) | — | — |
| A1 | — | — | — |
| A2 | observer (fitted after A1) | no | no |
| A2g(λ) | observer | no | **yes** (Premakumar's condition) |
| A3 | self-forecast | **yes** (stop-gradient target) | no |
| A4(λ) | self-forecast | **yes** | **yes** (the full loop) |

λ ∈ {0.1, 1} with the learned reader; A1, A2g(1), A4(1) with the J reader; A1 with the random reader.

## Hypotheses and pre-declared readouts (400 held-out traces, 95 % bootstrap CIs over traces)

* B1 necessity (sanity, expected): LM loss with own z < with a foreign z (other trace, same segment) and < time-shuffled z.
* B2 simplification (Premakumar replication in this system): A2g(λ) vs A1 — participation ratio of z, decodability of
  content (dolly category), LM benefit of the loop (A0 − arm), all lower with λ.
* B3 **closing the loop** (the new contrast): A4(λ) vs A2g(λ) at the same λ. "Self-control" predicts equal or better
  self-prediction with smaller losses in content information and LM benefit, and a large self-fulfilment index; "runaway
  collapse" predicts the opposite. Self-fulfilment index: replace ẑ_{k+1} by a donor's forecast (other trace, same k)
  and measure how far z_{k+1} moves toward the donor's forecast (cosine gain) and the LM-loss cost on the true
  continuation.
* B4 what the state carries: probes (grouped by trace) for segment index, dolly category, the next segment's own
  surprise (base-model NLL), hedging / reconsideration ("Wait", "maybe", "Hmm", "Actually"), self-reference ("I", "me");
  process-level vs content-level, J vs learned (LESSONS #2 predicts J carries stance / epistemic state).
* B5 geometry: dimension (PR, TwoNN), the shape of the segment-index code (a curved, counting-like manifold?), the local
  spectrum of S (a rotation = a self-predictable time code), all across arms in a common frame.
* B6 tracking and identity (H3): corrupt z at segment 3 with a donor's; does the system return to its own trajectory or
  adopt the donor's, and does a fresh self-forecast track the corrupted system better than a stale one?

Each outcome is informative: B3 is the original question in its cleanest form; B2/B3 together say whether feedback
changes what self-modelling does to a system.

## Cost

8B, batch 12, 8 segments × 70 positions, backpropagation into 14 blocks only: ~1.5–2 s per step (measured in the smoke
test before the runs), 600 steps per arm; 11 arms + trace generation + evaluation ≈ 3.3 h on a 48 GB GPU ≈ $2.3.
Credit $3.40; the 14B contrasts (A1, A2g(1), A4(1)) need a top-up of ~$2–3.

## Revision 3 (after review; this is what runs)

* **Matched arms.** The two channel positions carry: a constant (`const`), the previous state again (`state`: an extra
  memory channel of identical capacity) or the self-model's forecast (`fcst`). The self-model S is trained in every arm
  with its own prediction loss only (detached inputs and targets), so its learning is identical across arms. lam is
  the pressure on the *system* (reader, writers) to make its next state predictable: the same loss through a
  parameter-detached copy of S. Primary contrasts: `fcst` vs `state` at lam = 0 (does a self-forecast as input differ
  from a memory channel?) and at lam = 1 (closing the loop under self-predictability pressure), `const` lam = 1 vs 0
  (Premakumar's condition: state simplification, not parameter complexity), `cut` (floor), J reader for `const` lam 0
  and `fcst` lam 1.
* **Central evaluation in free generation.** Teacher-forced replay only changes hidden states. In evaluation the system
  also thinks on its own: segment 0 is its own trace, segments 1–7 are generated with only the loop as memory; own vs
  corrupted (a donor's state and channel at segment 3) vs channel-only swap, with identical sampling randomness; topic
  retention (embedding similarity to the own vs the donor trace) and states, with text examples.
* **Easy positives downgraded.** Necessity (own vs foreign / stale state) is a precondition, known from compressive
  memory work; a gate stops the run if the plain loop carries < 0.01 nats/token. Self-prediction is compared with
  persistence, segment-mean and linear predictors and with a post-hoc observer; boundary-token NLL is reported
  separately.
* **Probes with content baselines.** Features of the next segment (hedging, reconsideration, self-reference,
  answer phase, the model's own surprise) are predicted from z, from a bag-of-words of the current segment plus segment
  index, and from both.
* Single seed per arm: exploratory. Deferred to a second round if the loop works: transfer of a processing disposition
  across topics by transplanting a forecast component (review item 6), 14B replication (needs a credit top-up).
* Smoke test (8B, 6 steps): runs end to end, 1.4 s per step at batch 10, 31 GB. Reference NLL of the model's own
  thought: 0.30 nats/token in full context vs 2.0 for a segment alone (boundary tokens 0.31 vs 4.9) — the loop has a
  large gap to close.

## Predictions written before any result (12 Oct, 17:35 UTC)

1. `const` lam 0: the loop carries something (foreign − own > 0.05 nats/token), most of it on the boundary tokens.
2. lam = 1 offers the system three routes to self-predictability: lose information (any arm), **persist** (copy the
   previous state through the channel: `state`; S then learns ~identity) or **fulfil** (copy the written forecast into
   the state: `fcst`). Prediction: `const` lam 1 pays in information (lower PR, smaller LM benefit); `state` lam 1
   freezes (persistence baseline ≈ S_cos ≈ high, little change across segments); `fcst` lam 1 reaches high S_cos with
   *lower* persistence, i.e. the state keeps moving but along its own self-model's predicted path (large forecast-swap
   pull), at a smaller LM cost than `const` lam 1. If this holds: a written self-forecast lets the system be
   self-predictable without freezing — the strange-loop regime. If `fcst` lam 1 ≈ `state` lam 1 (both freeze), the
   forecast adds nothing beyond memory.
3. lam = 0: `fcst` ≈ `state` in LM benefit; forecast-swap pull small (no pressure to follow the forecast).
4. J reader: carries less trace-specific content than the learned reader (smaller necessity), more process-level
   features (hedging / reconsideration) relative to content (category).
5. Free generation: with only the loop as memory, topic similarity to the own trace decays over segments but stays
   above the donor's; corruption at segment 3 shifts later segments toward the donor's topic; in `fcst` lam 1 the
   channel swap alone shifts it too.
