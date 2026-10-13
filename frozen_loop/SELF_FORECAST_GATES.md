# Phase Q — can the loop's compact state know something about the system's own processing?

Qwen3-8B (bf16), F2 interpretation-under-evidence scenarios, validation split (100 scenarios × 11 evidence variants),
12 Oct 2026, Vast RTX 4090 48GB. Code: research_v5/code/qual.py (memoryless loop), qual_mem.py (memory loop). Raw
results: research_v5/results/q/, HF Pixedar/jspace-v3 v5/q, v5/qm. Protocol: PROTOCOL_V5.md rev 3, amendments Q-1–Q-3
(§13), all made from smoke runs before full-size outcomes existed.

## What the gate asks

Before testing whether a self-forecast keeps up with changes its own use causes, the compact self-state z (32
coordinates: PCA of the J-lens reconstruction, Z_J, or of the standardised residual, Z_R) must carry information about
the system's processing that is not already given by the prompt's content, its current output and its own actions.
G2 measures it directly: under random writes, two identical predictors of the next write-induced change of the
contradiction response (Δc) — one given z of the system itself, one given z of a content-matched other rollout — and
the difference in cross-validated R².

## Memoryless loop (the v3 / revision-2 design): every step recomputes the decision position from the fixed prompt and the current write

| gate | result |
|---|---|
| G0 numerics | replay bit-identical; batch order no effect; cached vs full recomputation q95 0.50 nats (lo) / 0.62 (c); bf16 vs fp32 q95 0.64 / 1.06 nats, no sign flips — rounding on large log-odds |
| G1 | natural contradiction response 34 nats, irrelevant-sentence control 2.9 nats; candidate mass ≥ 0.5 in 90 % |
| gain rule | RMS Δc 1.0 / 2.0 / 4.2 / 9.7 nats at g = 0.25 / 0.5 / 1 / 2 (linear); g* = 1 (threshold 3.1 nats = 5 × path noise) |
| G1(c) | signal-to-noise of the write-induced change 6.7 |
| **G2** | Z_R: ΔR² = +0.001 [−0.013, 0.016]; Z_J: −0.008 [−0.022, 0.005] (both predictors R² ≈ 0.43; 4 760 transitions) — **fail** |
| G3 positive control | a hidden state component (σ v_h at L22) is picked up by the same contrast (Z_R / Z_J): ΔR² +0.33 / +0.25 on Δlo, +0.071 / +0.044 on Δc, tight intervals; σ decodable from z at 0.77 / 0.69 (strict threshold 0.9 not met) |
| pilot | provisional forecaster's closed-loop skill on Δc ≈ 0.01; fresh vs stale forecasts after a corruption: ratio 1.00; donor coverage 14 % / 45 % |

**Diagnosis (conditional).** The loop is memoryless by construction: the state at step t is a function of (prompt,
write at t − 1), and the observables at t + 1 are a function of (prompt, write at t). Given the complete prompt and the
writes, the state adds no Shannon information; the G2 baseline gets coarse content labels and the two write codes, and
the predictors are finite MLPs, so the precise statement is: no incremental predictive advantage of the compact state
was detected, and none is expected unless the compact state compresses scenario details beyond the labels. G3 shows
that the measurement can see state information when it exists.

## Memory loop (amendment Q-4): every loop step stays in context as a new position

The decision token is appended at each step with the write added at L22; later steps attend to all earlier loop states,
so the system's state now depends on its whole history of writes. Deltas are taken against the same prompt's no-write
memory rollout (repeating the token alone drifts lo by 2.6 nats after one step and 8.1 after six).

| gate | result |
|---|---|
| G0 | replay bit-identical; batch order no effect; cached vs full q95 0.62 nats |
| gain rule | RMS Δc 0.42 / 0.77 / 1.53 / 3.23 at g = 0.25 / 0.5 / 1 / 2 (≈ 3× weaker than memoryless); g* = 2; under continuing random writes the RMS change at the last step is ~36 % of that at the first step (a ratio of response sizes, not a test of single-write persistence) |
| G1(c) | SNR 5.2 |
| G3 positive control | hidden component decodable from z: Z_R 0.98 (passes 0.9), Z_J 0.84; contrast detects it: ΔR² on Δlo +0.36 [0.32, 0.40] (Z_R) / +0.21 (Z_J), on Δc +0.034 [0.007, 0.059] / +0.007 (below the declared 0.05) — the measurement works; the hidden mode moves the answer far more than the disposition |
| **G2** | own vs content-matched swap: Z_R −0.007 [−0.029, 0.014]; Z_J −0.033 [−0.055, −0.013] — **fail**. Adding z to the full action history (ε_0 … ε_t): −0.04 / −0.06 (worse: no information, extra inputs). The full action history itself predicts no better than the last two actions (R² 0.31 vs 0.33). |

## What this establishes

* In Qwen3-8B's decision-position loops — memoryless or with memory — a 32-coordinate compact readout of the read-layer
  state (J-derived or residual-derived) gave no incremental advantage in predicting how the system's own writes modulate
  its disposition to change its answer under contradiction, beyond the prompt's content, its current output and its
  recent actions. The modulation is predictable (R² 0.33–0.43), but from those. Caveats (review): readers fitted on
  no-write states may discard intervention-sensitive directions; six repeated decision tokens give limited memory; the
  contradiction response compares separately evolved prompt histories rather than challenging one accumulated state.
* This is the precondition for a self-forecast that knows something about its own processing; it fails here, so the
  confirmatory Phase P was not run (inconclusive under the protocol). It agrees with v3's H2 result (dispositions
  explained by the evidence label at 8B) and extends it to dispositions modulated by the system's own writes.
* Exploratory Phase P / E (one writer, Z_R) run afterwards, labelled as exploration: §E below.

## E. Exploratory Phase P (memoryless loop, writer 1, Z_R, g* = 1; test set 2 200 items) — not confirmatory

* Training (policy iteration): validation loss 0.634 / 0.634 / 0.582 / 0.758 over four rounds — not monotone, because
  every new forecaster changes the process it must predict. Own-policy skill of the final forecaster on the
  write-induced disposition change Δc (h = 1): 0.20. The content-matched control forecaster F_swap reaches the same
  validation loss (0.733 vs 0.758).
* E1 (self-forecast vs content-matched control on the same closed-loop outcomes): −0.21 [−0.32, −0.12] — forecasting from
  the system's own state is *worse* than from a content-matched other state.
* E2 (manipulation): corrupting only the disposition forecast at step 2 changes the following Δc by RMS 1.53 nats; a
  random write displacement of the same size changes it by 1.29 — the forecast channel acts like an ordinary write
  perturbation.
* E3 (tracking): fresh forecasts after the corruption beat stale ones by 8 % (tracking ratio 0.92 [0.85, 0.99]; Holm
  interval [0.81, 1.03]; the declared threshold, upper bound < 0.9, is not met); noninferiority 0.88 [0.79, 1.01].
  E4: the content-matched control tracks less (difference −0.11 [−0.18, −0.02]; Holm [−0.21, 0.03]).
* E5: after a one-step corruption, restoring the system's own forecasts brings the trajectory closer to the
  uncorrupted one (distance 0.69) than a matched sham restoration (0.97) or continued corruption (2.07).
* Donor coverage was 4.7 % (104 rollouts, 70 scenarios), far below the declared 70 %, so E3–E5 rest on few rollouts.
* Reading: in a memoryless loop the forecaster can only map content and its own writes to their consequences; weak
  tracking after a corruption is expected mechanically (the fresh state determines the next write). Nothing here
  indicates self-knowledge beyond content and action.

## What would make the gate passable (design consequences)

The disposition must depend on something the system keeps internally that is not a function of its input and recent
actions. Candidates, in order of directness: (i) a trained coupling (the proposal's Architecture B) where the
read/write interface and a LoRA can learn to store state in the loop; (ii) natively recurrent models (Ouro, Huginn)
whose recurrent state is not recomputed from the input each pass; (iii) longer self-generated contexts (the loop
running during the model's own thinking), where the system's earlier processing is genuinely part of what it is.

## E2. Exploratory Phase E — the 2024 tension (memoryless loop, writer 1, Z_R, 60 batches × 64 rollouts) — not confirmatory

Error = standardised squared error of the forecaster's one-step forecast of Δc on the outcomes of the batch it
forecast (before its update); probe = the same on a fixed set of transitions from the original policy (Phase P's F_4).

| system | batch error, 0 → 50 | probe error, 0 → 50 | late step of z | late-state dimension |
|---|---|---|---|---|
| online (forecasts written, learning) | 0.60 → 0.44–0.50 | 0.75 → 0.88 | 0.08–0.16 | 12.4–14.5 |
| frozen (P's F_4) | 0.66–0.76 (one spike 1.14) | 0.75 | 0.08–0.10 | 12.2–13.3 |
| swap (learning, content-matched other state as input) | 0.92 → 0.46–0.50 | 0.82 → 0.99 | 0.08–0.15 | 12.7–14.5 |
| observer (learning, forecasts not written) | 0.43 → 0.35–0.39 | 0.78 → 0.97 | 0.044–0.049 | 10.7–13.6 |

* The learning self-forecaster keeps up with a process its own forecasts keep changing (≈ 0.47 vs ≈ 0.72 frozen), and
  drifts away from what the system used to be (probe error rises) — the 2024 tension in miniature.
* A forecaster given a content-matched other state keeps up just as well (≈ 0.48): in this memoryless loop the advantage
  comes from learning the content-and-action mapping, not from knowing its own state (consistent with G2 and E1).
* No self-simplification: late step and dimension do not fall over learning.
* The observer's target does not move, but its loop also differs (no forecast input, much smaller late motion), so the
  cost of self-influence is not isolated by this comparison; a matched design needs the forecast written in both systems
  with only the learning switched (e.g. replaying the online system's forecasts into a frozen-learning twin).
