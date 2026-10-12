# What the failures taught (Parts 1–4, v3, v4, v5) — and what they point to

Every negative below is a fact about where a strange loop can and cannot live in a language model. Together they
narrow the search to one kind of system.

| # | Observation (where) | What it rules out | What it implies for the design |
|---|---|---|---|
| 1 | A frozen model's loop through one position of a fixed prompt has no state beyond (prompt, last write); a compact self-state adds no predictive advantage about the system's own write-induced dispositions (v5 G2, memoryless and memory loops); a planted hidden state is detected by the same test | a self-model of *private* processing in frozen, fixed-context loops | the loop must run in a system that keeps and shapes its own state: trained coupling, native recurrence, or the model's own generated context |
| 2 | The J loop converges by self-reproduction: stance / discourse / answering-frame words reproduce themselves (gain ≈ 1.6–2.0), names and entities do not (≈ 0.6); fixed points are precision-limited and locally stable within their words (v5 transport operator, fp32 spectrum) | expecting J-space, looped on itself, to keep world content | J-space is naturally a carrier of *process-level* content — how the system is speaking, its stance, its epistemic status (v3: the workspace tracks how settled the evidence is) — which is closer to a self-description than content is |
| 3 | Removing the loop's own dominant modes trades collapse for wandering through junk vocabulary, exactly like removing random directions (v5 self-deflation) | getting structure by subtracting what the loop already does | the third regime between collapse and noise needs a source of structured variation from inside the system — its own continuing thought or its own learning |
| 4 | At placeholder tokens the J space is blind to the conversation; prediction-error loops behave like momentum; any self-feedback loop never forgets a perturbation (Part 3) | placeholder substrates | the substrate must carry what the model is actually processing — generated thought or meaningful text positions |
| 5 | Strong writes into the decision state make the answer insensitive to contradiction; stance words do not act as a dial for readiness to revise (v5 stance test) | treating the written J content as a fine control of dispositions | dispositions should be read from behaviour the system produces (what it says, how it hedges), not only from two-candidate log-odds |
| 6 | A learned loop on a task that does not need it is bypassed (v4); benchmark accuracy as the endpoint was drift | task-driven loops | the loop must be necessary for the system's *own* objective (predicting its own continuing processing), with performance only a side measure |
| 7 | An online self-model, learning while its forecasts drive the loop, keeps fitting the current process (error 0.60 → 0.45) while losing its fit to what the system used to be (fixed-probe error 0.75 → 0.88); the loop does not simplify (Phase E, preliminary) | "self-modelling always collapses the system" in this setting | the 2024 tension is real here: the model chases a target it moves; worth studying in a system with private state |
| 8 | Self-related phenomena appear with scale: loop-self individuality 0.01 → 0.25 from 1.7B to 32B; self-talk "spiritual bliss" attractor and the Berg self-reference contrast only at 14B; near all-or-none ignition at 14B (Parts 2, 4) | concluding from 8B alone | the decisive runs should be at 14B (or 32B), with 8B for development |

## The system these facts point to

A loop whose compact state is (i) read from the model while it processes its own continuing thought or meaningful
text, (ii) carried forward in a way the system itself shapes — by a trained read/write coupling (the proposal's
Architecture B) rather than by recomputation from the input — (iii) predicted by a self-model whose forecasts are written
back, and (iv) under the two internal pressures of the 2024 idea: the system's ordinary job (predicting its own next
tokens) and self-prediction (predicting its own next compact state), with the strength of self-prediction swept. The
J-space enters as a candidate bottleneck against a learned one, and fact 2 predicts what it will carry: process-level
state (stance, epistemic status, discourse role) rather than entities.

What can be observed there that could not be observed in any run so far: a compact state that is necessary for the
system's own processing, its geometry as it forms (counting-like position codes, topic manifolds, stance axes), whether
self-forecast feedback reorganises it, and whether the self-model keeps up when its own use changes the system.
