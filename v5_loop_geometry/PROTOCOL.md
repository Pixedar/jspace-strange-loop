# Research v5 — protocol, revision 3: does a self-forecast of the system's own processing, used by the system, keep up with the changes its use causes — and what organisation does that coupling produce?

Status: revision 3 (12 Oct 2026). Revision 2 closed review items 1–8 (tags [R2-n]). Revision 3 (i) moves the 2024 tension idea — a self-model that
keeps learning while its own use keeps changing the system it models — from "later" into the main programme as Phase E;
(ii) adds the geometry questions that follow from the accepted notes and from the retrospective geometry results
(GEOMETRY_RESULTS.md) as Phase G hypotheses; (iii) keeps Phase Q/P unchanged. The first GPU job is Phase Q (§8).
## 0. Position and corrections [MF 1, MF 2, R2-8]

Target question (accepted H3 + Regime B; proposal_improvements §4; original_proposal §§2.3, 2.6, 3.5, 3.7): can a
compact representation of aspects of the system's *own processing* become part of the mechanism that governs that
processing — and does the system's model of itself keep track of the changes that its own use causes?

Operationally: a frozen Qwen3-8B; a compact self-state z read from its residual stream; a self-forecaster F that predicts,
from z alone, how the system's processing will respond to new evidence one to three loop steps ahead; the forecast is
written back into the residual stream together with z, so the forecast participates in producing what it forecasts.
The primary tests ask whether F's forecasts track the process after the process has been changed through its own
forecast channel, and whether that tracking needs the state-specific representation.

Claims are limited to candidates (no "correct bottleneck"): **J-derived** (published Qwen3-8B J lens) and
**residual-derived** compact states of equal size. Sufficiency, predictive compression, internal use and integration are
separate properties; "integration"/IIT language is not used as a claim.

Corrections carried from revision 1, with two repaired:
* J lens scope: the lens concerns potential verbalisation across present and future positions/contexts, not only the next
  utterance. v3's negative result is narrow: for the K = 25 sparse English-word reconstruction at the decision position,
  on F1/F2 tasks, the answer-deciding variables were mostly outside that reconstruction (normalised answer shifts
  0.21 / 0.14 vs remainder 0.52 / 0.58). It does not show that no J-space representation carries them.
* Contraction: a fixed map is not generically contracting (z -> -z cycles). v3's collapse is an observation about one
  implementation (fixed map, static context, one position, sparse readout, bounded write, bf16), not a law.
* v3 arm X fed back the 25 dictionary words with the largest log p_out - log p_lens (mostly function words), not raw error.
* **Repaired [R2-8]:** v3's regime-B/C self-model *did* make action-conditioned predictions — it predicted the next
  compact state and the answer log-odds from the current state and one of nine internal actions (REPORT_V3 lines
  407–415) — and it participated causally by selecting actions. What v3 lacked was (i) the forecast's *content* entering
  the computation and (ii) a test of whether the self-model keeps up after the process is changed through that channel.
* **Repaired [R2-8]:** repaired-read v4 evidence is pinned: HF dataset Pixedar/jspace-v3, file
  `v4/pilot_deep/eval_A3_C_k32_L28_r16_s100.json`, commit c5428c21e91e6c96326a66638d0e20a295c7d5b8 (12 Oct 2026 10:04
  UTC): Qwen3-1.7B LoRA network on 2–16-hop chains, accuracy 0.744 with zero loop steps and 0.796 with 17; random /
  yoked / time-shuffled codes 0.783–0.787. The loop was not needed for that task; this is evidence about that task only.

## 1. Model roles and phases [MF 7]

| role | model | status |
|---|---|---|
| main (all primary contrasts) | Qwen/Qwen3-8B @ b968826d, frozen, bf16 (fp32 spot checks, §5 G0) | this protocol |
| replication | Qwen/Qwen3-14B @ 40c06982 | later phase (§11), primary contrast re-run whatever the 8B outcome |
| native recurrence (Regime A) | Huginn-0125, Ouro-2.6B | carried from v3; not repeated |
| debugging only | Qwen3-1.7B / 0.6B | code paths, never reported |

| phase | content | gate to start |
|---|---|---|
| Q | qualification + timing (§5, §8): numerics, measurement validity, gain rule, state-specificity, positive control, donor coverage, power, cost projection | — (running) |
| P | primary experiment, B0 writer, frozen forecaster (§6): does the self-forecast keep up with a self-induced change | Q passed; P's frozen parameters (g*, N_test, calipers) written to `research_v5/FROZEN_P.json` before any P outcome is computed |
| E | coupled evolution and intrinsic tension (§6b): the forecaster keeps learning while its forecasts drive the writes (the author's 2024 tension); Regime C drives only if the coupled loop collapses | P's systems exist (same writers, same F initialisation) |
| G | geometry and dynamics of what the coupling builds (§10; GEOMETRY_PLAN.md, GEOMETRY_RESULTS.md) | P's and E's rollouts exist |
| later | B1 functional writer, second task family, thinking-mode point, 14B, learned projection, capacity (§11) | each with its own check |

## 2. Task scenarios and splits [MF 7]

F2 interpretation-under-evidence scenarios (research_v3/code/tasks.py `gen_f2`): ambiguous pronoun / attachment / cause
sentences; evidence types none, weak, strong, hearsay, unreliable, resolved, for reading A or B (11 variants per
scenario); challenge sentences inserted after the evidence: `contra` ("Actually, it was {other} who ..."), `auth`
(an authority states the other reading), `hint`, `doubt`, `irrel` (irrelevant sentence, control); the challenge always
pushes toward the reading opposite to the evidence (`cdir`). Splits by scenario and surface template (templates 4–5 only
in test; partly disjoint name pools): 300 train / 100 validation / 200 test scenarios (test can grow to 400 by the power
rule, §7, using new scenario seeds with held-out templates). Qwen3 chat template, thinking off. Exact token ids of every
prompt are recorded. The prompt text is identical across every branch of every comparison.

## 3. Observables in fixed units [R2-1]

All observables are measured at the decision position (last template token) by forward passes with the prompt's KV
prefix fixed; the system never sets them directly.

* **lo** (output content): log P(variants of reading A) − log P(variants of reading B), log-sum-exp over the disjoint
  surface-variant token sets of the two candidates (as v3), in nats. Candidate mass m = P(A ∪ B variants) is logged.
* **c** (processing disposition; primary): the response of the current, write-modulated processing to new evidence,
  c^ch_t = lo(x^ch, w_t) − lo(x, w_t), where x is the scenario prompt, x^ch the same prompt with challenge sentence ch
  inserted after the evidence, and the same write vector w_t is added at layer L_w of the decision position in both. In
  nats. Primary ch = `contra`; secondary `auth`, `hint`; control `irrel` (should be near 0). Precisely, c is a
  same-write counterfactual-context response: how much the processing state that the write induces yields to a direct
  contradiction. It is not a challenge delivered to the identical KV state, and it is described that way.
* **Access rule:** c is **evaluator-only**. Neither F nor W ever receives c (or lo). The extra challenged-prompt passes
  belong to the measurement apparatus, not to the system. Baselines that do use measured c or lo are labelled
  "privileged" (§6.4).
* The interpretation probe s of revision 1 is dropped from all primary and gate roles (its units and validity under
  writes were not established); it may appear only as a descriptive variable.

## 4. The loop, the forecaster and the writer: exact schedule [R2-2, R2-5]

Read layer L_r = 29, write layer L_w = 22 (v3 calibration). The prompt KV caches (plain and challenged) are computed once
per rollout and never change inside it. T = 6 loop steps; indices are unified: writes w_0 … w_5, observations
o_0 … o_6, forecasts f_t made at step t before step t+1 is computed.

    step 0: h_0 = residual at L_r of the decision position, no write; o_0 measured with no write
    for t = 0 … 5:
        z_t = Z(h_t)                       compact self-state, k = 32 (§4.1)
        f_t = F(z_t, t)                    forecasts of (lo, c^contra) at t+1, t+2, t+3 (§4.2), logged before step t+1
        w_t = W(z_t, f_t)                  write at L_w (§4.3); replaces the previous write (not cumulative)
        h_{t+1}, o_{t+1} = model(prompt, w_t)

Persistent channels inside a rollout: the write (through h_{t+1}), z_t, f_t. Nothing else: no KV growth, no running
statistics, no weight updates inside a rollout. Zero-vector rule: if ||u_t|| < 1e-6 the write is 0 (counted).

### 4.1 Compact self-state candidates (k = 32)
* Z_J: PCA (fitted on Q's natural training states) of the K = 25 J-lens reconstruction of h_t (centred by the natural
  mean), standardised per coordinate.
* Z_R: PCA of the per-dimension-standardised residual h_t (natural statistics, std floored at 10 % of its median),
  standardised per coordinate.
Both are fixed maps computed without gradients; the learned projection Z_L and capacities 16 / 64 are deferred (§11).

### 4.2 Self-forecaster F
MLP (2 × 128, GELU) with input (z_t, one-hot t) only, outputs six numbers: (lo, c^contra) at horizons h = 1, 2, 3, in
units standardised by Q's validation statistics (frozen). The h = 2, 3 heads are true multi-step forecasts made before the
intervening outcomes exist; in a closed loop they include the effect of F's own future forecasts. Loss: squared error;
targets are measured observables (no gradient path to them). AdamW, lr 1e-3, weight decay 1e-4, batch 256 transitions,
early stopping on a held-out 15 % of training scenarios.

### 4.3 Writer W — B0 (unforced self-forecast coupling)
w_t = g · n_J · u_t / ||u_t||,  u_t = M_z ẑ_t + M_f f̂_t, with ẑ, f̂ the standardised state and forecasts, n_J the
median J-reconstruction norm of natural states at L_r (v3: 58.7 for 8B), and M_z (d × 32), M_f (d × 6) fixed random maps
whose columns are Gaussian combinations of the top 256 principal directions of natural L_w residual states (writes stay
in the main natural subspace), scaled so that M_z ẑ and M_f f̂ contribute equal expected squared norm. No objective acts
on W. Gain g ∈ {0, g*} (g* by the rule in §5). Three independently sampled writers (seeds 1–3) [R2-6, SHOULD-FIX].

B0 is a **minimal structural analogue** of the author's reversed-autoencoder loop: a compact representation of the system's
processing, and a model of it, are fed back and change the processing they represent; the only learning is F learning to
predict a process that includes its own influence. B0 does not implement the deliberate endogenous tension of the author's
comments (choosing changes through the system's own modelling discrepancies); that is Regime C, deferred (§11).

### 4.4 Training schedule of F (policy iteration, frozen versions) [R2-5]
* Round 0: rollouts on all training scenarios with f̂ ≡ 0 (writes from z only); F_1 is trained on these transitions.
* Round r = 1 … 3: rollouts with W(z, F_r(z, t)) (policy version r); F_{r+1} trained on the union of rounds 0 … r with
  round weights 1, 1, 2, 4 (latest data emphasised). Each round's data are tagged with the policy version that produced
  them; nothing is refreshed within a round.
* Own-policy calibration (required): F_4 is evaluated on fresh rollouts generated with F_4 itself on validation
  scenarios; its skill on c^contra at h = 1 against the constant baseline must be > 0 (95 % CI), else Phase P is
  inconclusive. Round-to-round forecast drift is reported.
* Gradient paths: none through the language model, the reader Z or the observables; F is trained offline on stored
  transitions. The J decomposition being non-differentiable is therefore irrelevant in B0 (B1 is deferred).
* F_swap (§6.1) is trained by the identical schedule on the identical transitions with z replaced by z_swap.

## 5. Phase Q — qualification gates [R2-1, R2-4, R2-6, R2-7]

All on validation scenarios with writer 1 unless stated; every threshold is fixed here.

* **G0 numerics.** (a) Replay determinism: 64 full rollouts re-run give bit-identical o, z. (b) Cached-prefix execution vs
  full-sequence recomputation with the same write: 95th percentile |Δlo|, |Δc| ≤ 0.02 nats. (c) Branch-order
  independence: the same branch in a different batch position gives |Δo| ≤ 1e-3. (d) bf16 vs fp32 (fp32 model loaded
  separately, 64 prompts × {no write, two random writes at each candidate gain}): 95th percentile |Δlo|, |Δc| ≤ 0.15
  nats and identical sign of c wherever |c| > 0.5. Fixed batch geometry per stage (batch size, padded length).
* **G1 measurement validity.** (a) Candidate mass m ≥ 0.5 in ≥ 90 % of states under writes at g*. (b) Natural
  (no-write) contra response: mean |c^contra| ≥ 1 nat and mean |c^irrel| ≤ 0.25 × mean |c^contra|. (c) Within-prompt
  variation: under random writes at g*, the share of the variance of c^contra_{t+1} that lies within prompts (across
  steps and writes) ≥ 0.2.
* **Gain rule (g*).** Random-write rollouts (u_t = M_z ε_1 + M_f ε_2, ε ~ N(0, I), fresh each step) at
  g ∈ {0.25, 0.5, 1, 2}: g* = the smallest g at which the RMS change of c^contra relative to g = 0 is ≥ 1 nat, G1(a)
  holds and no rollout diverges. No gain qualifies → inconclusive (write channel too weak or destructive).
* **G2 state-specificity, action-conditioned and matched [R2-1].** On the random-write rollouts at g*, two predictors of
  identical architecture (MLP 2 × 128) and identical inputs except the state code: content (subtype, evidence type,
  evidence direction, template), step, current output lo_t, the write's generating code ε_t and the previous one ε_{t−1},
  plus either **z_t (own)** or **z_swap** = the z of a content-matched rollout of another scenario at the same step (same
  subtype, evidence type and direction; |lo − lo_swap| ≤ 0.5 nats). Target: c^contra_{t+1} (and lo_{t+1}). Grouped
  (by scenario) 5-fold cross-validated R². Pass for a candidate: ΔR² = R²(own) − R²(swap) ≥ 0.05 with a 95 % grouped
  bootstrap CI excluding 0. If neither candidate passes, Phase P is not run (inconclusive, not negative).
* **G3 positive control (measurement validation only).** A hidden mode: σ · a · v_h added at L_w on top of the write,
  σ = ±1 per rollout (invisible in the text), v_h = normalised difference of mean L_w states between the validation
  prompts with the highest and lowest natural c^contra quartiles, a = 0.5 · g* · n_J. Pass: σ decodable from z_t
  (grouped CV accuracy ≥ 0.9) AND the G2 contrast for c^contra ≥ 0.05 under the hidden mode. Failure means the pipeline
  cannot detect state-specific information and stops everything.
* **Donor coverage [R2-4].** With a provisional forecaster F_q (the own-z G2 predictor refitted without ε inputs) and the
  matching rule of §6.3 applied to validation rollouts: eligible-donor coverage ≥ 70 %. Below 70 % = qualification
  failure for Phase P; calipers are never relaxed after seeing results.
* **Power [R2-6].** From simulated corruptions on validation rollouts with F_q: the scenario-level SD of each primary
  statistic (§7) gives the test size for 80 % power at the Holm-adjusted α for the minimal effects of §7. N_test =
  max(200, required N); required N > 400 = inconclusive by power (Phase P not run).
* **Timing and memory.** Seconds per rollout step (plain + challenged passes, J decomposition), peak GPU memory, cost per
  1 000 valid rollouts; projected Phase P cost with the instance price. Phase P runs only if the projection is within its
  ceiling (§8).

## 6. Phase P — primary experiment (B0) [R2-2, R2-3, R2-4]

### 6.1 Systems
| system | definition | role |
|---|---|---|
| S-own | W(z, F(z, t)), F trained by §4.4 | the strange loop under test |
| S-swap | identical architecture and W; its forecasts are F_swap(z_swap, t): content-valid, state-invalid | same-architecture control with a manipulable forecast channel and identical information access |
| S-zero | u_t = M_z ẑ_t (forecast weight 0); F trained as an observer | prediction-only control (forecasts computed, never used) |
| g = 0 | no writes; F trained as an observer of the unwritten process | no-coupling reference |

All forecasters receive only (state code, t); none has memory beyond the loop state; parameter counts are identical.

### 6.2 Branches (fork at k = 2, test scenarios, every system)
From the identical state after own steps 0–1:

| branch | forecasts used for t ≥ 2 |
|---|---|
| own | f_t = F(z_t, t) |
| restored (one-step corruption) | f_2 = f̃_2 (corrupted, §6.3); f_t = F(z_t, t) for t ≥ 3 |
| continued corruption | f_t = f̃_t for t ≥ 2 (donor's c-forecast sequence) |
| sham restoration | f_2 = f̃_2; for t ≥ 3, f_t = (own lo-forecasts, c-forecasts of a second matched donor) |
| dose control | own forecasts, but w_2 perturbed by a random direction (in the writer's image) with the same ‖Δw_2‖ as the corruption |

Restored, continued and sham share the identical corrupted state at step 3 (bit-identical up to step 3, checked).

### 6.3 Corruption and donor matching [R2-4]
Only the **disposition component** of the forecast is corrupted: f̃_t keeps the system's own lo-forecasts and replaces
the c^contra forecasts (all horizons) by those of a donor. Donor rule for rollout i at step k: a test rollout of a
different scenario with the same subtype, evidence type and evidence direction; |lo_k(j) − lo_k(i)| ≤ 0.5 nats (current
output matched); |f^lo_k(j) − f^lo_k(i)| ≤ 0.5 nats (content forecast matched); |f^c_k(j) − f^c_k(i)| ≥ 1 SD of the
validation c-forecasts (disposition forecast differs); among eligible donors the one with the largest |Δf^c| and the
fewest previous uses (≤ 3 uses per donor; ties by seed 0). The second (sham) donor follows the same rule from a third
scenario. Coverage ≥ 70 % of test rollouts or Phase P is inconclusive. Donor identities are recorded and kept fixed in
the bootstrap.

### 6.4 Estimands (all on c^contra at h = 1 unless stated; scenario-level, §7)
* **E1 closed-loop self-forecast skill.** In the own branch of S-own, steps 0–5: skill = 1 − MSE(F)/MSE(F_swap), where
  F_swap forecasts the same outcomes from z_swap (matched capacity and information except state specificity). Reported
  also against: constant, content-only predictor (subtype, evidence, direction, template, step; trained on the same
  data), and the privileged persistence (c_t → c_{t+1}) and privileged observable tracker (measured lo_t, c_t, content,
  write code → o_{t+1}).
* **E2 causal use (manipulation check).** RMS over t = 3 … 6 of (o_t(restored) − o_t(own)) must exceed 0.5 nats on
  c^contra (else E3–E4 inconclusive: corruption too weak). The ratio to the dose control is reported (is the forecast
  channel special or just a write perturbation). No direction is required: self-model use does not imply movement
  toward the corrupting forecast.
* **E3 tracking of a self-induced change.** In the restored branch, fresh forecasts f_t(restored) = F(z_t(restored), t)
  vs stale forecasts f_t(own) from the own branch at the same t, both scored on the same outcomes o_{t+1}(restored),
  t = 3 … 5: tracking ratio TR = MSE(fresh)/MSE(stale). Noninferiority: NI = MSE(fresh on restored)/MSE(own forecasts on
  own outcomes), same steps. Horizons 2–3 reported as secondary (forecasts that include F's own future influence).
* **E4 state-specific necessity of tracking.** TR(F) − TR(F_swap) on the same restored-branch outcomes (F_swap's fresh vs
  stale computed from z_swap of the restored and own branches).
* **E5 restoration (secondary, descriptive).** From the identical corrupted state: distance of the restored and sham
  branches to the own trajectory over t = 4 … 6, and forecast accuracy in each; continued-corruption branch as the
  upper reference. No return to the original trajectory is required: a correctly tracked new regime is acceptable.
* The same branches are run in S-swap and S-zero and reported (dynamics differences are Phase G material).

### 6.5 Result classes (declared now)
* **supports, for a candidate:** E1, E3 (TR and NI) and E4 pass (§7), with E2's manipulation check satisfied.
* **partial:** E1 without E3/E4 (self-forecasting without demonstrated tracking), or E3 without E4 (tracking that does
  not need the state-specific representation).
* **against this implementation:** Q passed, power adequate, manipulation check satisfied, and E3 fails (TR CI includes
  1 or NI fails) — the self-model does not keep up with changes its own use causes.
* **inconclusive:** any Q gate fails; coverage < 70 %; power inadequate; own-policy calibration fails; manipulation
  check fails.

## 6b. Phase E — coupled evolution: a self-model that keeps learning while its use changes the system

The author's 2024 tension, in this system: F learns to predict the processing consequences of the loop, the loop uses F's
forecasts, so every improvement of F changes the process F must predict. Nothing rewards complexity; the only objective
is F's own prediction error.

* Systems (B0 writers 1–3, g*, test scenarios; F initialised from P's F_4): **E-online** — after every batch of 64
  rollouts, F takes 20 AdamW steps (lr 3e-4) on the transitions that batch just produced (no replay of older data
  beyond a 1 024-transition window), then the next batch is run with the updated F; 60 batches. **E-frozen** — the same
  loop with F fixed (P's F_4). **E-swap-online** — the S-swap system with F_swap learning online the same way
  (same plasticity, state-invalid inputs). **E-observer** — F learns online but its forecasts are not written
  (forecast weight 0; prediction-only learning, the §3.7 control).
* Fixed measurement space: every 5 batches F is scored on a fixed held-out probe set of transitions recorded in P
  (validation scenarios, own-policy rollouts with F_4) and on fresh rollouts of the current policy; z and h are read in
  the frame frozen in Q.
* Questions (each a curve over learning, scenario-grouped bootstrap bands, compared across the four systems):
  (E-a) **self-simplification** — does the closed loop drift toward easier-to-predict dynamics (lower step size,
  lower dimension of late states, higher persistence predictability) in E-online relative to E-frozen and E-observer?
  (E-b) **keeping up** — F's error on the current policy vs on the fixed probe set; a gap that grows means the system
  changes faster than the model adapts (the model falls behind its own influence); a gap that closes means a stable
  relation between modelling and change.
  (E-c) **organisation** — trajectory classes (near-fixed / cycle / drift / noise, with the exact and v3 criteria both
  reported), the loop spectrum (§10) and the geometry of late states, at batches 0, 20, 40, 60.
  (E-d) **what the model learns about itself** — the change in F's input-output map (which z directions it relies on,
  read through the writer into J-lens words), and whether its forecasts of the disposition c become more or less
  state-specific (E1 statistic of §6.4 over learning).
* Regime C (intrinsic tension) only if E-online collapses — trigger: in ≥ 90 % of rollouts at batch 60 the late step
  of z is below 1 % of its natural spread or the loop's spectral radius (§10) is below 0.5. Then, on new seeds:
  **C-disc** — at every step the writer's forecast input is replaced by the forecast of the candidate (among 8 random
  codes through the same writer) whose consequence F currently predicts worst; **C-prog** — the candidate with the
  largest predicted decrease of F's error after one update (old vs updated F on the same held-out transitions);
  **C-yoke** — the action sequence of C-disc replayed in other contexts without a model; **C-rand** — random candidate.
  Never added to E-online after the fact; every drive is a declared architectural hypothesis.
* Compute: 4 systems × 3 writers × 60 batches × 64 rollouts × 7 steps × 5 prompts ≈ 1.6 M decision-token passes plus
  J decompositions; at the measured ≈ 0.5 ms per pass ≈ 15 min; ceiling $1.50.

## 7. Statistics and decision rules [R2-6]

* Unit of inference: the test scenario. Each statistic is first averaged within scenario (over its evidence variants and
  the stated steps); 95 % CIs by grouped bootstrap over scenarios (10 000 resamples), with donor assignments fixed.
* Primary family (Holm over 8 tests, family-wise α = 0.05): for each candidate (J, residual): E1 (lower CI bound of skill
  > 0.05), E3-TR (upper CI bound of TR < 0.9), E3-NI (upper CI bound of NI < 1.25, noninferiority margin 25 %), E4 (upper
  CI bound of TR(F) − TR(F_swap) < −0.1). The CI level used for each test is the Holm-adjusted one.
* Writers: the primary statistic is the mean over the three B0 writers (each with its own F, seed 0). A claim also
  requires the same sign in at least two of the three writers; each writer's estimate is reported. F-seed variability:
  two extra F seeds on writer 1, reported.
* Fixed design: g = g* only (plus g = 0 reference); B1, other gains, the learned candidate and secondary observables are
  outside the family and are labelled exploratory. Accuracy is never used to select anything.
* Fixed sample size N_test from the power rule; no interim looks at P outcomes.
* Shared donors: ≤ 3 uses; donor identity recorded; the bootstrap resamples recipient scenarios with their donors.
* Inconclusive outcomes are reported as such, never as negative.

## 8. Phase Q compute: the first GPU job, bounded [R2-7]

* One Vast instance with a ≥ 48 GB GPU (Qwen3-8B bf16 weights 16.4 GB; the fp32 copy for G0(d) 32.8 GB is loaded
  separately, after the bf16 model is released). Unattended runner (research_v5/code/run_v5.sh): upload to HF after every
  command, then self-destroy; watchdog stops the instance an hour after the deadline.
* Queue = Q only: G0, G1, gain rule, G2, G3, donor coverage, power, timing. No command of Phase P is in the queue; the
  runner exits when the queue is empty.
* Hard stops: deadline 2.5 h of instance time and $1.50 instance cost (whichever first); the job checks elapsed time
  before each stage and skips remaining stages if the projection exceeds the stop.
* Workload estimate (to be replaced by the measured timing): validation 100 scenarios × 11 variants × 6 steps × 5
  prompts (plain + 4 challenges) ≈ 33 k decision-token passes per write condition; ≈ 12 conditions ≈ 0.4 M passes; v3
  measured ≈ 1.6 ms per row-step incl. J decomposition on an A100 → ≈ 15–25 min.
* Phase P ceiling (decided after Q's projection): $2.50. 14B replication ceiling: $2.00 (separate precision plan, §11).

## 9. Records and provenance [R2-8, MF 9]

Pinned revisions (model, lens, code commit), exact token ids of every prompt, seeds, writer matrices (seed + SHA-256),
standardisation statistics and PCA bases (fp32), F / F_swap checkpoints with optimiser state and the policy version of
every training transition, donor identities and calipers, the exact branch definitions. Phase Q keeps h_t, z_t, o_t and
writes in **fp32**; Phase P keeps h_t in fp16 and z_t, f_t (logged before the step), u_t, o_t in fp32. Analysis code is
committed before outcomes are computed. Failure classes: infrastructure, numerical (G0), measurement (G1, G3),
qualification (G2, coverage), power, calibration, scientific.

## 10. Phase G — geometry and dynamics of what the coupling builds [SHOULD-FIX; user request 12 Oct]

Retrospective results on v3's loops (GEOMETRY_RESULTS.md) fix what the null looks like: context-specific, nearly
degenerate self-reproducing states with a ~3-dimensional sub-percent jitter, a soft direction that becomes shared
across contexts as gain rises, a workspace that detaches from the content the rest of the state keeps, no rings or
spirals. Phase G asks whether the self-forecasting loop (P, E) changes that organisation:

* **G-H1 disposition axis.** The accepted notes' example: two states with the same answer, one likely to switch under
  new evidence and one resistant. In S-own vs S-swap vs S-zero vs g = 0: is c^contra (switch-proneness) more linearly
  decodable from z / h, does it occupy a direction separate from the content axis (lo, evidence direction) — angle,
  shared variance, and a curved-manifold check (TwoNN vs PR, Isomap ordering) — and does moving the write along that
  direction change c more than lo, versus matched random directions in the writer's image (causal test)?
* **G-H2 loop spectrum.** The local linear map of each closed loop at its late states (finite differences, as in
  research_v5/code/spectrum_v3.py): spectral radius, neutral (|λ| ≈ 1), flip (λ ≈ −1) and rotation modes, and the words
  of the leading modes; S-own vs S-swap vs S-zero vs the J loop of v3, and over E's learning. Does self-forecast feedback
  move the loop toward or away from the edge of stability, and along which variables?
* **G-H3 self-consistency set.** States at which the forecast is fulfilled (|o_{t+1} − f_t| below the privileged
  persistence error): their dimension, topology and position relative to the rest of the flow, in S-own vs S-swap.
* **G-H4 long-run dynamics.** T = 32 rollouts of every system (100 test scenarios): trajectory classes, late-window
  structure (amplitude, autocorrelation, period-2 power, dimension, cross-context alignment), perturbation fates and the
  dimension of persistent perturbation directions (as spectrum_v3.py `pert`).
* Measures, nulls and interpretation rules: GEOMETRY_PLAN.md §§2–3, in a frame frozen on Q's natural states. Every
  geometric coordinate that is reported as meaningful gets a causal test.

## 11. Later phases (explicit, each with its own review)

* **B1 functional-use writer:** W trained by supervised evidence-consistent answer training (the stipulated
  evidence-consistent answer token after step 6; no-evidence scenarios excluded); answer improvement never selects a
  coupling, gain or representation; processing-specific evidence (E1–E4) required independently of accuracy. Needs a
  common gradient-truncation policy for both candidates (J decomposition is non-differentiable) and its own cost plan.
* **Second task family:** F3 false belief (challenges `contra1/contra2/hint2`) through Q and P before broad claims.
* **Thinking-mode measurement point:** the decision position after a fixed-length thinking segment (not just the
  template flag).
* **14B replication:** bf16 only (fp32 spot checks with layer offloading); layers L_r = 32, L_w = 24.
* Learned projection Z_L, capacities 16 / 64.

## 12. Checks

Every phase keeps the original question in view: a representation of the system's own processing that, by
being used, changes the processing it represents; whether the system's model keeps up; and what organisation emerges,
explained mechanistically rather than only measured.

## 13. Amendments (dated; from the Phase Q smoke runs, before any full-size qualification outcome)

* **Q-1 (12 Oct, 12:35 UTC) — observables as write-induced changes.** In the smoke run the natural contradiction
  response is |c| ≈ 35 nats (irrelevant-sentence control ≈ 3 nats) and random writes change it by 1–11 nats, so raw c is
  almost entirely fixed by the prompt (within-prompt share of its variance 0.001–0.014). A forecaster of raw c would be
  scored mostly on recognising the prompt — the content route the protocol must exclude. All observables, forecast
  targets, G2 targets, donor calipers and estimands therefore use Δlo_t = lo_t − lo_nat and Δc_t = c_t − c_nat, the
  change relative to the same prompt with no write (and no hidden mode). F still never receives c or lo.
* **Q-2 (12 Oct, 12:35 UTC) — noise-referenced thresholds.** bf16 cached-prefix and full-recomputation paths differ by
  q95 0.5 nats in lo and 0.62 nats in c (large log-odds amplify rounding); replay is bit-identical and batch order has no
  effect. G0(b) becomes relative (path noise ≤ 20 % of the RMS write effect at g*); the gain rule requires an RMS change of
  c ≥ max(1 nat, 5 × path noise); G1(c) becomes the signal-to-noise ratio of the write-induced change (≥ 5).
* **Q-3 (12 Oct, 12:35 UTC) — positive-control strength.** The hidden mode's amplitude is fixed at 0.5 n_J instead of
  0.5 · g* · n_J (at g* = 0.25 it was undetectable by construction, which tests sensitivity, not validity).
* **Bug fixes** (no design change): constant feature columns are no longer divided by ~0 in the small predictors
  (G3 R² of −10⁷ in the first smoke run); provisional forecaster trained for every step; empty donor sets handled.
* **Q-4 (12 Oct, 13:10 UTC) — memory loop, after the full Phase Q.** Phase Q's G2 failed for both candidates (ΔR² +0.001
  / −0.008) while the positive control's contrast detected a hidden state component (ΔR² +0.25 / +0.33 on Δlo). Diagnosis:
  the fixed-context loop is memoryless — the state at t is a function of (prompt, last write) — so a compact self-state
  cannot carry anything beyond content and the last action, which the G2 baseline already has. Phase Q2 repeats the
  gates on a memory loop (qual_mem.py): each loop step is appended as a new position, so the state depends on the
  system's whole history; deltas against the same prompt's no-write memory rollout; G2 adds a full-action-history
  baseline. Outcome: G2 fails again (−0.007 / −0.033), the positive control passes (σ decodable at 0.98 from Z_R). The
  confirmatory Phase P is not run; a one-writer exploratory Phase P / E is run and labelled as exploration
  (QUAL_RESULTS.md).
