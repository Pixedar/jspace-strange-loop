# Research v3 — frozen protocol (written 2026-10-11, before any v3 data existed)

This protocol implements the author's programme (the original proposal) as amended by the improvement notes and the
user's own comments. Where the two documents conflict, the amendment wins: endogenous tension is allowed as a separately
declared arm; Experiment 1 tests open questions rather than re-establishing published J-space properties. Everything
not declared here as confirmatory is exploratory and will be labeled so in the report.

Budget: $11.22 Vast credit at start (account 182089). Ceiling for this program: $8.50; per phase caps below.

## 0. What is already known (do not re-test)

From the J-space paper (transformer-circuits.pub/2026/workspace, the paper's models only):
* Jacobian lens: `J_l = E[d h_final,t' / d h_l,t]`, averaged over source positions t, all t' >= t, ~1,000 prompts.
  Lens vectors are rows of `W_U J_l`; the J-space is the set of sparse non-negative combinations of k <= 25 of them
  (gradient pursuit). The J-component is <= 10 % of activation variance.
* Established in the paper's models: silent intermediate reasoning, deliberate bring-into-workspace, J-dependence of flexible
  multi-hop reasoning (ablation) but not of fluency/classification, preferential MLP/attention composition with J
  directions ("broadcast"), capacity of tens of concepts, a workspace band at ~38-92 % depth.
* The authors' own cautions: the readout is "a bag of ideas, not a sentence"; "most of the model's computation runs
  outside the J-space"; the decomposition is not unique; no recurrence in a standard transformer.
* No Qwen result appears in the paper. Neuronpedia publishes lenses for Qwen3 1.7B-32B fitted with the paper's jlens
  code (wikitext, 128-token prompts, ~460 prompts, bf16). No lens exists for Huginn or Ouro.
* Recurrent models (arXiv 2609.01924, Wang & Reid): Huginn carries workspace content across all recurrences but reads,
  writes and ablations act only within a ~2-recurrence window; Ouro rebuilds content every loop. No lens files released.
* This repository (Parts 1-4 and the self-workspace study, see HANDOFF.md): on Qwen3 the lens is conceptual only past
  ~0.6 depth up to 8B; placeholder-token rings carry no task information; ring v1 "surprise" was a cosine-magnitude
  artifact; finite action menus offered no state-specific headroom; J long loops switch while residual loops settle.

## 1. Substrate and measurement (shared)

* Main model `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218`, bf16 weights, SDPA, hook arithmetic in
  fp32. Replication `Qwen/Qwen3-14B` @ `40c069824f4251a91eefaf281ebe4c544efd3e18` for selected measurements.
  Natural recurrence: `tomg-group-umd/huginn-0125` @ `bb6621b6...` (identical weights to the `nikvisel` mirror).
  Historical control Qwen3-4B @ `1cfa9a72...` only for the Phase-0 reanalysis.
* Lens `neuronpedia/jacobian-lens` @ `9fa267f8...`, file `qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt`
  (sha256 recorded at load). Dictionary atoms `a_l(w) = J_l^T (gamma * (u_w - mean u))`, unit norm, over common English
  words (zipf >= 3) — the full-token dictionary halves self-reading on Qwen (Part 2); the full dictionary is a
  robustness check in E1. Residuals are centred by a per-layer mean computed on a calibration split and frozen.
* Decomposition: greedy pursuit with least squares, final non-negative coefficients, K = 25 primary; K in
  {8, 16, 32, 64} for capacity comparisons. Saved per state: residual (fp16), atom ids, coefficients, reconstruction
  norm, remainder norm, next-token log-probs of candidate answers and the top-20 tokens.
* Layers: calibration scan at 45/60/70/82 % depth (8B: blocks 16, 22, 25, 29; 14B: 18, 24, 28, 33).
* Prompt format: Qwen3 chat template, `enable_thinking=False` for all primary measurements (empty think block); the
  decision position is the last template token, the answer is the first assistant token. Candidate answers are scored
  as logsumexp over surface variants (case / leading space) of each candidate; outcome = log-odds A vs B.
* Thinking mode is a separate, explicitly labeled variable measured on a matched subset (E1-T), never pooled.
* Numerical qualification: identical rows in one batch must be bit-identical; cross-batch-geometry noise on the
  log-odds is measured and every causal effect is reported against 10x that floor. Compared conditions of one item
  always run in one batch with identical padding.

## 2. Task families (programmatic, ground truth explicit)

Splits are by source scenario AND by template: each family has >= 6 surface templates, 2 reserved for test; filler
pools (names, objects, colors, containers) are partly disjoint between train and test. All variants of one scenario stay
in one split. Pilot: 40 scenarios per family. Main: 300 train / 100 val / 200 test scenarios per family (the test set is
generated once from a fixed seed and never enlarged after results are seen).

* F1 Binding (H1). Same token multiset, different bindings: attribute (object-color), role (giver-receiver), spatial
  (left-right), possession (person-item). Several word orders per relation so that "most recent word" is not the label.
* F2 Interpretation under evidence (H2, Exp 2). Ambiguous core sentence (pronoun reference, PP-attachment, competing
  causes) kept fixed; evidence variants: none, weak-direct, strong-direct, hearsay, unreliable source, conflicting-
  then-resolved, each favoring A or B.
* F3 Belief / perspective (H1, H2, Exp 2). ToMi-style object moves with false/true belief; questions about reality vs a
  character's belief; evidence variants about what the character could see.
* F5 Chains (Exp 2 benchmark track). Variable/relation chains of 2-4 hops with distractors, single-token answers.
  (F4 goal-switching from the original proposal is dropped: it tests flexible access, already shown in the paper's models.)

## 3. Experiment 1 — open questions about the workspace

### H1 Structured workspace: is binding carried beyond the bag of active concepts?
Position: end of scenario, before any question (bindings must be held, no answer is imminent); layers as above.
1. Binding index (RSA). For each scenario, states A (box=red, table=blue) and B (swapped) in several word orders.
   BI = [mean d(A_i, B_j) - mean d(A_i, A_j)] / mean d(A_i, A_j) over cross-template pairs i != j, cosine distance.
   Representations: residual, J reconstruction, J code (vocabulary-indexed coefficients), bag (binary atom set,
   Jaccard), remainder (residual - J reconstruction), PCA-k and random-projection-k (k = 25, 64). Null: label
   permutation within scenario. BI > 0 = binding encoded beyond word order.
2. Model-as-decoder transplant. Write the representation from A's scenario end into a neutral query prompt
   ("What color was the box?") at the same layer at a placeholder position; score the answer. Controls: bag-only code
   (equal coefficients), shuffled coefficients, remainder only, random norm-matched, unrelated donor. Diagnostic:
   accuracy split by whether the queried item was mentioned first or second (a recency shortcut scores only one side).
   Qualification: the full-residual transplant must reach >= 65 % accuracy, else the transplant arm is not qualified.
3. In-context patch at the scenario end (J component vs remainder vs full residual from A into B). Qualification: full
   residual flips >= 30 % of answers, otherwise binding is retrieved from token positions and the single-position
   workspace cannot carry it (a substantive result for the loop design).

### H2 Processing dispositions: does the workspace carry how the model would respond, beyond its current answer?
States: F2/F3 scenario x evidence variant; representation at evidence end E and at the unchallenged decision position.
Dispositions: change of log-odds under 5 text challenges inserted after the evidence (direct contradiction, weak
counter-hint, authority counter, doubt, irrelevant distractor) and 2 internal challenges (patch alpha = 0.5, 1 of the
opposite-evidence state's J component at the decision position).
* Qualification (on training data, pre-declared): within-scenario (state-specific) variance >= 20 % of total
  disposition variance for at least one challenge, and output-only features leave >= 30 % of it unexplained.
  If not met the family is reported as unsuitable for H2; no larger predictor is trained to compensate.
* Primary test: within-scenario-centred ridge prediction of dispositions from J (K = 25) beyond output features
  (log-odds, candidate entropy, top-1 probability, full entropy); partial R^2 on held-out templates with grouped
  bootstrap CI; Holm over the 4 layers x 2 positions. Comparisons at matched capacity: PCA-k, random-k, residual,
  remainder, K in {8,16,32,64}. Generalization: leave-one-challenge-out with a bilinear state x challenge model.

### E1-T Thinking mode (matched subset, descriptive)
60 F2/F3 items generated with thinking on (<= 384 tokens); J codes at the end of thinking vs the non-thinking decision
position; accuracy per mode.

## 4. Experiment 2 — the J-space loop (H3)

Loop map at the decision position D of a running context c (one step = recompute position D from the write layer up,
prefix KV fixed):  h_Lw <- b_c + g * write(code_t);  code_{t+1} = decompose(h_Lr).  Default L_r = 25 (0.7 depth),
L_w = 16 (0.45 depth), chosen before data from Part 2/ring history and re-checked in the E1 calibration scan.
* Concept transport (primary): the code (which concepts, how strongly) is written with layer-L_w atoms of the same
  words. Equal-footing transport for every representation (controls): output-equivalent pull-back
  u = argmin ||J_Lw u - J_Lr v||^2 + lambda ||u||^2. Write norms are matched across conditions (g in units of the
  median natural J-reconstruction norm at L_w). Transport fidelity (read-after-write cosine) is reported.
* Memory test: initial code read from an evidence context (A or B); the loop then runs in the neutral context with the
  evidence removed. The loop is the only carrier of the evidence. Persistence = log-odds(A-init) - log-odds(B-init).

Regime A (unforced): J loop g in {0, 0.5, 1, 2, 4}; controls at matched norm: dense residual loop, PCA-25 loop,
random-atom loop, shuffled-coefficient (bag kept) loop, foreign code (other trajectory, same step), time-shuffled own
code. T = 64 steps.
Regime B (self-model with endogenous action, no tension): a shared online self-model M (per-action ridge with
forgetting 0.98) predicts the next compact state z (PCA-32 of the J reconstruction) for 9 internal actions (none,
amplify / suppress one of the top-4 current concepts, write norm renormalized). B chooses the action with the lowest
expected self-prediction error. Predictions are logged before the outcome.
Regime C (intrinsic tension, separate arm): C1 highest expected error; C2 highest learning progress (error decrease
over a recent window). Controls: random actions with M learning (prediction-only), yoked actions copied from C1 of
another context, frozen M with the C1 rule. B/C run at g in {1, 2}, 3 seeds.
Benchmark track: answer accuracy at the decision position across steps for every condition (F2/F3/F5).

Pre-declared trajectory classes (late window, steps 48-63): fixed point (max 1 - cos between consecutive J
reconstructions < 0.01); short cycle (period p <= 4 with cos(z_t, z_{t-p}) > 0.99 throughout); otherwise aperiodic,
labeled structured if a ridge model fitted on other contexts predicts late z_{t+1} from (z_t, context code) with
R^2 >= 0.5, else noise-like. Task-information retention = held-out probe accuracy for the interpretation from z_t.

## 5. Experiment 3 — emergent organization (analysis of Exp 2 + Huginn; exploratory unless stated)

* Dynamics: fixed points, cycles, metastability, perturbation return (attractor claims need return from perturbed
  states), comparison against linear/AR models and phase-randomized surrogates.
* Perturbations at step 16 of twin trajectories: semantic (swap the A-concept for the B-concept), state transplant
  (opposite-evidence code), matched random, self-model intervention (swap M's predictions between actions).
  Outcomes: decays / persists numerically without meaning / persistent interpretable change.
* Geometry: organization of visited codes across contexts (clusters, transitions, cycles), concept co-activation
  structure, how geometry differs across regimes and representations; functional role tested by moving states.
* Meaning: which words dominate and why (trace to context, lens priors, ordinary generation baseline).
* Self-model co-evolution: M error vs system change over time (converge / diverge / oscillate / stabilize).
* Phase 0 (historical): mechanism of the 64-pass J switching in the Qwen3-4B self-workspace data — decision-margin
  crossings, atom-support changes at switches, open-loop fixed-support and soft-mixture counterfactuals.

## 6. Huginn natural recurrence
Readout: coda exit lens (apply coda + norm + head to the state after recurrence k); validated by answer emergence
over k before use. Measurements on F2/F3/F5: accuracy vs recurrences r in {1,2,4,8,16,32,64}; the same memory test
(recurrent state initialized from the evidence context, recurrence continued with the neutral context's input
injection); state swaps between A/B contexts at k; perturbation dynamics.

## 7. Statistics
The independent unit is the source scenario. Grouped bootstrap (2,000 resamples), paired comparisons, Holm within
declared families, effect sizes with 95 % CI, detection thresholds stated for null results. Exploratory analyses are
labeled; nothing is tuned on the test split.

## 8. Phase cost caps
E1 (8B) $1.5; Exp 2 (8B) $2.5; 14B replication $1.5; Huginn $1.5; reserve $2.7. Every instance self-uploads to the
private HF dataset `Pixedar/jspace-v3` after each stage and destroys itself at the end of its queue (stops if the
upload failed); a watchdog stops it one hour after its deadline.

## 9. Amendments after the pilot (2026-10-11 ~21:00 local, before any main-scale data)

The pilot (40 scenarios per family, Qwen3-8B) was used, as declared, only to validate measurement and intervention
design. Changes:
1. Numerics. Replay of identical batches is exact, but the same prompt in a batch of different size moves the log-odds
   by up to 1.0 (bf16 kernel selection), against a median |log-odds| of 10. All E1 forward passes now use a fixed
   batch shape per stage (fixed padded length and batch size, dummy rows), which makes each row's result independent of
   its batch companions; this is verified in the qual stage.
2. H1. The scenario-end transplant and the scenario-end in-context patch did nothing even for the full residual at every
   layer (transplant 50 % = chance; patch 95.5 % vs 95.6 % unpatched). Per section 3 the transplant arm is not qualified
   and binding is evidently read from earlier token positions. Added causal test: "context patch" from the swapped-
   binding prompt into the original at every position whose token is identical in both prompts (after the first
   differing token, inside the scenario) — these positions differ only through context, i.e. through binding. Full /
   J component / remainder / random-J swaps, at each layer. Positive control: decision-position patch (answer content).
3. E1-T: 80 % of thinking traces were unfinished at 384 tokens; budget raised to 1,024.
4. E2 gain unit. With g in units of the natural J norm at the write layer (1.2 vs 18.8 at the read layer in the smoke
   model) every write was ~3 % of the stream and changed log-odds by ~0.01, so no regime could appear. New unit: the
   median J-reconstruction norm at the READ layer (g = 1 re-broadcasts the workspace at the strength it was read); J
   sweep g in {0.25, 0.5, 1, 2, 4}. Controls, B/C and perturbations run at two gains chosen by a rule fixed now: g1 =
   smallest grid gain whose final code shares < 80 % of atoms with the natural code (mean over std contexts), g2 = next.

## 10. Amendments after the E2 pilot (2026-10-11 ~21:10 local, before main E2 data)

1. Bounded broadcast. With writes proportional to the read code's magnitude the loop is a positive-feedback amplifier:
   by step 64 the write norm reached ~1.7e3 (g=1), ~4.8e5 (g=2) and inf (g=4) — a trivial divergence. Every write now
   has norm exactly g * m_J (m_J = median natural J-reconstruction norm at the read layer); the loop carries content
   (which concepts, in which proportions), not amplitude. This is an architectural constraint comparable to a
   normalization layer, not a complexity objective, and applies identically to all representations and controls.
2. Same numerical path. Initial codes were computed by a full-prompt forward while steps use the KV-cache single-token
   path; bf16 path differences alone changed ~17 % of greedily selected atoms. Initial codes now use the step path.
3. Loop layers. Chosen by a channel-qualification rule on VALIDATION contexts (12 scenarios per family), before any
   test-context data: candidates (Lr, Lw) in {(25,16), (29,22), (29,16), (25,22)}; pick the pair with the largest
   single-write memory effect at g = 1, i.e. mean over F2/F3 scenarios of |lo(N <- code_A) - lo(N <- code_B)| at step
   1 of the J loop; ties within 10 % go to the lower write layer.
4. B/C is run with the same bounded broadcast; the pilot B/C (unbounded rule) is exploratory only.
5. (added before the layer sweep ran) F6 concept-memory contexts: "{intro} The secret {category} is {word}." vs
   "... is hidden." — the remembered item is a verbalizable concept, in contrast with F2/F3 where pilot H1 showed the
   answer-determining variable at the decision position lies in the non-J remainder (decision patch at layer 25:
   full residual flips ~29 % of answers, remainder ~28 %, J component ~0 %). The layer rule's memory metric is
   averaged over F2, F3 and F6 scenarios; F6 is reported separately in every E2 result.

## 11. Exploratory arm X — the system's own linear self-forecast as the self-model (added 19:35 UTC, before any X data)

Motivation: the author's original tension was computed from the system itself, with no external algorithm. The J lens is
the model's own average Jacobian from the read layer to the output, so its readout at the read layer is the workspace's
linear forecast of the network's output; pilot H1 showed that at the decision position the network's answer is mostly
carried outside the J component. Arm X asks whether feeding back what the network does but its workspace does not
forecast lets the workspace come to carry the decision. Conditions (same loop, same bounded broadcast, gains g1/g2):
* inc — broadcast the 25 dictionary words with the largest log p_out(w) - log p_lens(w) (softmax-weighted);
* out — broadcast the network's 25 most probable dictionary words (plain output feedback: the ordinary explanation);
* lens — broadcast the forecast's 25 most probable words;
* J and no-feedback references. Same memory (mem) and standard (std) contexts as Regime A. Exploratory; reported
  separately and never pooled with the pre-declared regimes.

## 12. Ouro replication added (19:50 UTC, before any Ouro data except a 2-scenario smoke test)

Budget allowed the proposal's second recurrent substrate. ByteDance/Ouro-2.6B @ 1ed04250 (bf16): the same contexts and
the same tests as Huginn — exit-head log-odds per loop (loops 1–4 trained, 5–8 out of distribution), decision-state
transplant from A or B into N at the end of loop 1/2/3, full-state swap A <-> B at the same boundaries, 1 % perturbation
of the decision state at the end of loop 2. Readout: Ouro's own per-loop exit (final norm + LM head), no J lens.

## 13. Exploratory epistemic-concept test (added 20:20 UTC, after seeing the decision-position lens examples only)

The main lens-scan examples show epistemic-status words in the decision-position workspace ("Both", "unanimous",
"either", "unknown", "ambiguous", "conflicting", "insufficient", "Neither") and, at layer 29, words sharing the answer's
first letters. Because the proposal requires a causal test for every claimed semantic feature, stage `epi` steers the
decision state of held-out F2/F3 states along the unit mean of a fixed epistemic word list (25 words, listed in
e1.py) at layers 25/29 with alpha in {-1, -0.5, 0.5, 1} x natural J norm, against random word sets of the same size
and norm, and measures the change in |answer log-odds|. The descriptive loading by evidence type uses the saved H2
codes. Exploratory: the word list was chosen after seeing the examples, so results need confirmation on fresh data.

## 14. Full-token dictionary robustness check implemented (20:35 UTC)

Section 1 promised the full dictionary as a robustness check. Stage `h1full` repeats the decision-position patch at
layers 25/29 with a J dictionary over every non-special token (151k atoms), which contains the capitalised no-space
answer tokens absent from the English-word dictionary. Reported next to the primary H1 result.

## 15. Exploratory strong-coupling self-model regimes (added 22:35 UTC, before any B/C main data was analysed)

The pre-declared gain rule selected g1 = 0.25 and g2 = 0.5, where a broadcast changes the workspace only modestly
(final overlap with the natural code 0.79 / 0.60). Stage `BCs` repeats regimes B, C1, C2, R, F and Y unchanged (standard contexts only, for budget) at
g = 1 and 2 (overlap 0.33 / 0.13), where the loop dominates the decision-position workspace. Exploratory; reported
separately from the pre-declared B/C results.

## 16. Exploratory 14B self-model replication (added 02:35 UTC Oct 12, before any 14B B/C data)

Stage `BC1` runs regimes B, C1, C2, R, F and Y unchanged on Qwen3-14B at its rule-selected gains (0.5 / 1.0), loop
layers 32 / 24, standard contexts of the 14B E2 run, one seed (budget; replicate systems = 32-context chunks, 14 per
policy and gain). Analysed with the same `analyse_bc` contrasts as the 8B regimes. Reported with the 14B replication,
never pooled with the 8B data.

## 17. Exploratory content-route control for H2 (added 02:05 UTC Oct 12, after the main H2 analysis)

`an_h2_label.py`: the H2 ridge analysis repeated with the experimenter's evidence label (type x direction, and its
interaction with the ambiguity subtype; one-hot, centred within scenario) added to the output-only baseline. Same
splits, eligibility, alphas and grouped bootstrap as the primary analysis.
