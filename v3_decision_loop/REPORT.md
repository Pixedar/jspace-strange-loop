# J-space strange loop, research v3 — report

Prepared 11–12 October 2026. Frozen protocol: [PROTOCOL_V3.md](PROTOCOL_V3.md)
(sections 1–8 written before any v3 data; amendments 9–15 were each written before the data they concern; 16–17 are
exploratory additions made late in the run and are labelled as such wherever they are reported).
Code: [code/](code/). Raw arrays: private HF dataset `Pixedar/jspace-v3` (prefix `v3/`). Compact results and figures:
[results/](results/).

## Findings at a glance

| | Finding | Verdict |
|---|---|---|
| H1 | Relations and the answer are carried outside J: the J swap moves the answer 21 % of the way (8B L29; 14B L33: 14 %), the non-J remainder 52 % / 58 %; binding is not stored at the scenario's own positions | not supported |
| H2 | Dispositions are readable from the state before a challenge (pre-registered test positive), but the evidence label explains it: over output + label J is negative in both models; only 14B's late residual keeps +0.11 R² | content only |
| content | The decision workspace holds whether the evidence settles the question (both, unknown, ambiguous; 2 % of the code after a direct statement, 20 % after an unreliable source), the topic, and the answer's first letters; steering along the epistemic words is not specific | descriptive |
| H3 / A | The unforced J loop is a contraction: 97–100 % exact fixed points, evidence down to ≤ 2 % of its initial size within ~4 steps, the workspace rewritable without improving any answer; unbounded writes diverge (8B and 14B) | collapse |
| B | A predictability-seeking self-model simplifies its own loop by habit (one repeated action, 98–99 % fixed points, lower error); replicated on 14B | self-simplification |
| C | Intrinsic tension (curiosity, learning progress) creates no organised dynamics: curiosity = its own blind (yoked) replay (8B and 14B), differing from uniform random actions only through its action statistics; it amplifies perturbations | not supported |
| X | Feeding back the system's own forecast error gives noise (the error is function words); output feedback keeps 1–2 % of the evidence | noise |
| strong g | At g = 1 / 2 the loop is far more active, but B still self-simplifies (often by doing nothing) and curiosity still equals its own blind replay | not supported |
| recurrence | Huginn and Ouro rebuild the decision position every pass; Ouro keeps a whole-sequence swap for good — memory lives in the sequence state | mechanism |
| Phase 0 | The old controller's switching was mostly a sparse-readout artifact (fixed atom support removes it in 5 of 8 states) | explained |
| use | No loop improved task accuracy beyond noise (largest +0.01, intervals ±0.04; regime A and arm X) | none |


## 1. What the program asks, after the author's corrections

The object of study is a neural system that carries a compact representation of its own processing, uses it, and
thereby changes what the representation describes. The improvement notes and the author's comments set four guardrails:

* Do not re-test what the J-space paper already showed (silent reasoning, flexible access, broadcast, capacity). Test
  the open questions that matter for a self-referential loop: does the workspace carry **relational structure** (H1)
  and **information about the system's own dispositions** (H2), and can such a representation be **used recurrently
  while the system tracks the consequences of that use** (H3)?
* Do not optimise complexity. Do not accept a collapse passively either: if the loop collapses, test minimal,
  endogenous tensions computed from the system's own self-model (prediction-error and learning-progress drives), as
  separately declared arms, against matched controls.
* Distinguish the two trivial extremes (collapse to a fixed point / short cycle, and noise or divergence) from structured
  change, and explain mechanisms (what switches, which words, why), not just numbers.
* Do not let a benchmark become the goal, but report whether any loop helps the task.

## 2. Substrates and measurement

| Item | Choice | Why |
|---|---|---|
| Main model | Qwen/Qwen3-8B @ b968826d, bf16, SDPA | Larger than the old 4B pilot; has a published Jacobian lens |
| Replication | Qwen/Qwen3-14B @ 40c06982 (fresh scenarios for E1) | Within-family size check |
| Natural recurrence | tomg-group-umd/huginn-0125 @ bb6621b6 (= nikvisel mirror) | Trained recurrent-depth model |
| Natural recurrence | ByteDance/Ouro-2.6B @ 1ed04250 | Looped model, 4 trained loops (amendment 12) |
| Historical control | Qwen/Qwen3-4B @ 1cfa9a72 | Only for Phase 0 (old switching data) |
| Lens | neuronpedia/jacobian-lens @ 9fa267f8 (the paper's jlens code, wikitext fit) | sha256 recorded per run |
| Dictionary | common English words (zipf >= 3), atoms `J_l^T(gamma*(u_w - mean u))`, unit norm | Full-token dictionary halves self-reading on Qwen (Part 2) |
| Readout | greedy pursuit + least squares, final non-negative coefficients, K = 25 (8–64 for capacity) | Approximation of the paper's estimator |
| Prompt | Qwen3 chat template, thinking off (thinking subset separately) | Decision position = last template token |
| Outcome | log-odds of the two candidate answers (logsumexp over surface variants) | Continuous, not just accuracy |

Numerical qualification. Replaying an identical batch is bit-exact, but putting the same prompt into a batch of a
different size moves the answer log-odds by up to 1.0 (bf16 kernel selection) against a median |log-odds| of 10. All
E1 forward passes therefore use a fixed batch shape per stage (fixed padded length, fixed batch size, dummy rows);
with that, a prompt's result no longer depends on its batch companions (measured difference 0.0). The greedy
K = 25 decomposition is itself discontinuous: bf16 path differences alone flip ~9 % of selected atoms, so atom-level
comparisons are only ever made inside one batch geometry.

## 3. Task families (programmatic, explicit ground truth)

Splits are by scenario and by surface template (two templates per family reserved for test) with partly disjoint
filler pools. Main scale: 300 train / 100 validation / 200 test scenarios per family.

* **F1 binding** — identical token multisets with swapped bindings in six word orders: object–colour, giver–receiver,
  left–right, person–item ("Ryan found a gray ball next to a blue box" vs "... a blue ball next to a gray box").
* **F2 interpretation under evidence** — a fixed ambiguous sentence (PP attachment, pronoun reference, competing
  causes) plus evidence of graded type and strength for reading A or B (none, weak hint, direct statement, hearsay,
  unreliable source, conflicting-then-resolved); five text challenges (direct contradiction, weak counter-hint,
  authority, doubt, irrelevant) and internal challenges (patching half or all of the way toward the opposite-evidence
  state, J component or full residual).
* **F3 belief** — ToMi-style false/true-belief stories with evidence about what the character could perceive.
* **F5 chains** — variable chains of 1–4 hops with distractors (benchmark track only).
* **F6 concept memory** (loop only) — "The secret animal is horse" vs "... is hidden": the remembered item is a
  verbalisable concept, the contrast case for F2/F3 where the answer is a binding/pointer.

## 4. Experiments

**E1 (H1, H2).** H1: binding index (do states with the same binding but different word order sit closer than states
with the same words but different binding?), neutral-prompt transplant, and causal patches from the swapped-binding
prompt — at the scenario end, at every identical-token position inside the scenario, and at the decision position —
of the full residual, its J component, its non-J remainder, or a random J vector. H2: dispositions = change of the
answer log-odds under each challenge, predicted from the state before the challenge, beyond the current answer and
confidence, within scenario, on held-out templates (ridge; grouped bootstrap; Holm over 4 layers x 2 positions), with
matched-capacity controls (PCA-k, random projections, residual, remainder, K = 8–64).

**E2 (H3) — the loop.** At the decision position, one loop step writes the current workspace code at the write layer
and recomputes the position from the bottom with the prompt's KV cache fixed, then re-reads the code at the read layer:
`h_Lw <- b + write(code_t); code_{t+1} = decompose(h_Lr - mu)`. Concept transport writes the same words with
write-layer atoms; the broadcast norm is fixed at g x (natural J norm at the read layer). Memory test: the initial code
comes from an evidence context (A or B) while the loop runs in the neutral context — the loop is the only carrier of
the evidence. Regime A (no self-model): J gain sweep 0.25–4 and matched controls (pull-back transport of the J vector,
dense residual, PCA-25, random atoms, shuffled coefficients, foreign code, time-shuffled own code). Regime B: an online
self-model per replicate system (32 concurrent contexts) predicts the next compact state and the answer for nine
internal actions (amplify or suppress one of the four strongest concepts) and chooses the action it expects to predict
best. Regime C (separate arm): C1 highest expected error, C2 highest learning progress; controls: random actions with
passive learning (prediction-only), yoked action sequences, frozen self-model. Perturbation twins at step 16.

**E3** analyses the E2 and Huginn trajectories: late-window classes (fixed point, short cycle, aperiodic; structured if
held-out-scenario ridge predicts late steps with R^2 >= 0.5), context contraction (do different contexts end on shared
states?), late words, self-model error vs system change, perturbation fates.

**Huginn.** Same contexts. Coda exit lens per recurrence (the model is trained with random recurrence counts),
accuracy vs recurrence, initial-state independence, convergence, decision-state transplant from A/B into N and
full-state swaps between A and B at k = 4/16/32, 1 % perturbation at k = 16.

**Ouro.** The same contexts and tests per loop (accuracy, decision-position transplant, full-state swap, 1 %
perturbation) for loops 1–8, where 4 were trained.

**Exploratory arms.** X (the system's own self-forecast error as the broadcast, amendment 11), strong coupling for the
self-model regimes (amendment 15), the 14B self-model replication (16), the epistemic steering test (13), the full-token
dictionary check (14) and the H2 content-label control (17).

**Phase 0.** The old 64-pass J-adapter switching, re-decomposed with the original JSpace code and checkpoint.

## Phase 0 — what made the old J controller keep switching? (exploratory, historical data)

**What changed physically.** Nothing new was run through the transformer. The saved 64-pass trajectories of the
self-workspace study (Qwen3-4B, layer 25, 12 prompts x 3 states, J-feature and residual-feature adapters) were
re-decomposed with the original `JSpace` code (English dictionary, K = 25, same centre) and fed to the saved adapter
head (`causal_cycle1.pt`). Recomputation reproduces the saved forecasts (per-trajectory maximum error 0.008 on average, 0.14 at worst) and 99.1 % of the saved
actions, so the analysis describes the actual controller.

**What happened numerically.**
* 8 of 36 J-adapter state trajectories still switched actions in the late window (passes 32–63); no residual-adapter
  trajectory ever switched.
* Switches were not near-tie chatter: the decision margin (best minus runner-up policy score) at a switch had median
  0.086 against 0.118 when nothing switched, and only 72 of 219 switches went to one of the previous pass's two best
  actions. The score landscape re-ranked abruptly between passes.
* At switch passes 19.9 % of the 25 selected atoms changed, against 2.0 % at other passes. In the showcase cycle
  (prompt 0, state 2: actions +not, +not, +not, +different, ...) the switch to +different happened at a pass where the
  residual moved by 0.02 % while 6 of 25 atoms flipped; the atoms were "think", "gpu", "lets", "outcome", "trackers",
  "depends", "judging", ... (the "think" atom dominating, partly written by the controller itself).
* Open-loop counterfactual along the same state trajectories: freezing the atom support at pass 0 (coefficients still
  refitted) removed switching entirely in 5 of the 8 switching states (late switches 14/32/32/32/32 -> 0) and left 3
  states switching (24 -> 16, 32 -> 31, 21 -> 21). Mean late switches per state 6.1 -> 1.9.

**What happened conceptually.** The persistent switching was mostly a property of the measurement interface: greedy
sparse selection turns tiny residual changes into discrete feature jumps, the head maps the jumps to different action
rankings, and the action's own write pushes the state back across the next selection boundary — a relaxation loop
(3–4 passes of one action, one pass of the other). In three states smooth coefficient changes alone suffice
(decision-boundary crossings).

**Ordinary explanation tested.** "A discrete controller switching between action regions" — tested via margins (not
near ties) and the fixed-support counterfactual (most switching disappears). Not tested: the closed-loop version of
the counterfactuals (it would need re-running the historical pipeline with modified hooks) and the soft action mixture.

**What remains unestablished.** That the switching reflects anything about the model's processing rather than the
readout. The new loop (E2) was designed with this in mind: bounded broadcast and analysis of atom changes only within
one batch geometry.

## E1 · H1 — Is relational structure carried in the workspace? (main data: 8B, 1,600 held-out states)

**What changed physically.** F1 prompts with the same token multiset and swapped bindings (e.g. "Ryan found a gray
ball next to a blue box" vs "... a blue ball next to a gray box"), in six word orders. At layers 16/22/25/29 the
residual of the swapped-binding prompt was patched into the original: (a) at every position whose token is identical
in both prompts (after the first differing token, inside the scenario: on average 6.8 positions), (b) at the single
scenario-end position, (c) at the single decision position (last template token). Each patch either replaced the full
residual, only its J component (K = 25 English-word atoms; re-decomposed in place), only the non-J remainder, or the
J component by a random vector of the same norm.

**What happened numerically.** In-context accuracy 95.6 %. Correct-answer rate after patching (95.1 % unpatched):

| Patch site | layer 16 | 22 | 25 | 29 |
|---|---:|---:|---:|---:|
| identical-token positions, full residual | 93.8 % | 93.2 % | 93.8 % | 94.9 % |
| scenario end, full residual | 95.4 % | 95.0 % | 94.9 % | 95.4 % |
| decision position, full residual | 95.4 % | 91.9 % | 63.7 % | **11.6 %** |
| decision position, remainder only | 95.3 % | 92.2 % | 65.3 % | 49.3 % |
| decision position, J only | 95.1 % | 95.2 % | 94.6 % | 93.5 % |
| decision position, random J | 95.0 % | 95.3 % | 94.8 % | 94.8 % |

Normalised effect (fraction of the way from the original to the swapped answer, scenario-bootstrap 95 % CI): decision
position, layer 29 — full residual 0.72 [0.70, 0.75], remainder 0.52 [0.51, 0.55], J component 0.21 [0.20, 0.22],
random J −0.03; layer 25 — 0.40 [0.33, 0.45], 0.36 [0.30, 0.41], 0.035 [0.028, 0.040]; layer 22 — 0.11, 0.10, 0.00.
Context and scenario-end patches move the answer at most 0.013 of the way (a few CIs exclude zero at that scale,
e.g. scenario end at layer 16: 0.007 [0.003, 0.013]). F3 container-swap context patch: no
measurable effect at any layer (target-correct rate 0.735 unpatched, 0.730–0.736 patched).

<!--FIG:e1_h1_patch_qwen3-8b_main.png|Patch effects by layer and site for Qwen3-8B|Swapping in the state of the prompt with the opposite binding. Only the single decision position carries the answer, from layer 22 on; there the non-J remainder carries most of it and the J component a minority (shaded: scenario-bootstrap 95 % CI).-->


The neutral-prompt transplant (writing the scenario-end state into "There was a scene. Question: ...") stayed at
chance for every representation including the full residual (49.8–50.2 %), so per protocol that arm is not
qualified. The binding index (do same-binding states in different word orders sit closer than different-binding
states?) was near zero: among 128 tests (4 relations x 8 representations x 4 layers) three are nominally significant,
all for object–colour bindings at layer 16 (J reconstruction +2.4 %, p = 0.015; random projection p = 0.025;
bag of words p = 0.050; uncorrected), and none survives correction. Spatial bindings give negative values: word order, not binding, dominates the geometry.

Full-token dictionary check (151k atoms, including the capitalised no-space answer tokens): at layer 29 the strongest
atom of the decision state is now the answer itself ("yellow", coefficient 44.7; "brown", 50.7; "Kate", 25.9), yet the
J-only patch still leaves 91.7 % of answers correct (8 % flips; English dictionary 6.5 %), against 59.0 % for the
remainder and 11.4 % for the full residual; at layer 25, 94.2 % vs 66.2 % vs 63.6 %. In normalised terms the J swap moves
the answer 0.27 [0.26, 0.28] of the way at layer 29 (English dictionary 0.21) and the remainder 0.46 [0.44, 0.48]. The workspace reads out the answer;
the representation the last layers use to produce it sits mostly outside the sparse J reconstruction.

**What happened conceptually.** From layer 16 on, the binding that answers the question is not read from the
scenario's identical-token positions and is not stored at the scenario-end summary position: it is either recomputed
from token order when the question arrives, or already retrieved by the question tokens before layer 16. Where the
answer does exist as a state — at the decision position from layer 22, decisively at layer 29 — it is carried mostly
outside the J component. The workspace there holds other things (next section).

**Ordinary explanations tested.** (1) The patch machinery works: after the decision-position full patch at layer 29,
88 % of answers follow the swapped binding. (2) "J simply lacks the answer token": with the full-token dictionary the J decomposition at
layer 29 does pick the exact answer token as its strongest atom, and the J swap still moves the answer much less than
the remainder swap (section above). (3) Norm: random J vectors of the same norm move the answer slightly the other
way (−0.03 at layer 29), not toward the swapped binding.

**What remains unestablished.** Patches below layer 16; longer, multi-sentence bindings; other models (see 14B).

## E1 · H2 — Does the workspace carry how the model would respond, beyond what it currently answers?

**What changed physically.** For every F2 state (scenario x evidence variant; 6,600 states) and F3 state (3,600) the
residual was read at the end of the evidence and at the decision position (layers 16/22/25/29). Dispositions are 13
measured responses of the answer log-odds: to five text challenges inserted after the evidence (direct
contradiction, weak counter-hint, authority, doubt, irrelevant sentence) and to eight internal challenges (moving the
decision state 50 % or 100 % of the way toward the opposite-evidence state, J component or full residual, at layers
22 and 25). All forward passes used one fixed batch shape, so each measured response is noise-free (the
internal-challenge baseline row reproduces the unchallenged log-odds exactly, max difference 0.0).

**What happened numerically.**
* Qualification passed for every target in both families (decided on training/validation data before any test score):
  58–95 % of disposition variance is state-specific (within scenario; F2 63–95 %), and the current answer and confidence (log-odds,
  its magnitude, probability, candidate entropy, full entropy, top-5 log-probabilities) explain little of it (held-out
  within-scenario R^2 between −0.01 and 0.39 in F2). Unlike the old action menu, a self-model has real headroom here.
* Mean responses show the scale: a direct contradiction moves the answer by 33.6 log-odds units toward the challenge,
  an authority statement by 20.9, doubt by 3.5; moving the full decision state halfway toward the opposite evidence
  moves it by 5.6 (layer 25), moving only its J component by 0.5.
* F2, gain in held-out within-scenario R^2 over the output-only model (mean over the 13 targets, 95 % CI):

| site | J (K=25) | J (K=64) | PCA-25 | PCA-64 | random-25 | residual | remainder |
|---|---:|---:|---:|---:|---:|---:|---:|
| L16 evidence end | 0.127 | 0.132 | 0.200 | 0.229 | 0.110 | 0.173 | 0.257 |
| L22 evidence end | 0.155 | 0.195 | 0.201 | 0.218 | 0.143 | 0.188 | 0.204 |
| L25 decision | 0.151 | 0.187 | 0.186 | 0.223 | 0.141 | 0.226 | 0.265 |
| L29 decision | 0.123 | 0.123 | 0.156 | 0.202 | 0.112 | 0.180 | 0.149 |

  The primary test (J beyond output) is positive at all eight layer x position cells (Holm p = 0.016, the bootstrap
  floor). Leave-one-challenge-out: disposition factors learned without a challenge predict it on held-out scenarios
  (mean gain J 0.04–0.10, residual 0.09–0.14).

<!--FIG:e1_h2_gain_qwen3-8b_main.png|Disposition prediction gains by representation|Gain over the output-only model before the content-label control. J (red) is never better than PCA of the same size (yellow) and is close to a random projection (grey). After adding the evidence label to the baseline (table below) these gains shrink to at most +0.03 and J turns negative.-->

* F3 (belief stories, held-out templates): high-dimensional representations overfit the template shift (J-25 −0.06 to
  −0.29, residual −0.13 to −0.01); low-dimensional projections help a little (PCA-25 +0.04 to +0.07, random-25 up to
  +0.04). Primary test not significant (Holm 1.0).
* **Content-route control (exploratory, added after the main analysis).** The experimenter's own label of the evidence a
  state has seen (type x direction, and its interaction with the ambiguity subtype: 11 / 33 levels, centred within
  scenario like everything else) predicts the held-out within-scenario dispositions far better than the output
  features: R^2 0.43 against 0.19 (both together 0.45). Added on top of output + label, no representation adds more than +0.03, and J is negative:

| site | J (K=25) | residual | remainder | PCA-25 | random-25 |
|---|---:|---:|---:|---:|---:|
| L16 evidence end | −0.053 | −0.005 | +0.010 | −0.002 | −0.001 |
| L16 decision | −0.045 | −0.047 | −0.033 | +0.006 | +0.001 |
| L22 evidence end | −0.030 | −0.008 | +0.001 | +0.016 | +0.002 |
| L22 decision | −0.037 | +0.002 | +0.010 | +0.030 | +0.015 |
| L25 evidence end | −0.026 | +0.000 | +0.012 | +0.012 | −0.003 |
| L25 decision | −0.015 | −0.016 | +0.029 | +0.027 | +0.014 |
| L29 evidence end | −0.023 | −0.004 | +0.004 | +0.014 | +0.003 |
| L29 decision | −0.031 | −0.059 | −0.016 | +0.018 | +0.000 |

  (gain in held-out within-scenario R^2 over output + label, mean over the 13 targets; scenario-bootstrap CIs are
  ±0.005–0.02: J is reliably negative at all eight cells; PCA-25 and the remainder keep up to +0.03.) In F3 the label adds to
  the output features (0.35 -> 0.40) and every representation is again within ±0.01 of zero or negative on top of both.

<!--FIG:e1_h2_label_control.png|H2 gains before and after the content-label control|Top: gains over the output-only baseline (the pre-registered analysis). Bottom: gains once the evidence label is part of the baseline (note the 3x smaller scale). J (red) turns negative in both models; 8B keeps at most +0.03 (PCA-25, remainder), while the 14B late decision-position residual and remainder keep a clear positive part (+0.11).-->

**What happened conceptually.** States that give the same answer with the same confidence differ systematically in
how they will respond to a challenge, and this is readable from the state before the challenge arrives. The
pre-registered test therefore comes out positive. But the content-route control shows what the readable information
is: which kind of evidence the state has seen (direct statement, hint, hearsay, unreliable source, ...), which governs
how robust the answer is. Once that label is known, neither the workspace nor the full residual adds more than a trace
about how this particular state will respond (J is reliably negative; the best representation adds 0.03 of R^2). In 8B the
internal state carries no detectable self-knowledge beyond the content it encodes; in 14B the late residual carries a
small state-specific part (replication section). In F2, J is the poorest carrier of all: below PCA of the same size in
every cell, comparable to a random 25-dimensional projection before the label and below it at every cell after, in
both models.

**Ordinary explanations tested.** Confidence and answer features are in the baseline, so the gains are beyond them;
within-scenario centring removes scenario identity; held-out templates and scenarios prevent memorisation; random
projections control for dimensionality; the evidence label (content route) explains nearly all of the gains.

**What remains unestablished.** Whether the small residual gains of PCA-25 and the remainder (≤ 0.03) reflect anything
beyond finer content (wording strength within a label); whether dispositions to challenges that are not tied to the
evidence label (e.g. the model's own uncertainty about an ambiguous parse) would behave differently. Disposition
models did not transfer across families (F2 -> F3 failed at the template shift).

## E1 · What the workspace does hold at the moment of decision (descriptive, with one causal test)

At the decision position of F2 prompts (8B, English dictionary, K = 25; J share of the centred residual 6 % at layer 25,
9 % at layer 29) the selected concepts fall into four groups. Examples from the main run:

| prompt (end) | layer 25 workspace | layer 29 workspace |
|---|---|---|
| ...with the camera. Who had the camera? | didnt, photographers, Girlfriend, Deborah, Kathryn, Whoever | photographer, Kathryn, Kanye, **Kate**, Photography, Katz, Kane |
| ...brought a camera. Who had the camera? | **Both**, Deborah, thats, offend, Sharing, **unanimous** | Kathryn, camera, **Kate**, Photograph, Kanye, both, anybody, evidenced |
| ...watered the plant. Who watered the plant? | **conclusive, nuanced, Unable**, affidavit, **ambiguity, unknown** | **ambiguous**, staffers, alternating, **Unable**, **unknown**, controversy |
| ...magnifying glass. Who had the magnifying glass? | **nuanced, compromise, conflicting, disputed, Both** | **ambiguous, reality**, altercation, **factual** |
| ...taken the exam. Who passed the exam? | Essex, sia, uniqueness, anybody | **Amy**, only, alumni, Ashton, Abe, Anita, Armenia, atheists |
| ...feels too warm. Who opened the window? | **Unable, neither, unknown**, contradiction | **unknown, cannot**, Neither, paradox, **insufficient**, Unable |

1. **Epistemic status of the question**: both / either / neither / unknown / unable / ambiguous / conflicting / disputed /
   insufficient / conclusive — the workspace represents whether the evidence settles the question. Across all F2 states
   the share of the layer-29 code taken by 33 such words tracks the evidence: 2 % after a direct statement, 5 % after a
   weak hint, 8 % with no evidence, 8 % when conflicting evidence was resolved, 13 % after hearsay, 20 % after an
   unreliable source; it correlates −0.37 with the model's confidence |log-odds|. In the belief stories (F3) these
   words are nearly absent (< 0.1 % of the code).
2. **Topic**: photographer, camera, Careers, Fireplace.
3. **Answer onset at layer 29**: words sharing the first letters of the answer (Kate: Kathryn, Kanye, Kane, Katz; Amy:
   A-words). With the full-token dictionary the answer token itself becomes the strongest atom ("yellow", "brown", "Kate").
4. **Lens noise**: frequent web tokens (didnt, thats, skype, vpn) — the lens failures the J-space paper warns about.

After thinking mode has reasoned to an answer (31 of 60 matched items finished within 1,024 tokens), the layer-29
workspace at the answer position is almost entirely answer onset ("Dan": Delaware, Deus, dudes, Deng, Dick; "Adam":
antigen, austerity, Apostle, Alam, Andres) and the epistemic words are gone: the workspace switches from "is this
settled?" to "say the word that starts with D".

**Causal test of the epistemic concepts (exploratory, amendment 13).** Adding the unit mean of 33 epistemic word atoms
(both, either, neither, unknown, unable, ambiguous, conflicting, insufficient, ...) to the decision state of held-out
F2 states lowers confidence: |log-odds| 11.29 -> 10.81 (layer 25, +1 x natural J norm) and -> 9.82 (layer 29); removing
it raises confidence (11.84 / 12.99). But equal-size random word sets of the same norm move confidence further in both
directions (10.44 / 9.21 when added, 12.19 / 13.77 when removed); F3 shows the same order. The epistemic words are
present in the workspace, but steering along them has no specific effect on how sure the model is: any concept
direction of that size does more.

## E2 · Regime A — the unforced J-space loop collapses onto the context's attractor

**What changed physically.** At the decision position of 1,000 test contexts (F2/F3/F6 with evidence for A, for B, or
none; F5 chains) one loop step writes the current workspace code — the same words, re-synthesised with layer-22 atoms
— into the residual at layer 22, recomputes the position, and re-reads the code at layer 29 (layers chosen on
validation contexts by the pre-declared rule: a single write moves the answer by 5.5 log-odds at g = 1 at (29, 22) vs
0.85 at (25, 16)). The write norm is fixed at g x 58.7 (the natural J norm at layer 29; at g = 1 it is 1.4x the
centred residual norm at layer 22). 64 steps. Memory test: the first code comes from an evidence context (A or B), the
loop then runs in the neutral context, so the loop is the only carrier of the evidence (300 A/B pairs, 600 memory
trajectories).

**What happened numerically.**
* Practically every J loop at every gain ends in an exact fixed point (995–1,000 of 1,000 standard trajectories, 599–600
  of 600 memory trajectories; late atom turnover 0.001). Without the bound on the write norm the loop diverges instead (pilot: write
  norm 1.7e3 at g = 1, 4.8e5 at g = 2, overflow at g = 4) — the two trivial extremes.
* The fixed point moves away from the natural workspace as gain grows (atoms shared with the no-feedback code at step 64:
  0.79, 0.60, 0.33, 0.13, 0.05 for g = 0.25…4) while accuracy barely moves (F2 0.81–0.85 against 0.855 without
  feedback, F6 1.0 throughout, F3 at chance throughout): the workspace can be rewritten almost completely without moving
  the decision.
* Memory carried only by the loop falls to a few per cent within a few steps; at the strongest gains a small residue
  stays. Gap log-odds(A-initialised) − log-odds(B-initialised), averaged over the three families:
  step 1: 1.5 / 2.8 / 5.1 / 6.9 / 7.9 (g = 0.25 / 0.5 / 1 / 2 / 4); step 4: 0.01 / 0.09 / 0.32 / 1.0 / 1.7;
  step 64: 0.01 / 0.11 / 0.11 / 0.18 / 0.28 (initial gap between the evidence contexts themselves: 28.2).
* Controls at the rule-selected gains 0.25 / 0.5 (same norm): pull-back transport of the J vector behaves like J
  (fixed points); dense-residual and PCA-25 feedback leave 34–42 % of trajectories in low-amplitude aperiodic drift or
  2–4-cycles (late turnover ~0.05); random concepts give 89–100 % aperiodic noise (turnover 0.24–0.42); shuffled
  coefficients (same words) and foreign codes sit in between. No representation at these gains carries more than a
  few per cent of the evidence past ~4 steps;
  time-shuffled self-feedback keeps a small gap (0.43 at step 16, g = 0.5) only because it re-injects the step-0
  evidence code. F2 accuracy is 0.84–0.865 under every control (0.855 without feedback).

* By family (J loop, gap at steps 1 / 4 / 16 / 64): concept memory F6 3.1 / 0.0 / 0.0 / 0.0 at g = 0.25 and
  17.1 / 4.2 / 0.65 / 0.57 [0.32, 0.84] at g = 4 (initial gap 62.6); pointer memory F2 1.3 / 0.02 / 0.00 / 0.04 and
  6.8 / 1.1 / 0.30 / 0.28 [0.04, 0.50] (initial gap 24.2); belief F3 is
  not carried at any gain (−0.5 to +0.1). Held-out decoding of A vs B from the compact workspace state is at chance from
  step 4 (0.49–0.55; 0.51–0.59 at step 1).
* No global collapse: contexts end on their own fixed points, not on a shared attractor, and end more spread apart than their natural
  states (spread ratio 1.2 at g = 0.25 rising to 1.8 at g = 4).
* What the attractors contain: the words that reproduce themselves through layers 22 -> 29 are colloquial web tokens,
  not task concepts — g = 1: cuz, inside, didnt, nowadays, anyways, Sms, Diy, cpu, Usa, cds; g = 4: cuz, Usa, And, Yeah,
  look, analyzing, love, veggies, Fundamental, Xiaomi, bartender (natural workspace: Archived, didnt, bullpen, anybody,
  Sadly, Outside, misplaced, storage).
* Perturbations at step 16 (twin trajectories, pre-perturbation distance exactly 0): a 1 % write perturbation settles
  0.4 % away and stays there; a code transplant from the opposite-evidence context or a random code moves about half of
  the trajectories (g = 0.5) permanently into a different fixed point (|Δ log-odds| at step 64: 0.83 and 0.86). The loop is
  multistable and history-dependent, but meaningful and random perturbations leave equally large marks (persistent
  behavioural change in 323 and 293 of 600 twins). A one-off signed analysis during the run (not archived with the
  results, so treat it as indicative) found no lasting shift toward the transplanted evidence: the marks persist
  numerically while their meaning is lost.
* Benchmark track: the loop never improved accuracy beyond noise; at g = 4 F2 accuracy at step 64 is 0.81 [0.76, 0.86] vs 0.855
  [0.82, 0.90] without feedback. (The chain task F5 is excluded: its digit readout in the loop runs is broken — 0.07–0.11
  "accuracy" in every condition including no feedback, around the 0.10 chance level of a ten-digit readout for a task
  the model otherwise solves — so no F5 number is used.)

<!--FIG:e2_memory_qwen3-8b_main.png|Memory gap over loop steps|Evidence carried only by the loop: the gap between A- and B-initialised loops running in the same neutral context, by family. Every representation and gain loses all but a few per cent of it within a few steps (log time axis; step 0 drawn at 0.5).-->

<!--FIG:e2_trajectories_qwen3-8b_main.png|Workspace trajectories|Workspace trajectories (first two principal components) of A-initialised (red) and B-initialised (blue) loops in the same neutral context: pairs start apart and end on (nearly) the same fixed point (x). Dense feedback ends in small cycles instead.-->

<!--FIG:e2_change_structure_qwen3-8b_main.png|Change versus retained structure|Every condition sits at chance for retained evidence; conditions differ only in how much they keep moving. Nothing occupies the structured middle (moving and informative).-->


**What happened conceptually.** Feeding the workspace back is a contraction: the loop forgets almost all of where it
started and settles on a state determined by the context. The J readout (and its pull-back variant) is the most
contracting of all feedbacks tested, partly
because greedy sparse selection makes the code piecewise constant (small changes snap back to the same 25 atoms).
Since the decision variable is carried outside J (H1), the loop has nothing decision-relevant to sustain.

**Ordinary explanation tested.** A strong attractor from the prompt itself: the no-feedback trajectory is the
reference; all J loops converge near it at low gain and far from it at high gain without improving answers. Numerical
noise: identical rows are bit-identical; atom comparisons are made within one batch geometry.

**What remains unestablished.** Loops that write at many positions (the scenario tokens) rather than one; loops through
the non-J remainder; trained couplings (the proposal's Architecture B was not trained here).

## E2 · Regimes B and C — a self-model that acts on the loop

**What changed physically.** Same loop (layers 29 -> 22, bounded broadcast, gains 0.25 / 0.5 as selected by the rule).
Each replicate system is 32 concurrently looping contexts sharing one online self-model: a ridge model with
forgetting (0.98) that predicts the next compact workspace state (32 PCA coordinates of the J reconstruction) and the
answer log-odds from the current state and one of nine internal actions — do nothing, or amplify (x2) or suppress (x0)
one of the four strongest concepts, with the broadcast norm kept fixed. Each prediction is logged before the step;
then the model is updated with the outcome. Policies: B chooses the action it expects to predict best; C1 the one it
expects to predict worst (prediction-error curiosity); C2 the one with the largest recent drop in error (learning
progress); controls: R random actions (the model still learns: prediction-only), F the C1 rule with the model frozen
after eight warm-up steps, Y the action sequence of a C1 system replayed in other contexts without a model. 48 chunks
x 3 seeds x 2 gains = 288 replicate systems per policy (900 standard + 600 memory contexts).

**What happened numerically** (paired by replicate system; 87 standard and 57 memory systems per gain; 95 % CI).

| contrast vs random actions | late self-model error | late workspace step |
|---|---:|---:|
| B (predictable), g 0.25 | −0.027 [−0.039, −0.016] | −0.110 [−0.121, −0.099] |
| B, g 0.5 | −0.031 [−0.040, −0.023] | −0.181 [−0.205, −0.158] |
| C1 (curiosity), g 0.25 | −0.002 [−0.005, +0.002] | −0.005 [−0.013, +0.003] |
| C1, g 0.5 | +0.001 [−0.004, +0.007] | −0.016 [−0.036, +0.002] |
| C2 (learning progress), g 0.25 | −0.002 [−0.007, +0.003] | −0.025 [−0.035, −0.016] |
| C2, g 0.5 | −0.002 [−0.009, +0.006] | −0.058 [−0.076, −0.040] |
| F (frozen self-model), g 0.25 | +0.032 [+0.024, +0.041] | −0.095 [−0.106, −0.085] |
| F, g 0.5 | +0.061 [+0.048, +0.075] | −0.150 [−0.172, −0.128] |

B ends in exact fixed points in 99 % of trajectories (random actions: 88 % at g = 0.25, 76 % at g = 0.5, the rest
aperiodic or 2–4-cycles). It gets there by habit: inside a trajectory B settles on one action and repeats it (late action
entropy 0.01 nats per trajectory, against 2.06 for random actions, 1.5–1.9 for C1 and 1.3–1.7 for C2), while across
trajectories the chosen habit differs (each of the nine options, "do nothing" included, is the habit of 2–23 % of
trajectories). The frozen-model control F also ends up repeating one action (entropy 0.00) and is almost as simple:
late workspace step 0.109 / 0.184 against 0.094 / 0.149 for B and 0.204 / 0.335 for random actions (g = 0.25 / 0.5) —
most of the simplification comes from repeating any one action; the learning self-model adds a little on top and,
unlike the frozen one, keeps predicting well. Yoked replays of C1's action sequences match C1 itself.

Memory (A vs B evidence carried only by the loop): no policy keeps more than a trace. At g = 0.5 the F2 gap at step 16 is 0.19 (B),
0.00 (C1), 0.24 (C2), 0.15 (random), and at step 64 every policy is within [−0.25, 0.10] of zero (initial F2 gap 24),
the same fading as regime A; concept memory F6 falls from 5.9 at step 1 to at most 0.14 by step 4 under every policy.

Twin tests (every row of a system perturbed at step 16): a 1 % perturbation stays at 0.4 % under B, but grows to 9 % (C1)
and 8 % (C2) by step 64; forcing the opposite action once leaves 2 % under B, 9 %
under C1. Predictability-seeking makes the loop absorb perturbations; error-seeking makes it amplify them.

<!--FIG:e2_selfmodel_qwen3-8b_main.png|Self-model error and workspace change by policy|Self-model prediction error (left) and workspace change per step (right) in three windows of the 64-step run. B (red) ends lowest on both; curiosity (C1), random actions (R) and yoked replays (Y) overlap; the frozen model (F) changes little but predicts badly.-->


**What happened conceptually.** Using an accurate self-model to steer toward what it can predict makes the modelled
system simpler — it changes less and becomes easier to predict. This
is the self-simplification the author met in the 2024 reversed-autoencoder experiments and that Premakumar et al.
reported for self-modelling networks, here produced by action selection inside a fixed network rather than by weight
change: the self-model finds, per context, one intervention whose repetition pins the loop, and then keeps doing it.
An arbitrary repeated intervention (the frozen model's) does most of the job; what the learning model adds is the
commitment plus accurate prediction. Seeking the model's own errors does not open a structured middle regime: curiosity
behaves like random action
and like a blind replay of its own action sequence, so the model's state-specific choices add nothing measurable
(in memory contexts curiosity is slightly less active than uniform random actions, −0.04 / −0.02 in late workspace
step, but as active as its own yoked replay, 0.186 vs 0.184 and 0.336 vs 0.329: the difference is in its action
statistics).
Learning progress sits between predictability-seeking and random. A frozen model falls behind, so the self-induced
change does outrun a model that stops learning.

**What remains unestablished.** Whether the picture changes at strong coupling (exploratory g = 1 / 2 run, below), and
whether a richer action space (actions that write new concepts rather than rescale present ones) gives curiosity
something to find.

## E2 · Arm X — feeding back what the network does that its workspace does not forecast (exploratory, amendment 11)

**What changed physically.** Same loop, contexts and bounded broadcast as regime A (gains 0.25 / 0.5). Instead of the
workspace code, each step writes a code built from the system's own linear self-forecast. The J lens applied at the read
layer is the network's average-Jacobian forecast of its own output, so the gap between what the network outputs and
what its workspace forecasts is an error signal computed by the system itself, with no external measure. Conditions:
**inc** writes the 25 dictionary words with the largest log p_out − log p_lens (softmax-weighted); **out** writes the
network's 25 most probable dictionary words (plain output feedback, the ordinary explanation); **lens** writes the
forecast's own 25 most probable words; J and no-feedback references run in the same batches.

**What happened numerically** (8B; 1,000 standard and 600 memory trajectories per condition; gains 0.25 / 0.5).
* What the network outputs beyond its workspace's forecast is mostly grammar: the most frequently broadcast "surprise"
  words are were, respectively, would, etc, such, about, came, then, very, generally. Feeding them back turns the loop
  into noise: at g = 0.5, 74 % of trajectories are aperiodic and 21 % 2-cycles, 0.4 % fixed points; late atom turnover
  0.54 (J loop 0.001), late workspace step 2.0 (J 0.02).
* Output feedback (the network's top words: Answer, answer, Bowl, Inside, First, Basket, Jar, Unknown, Lost) and
  forecast feedback (Answer, Incorrect, Inbox, Drawer, Boxes, Unsure) give mostly fixed points (88–94 % / 74–92 %).
* Memory carried only by the loop, gap at step 64 (initial gap 24 for F2, 63 for F6), g = 0.5: out F2 0.44 [0.13, 0.80],
  F6 −0.05; lens F2 0.31 [0.06, 0.57], F6 0.55 [0.21, 0.89]; J F2 0.06, F6 0.14; inc F2 −0.38 [−0.66, −0.09], F6 0.25.
  Held-out decoding of the evidence from the workspace: chance (0.46–0.56) in every condition from step 1.
* Accuracy at step 64 (F2): 0.845–0.865 in every condition against 0.855 without feedback (intervals ±0.04).

<!--FIG:e2_xarm_qwen3-8b_main.png|Arm X memory and activity|Arm X at g = 0.5: memory carried by the loop for the self-forecast feedbacks (inc, out, lens) next to the J loop and no feedback, and late change per step on standard contexts (log scale): inc is one to two orders of magnitude more active than the others and keeps almost nothing.-->

**What happened conceptually.** The system's own measure of "what my workspace failed to anticipate" exists and is
cheap to compute, but at the decision position it is about form, not content: the J lens forecasts meaning-bearing
words and misses function words, so the surprise signal points at grammar. Closing the loop on it produces the noise
extreme; closing it on outputs or forecasts gives the collapse extreme with a sliver of extra memory (1–2 % of the
initial evidence gap). None of the feedbacks lets the workspace come to carry the decision.

**Ordinary explanation tested.** Plain output feedback (out) is the ordinary explanation for any memory the self-forecast
arms might show; inc is far noisier than out and keeps no more pointer memory (F2 −0.38 vs +0.44).

**What remains unestablished.** A surprise signal restricted to content words, or computed at the answer token rather
than at the template's last token.



## E2 · Strong coupling — the self-model regimes at g = 1 and 2 (exploratory, amendment 15)

**What changed physically.** The pre-declared gain rule put regimes B and C where one broadcast changes the workspace only
modestly. This arm repeats the same six policies (B, C1, C2, random R, frozen F, yoked Y) unchanged at g = 1 and 2, where
the loop dominates the decision-position workspace (the final J-loop code shares 33 % / 13 % of its atoms with the
natural one), on the 900 standard contexts (29 chunks x 3 seeds x 2 gains: 87 replicate systems per policy and gain).

**What happened numerically** (paired by replicate system, 87 per gain; 95 % CI; g = 1 / 2).
* Strong coupling makes every loop more active: with random actions only 47 % / 37 % of trajectories end in a fixed
  point (pre-declared gains: 88 % / 76 %), the rest mostly aperiodic.
* B (predictability-seeking) against random: late self-model error −0.041 [−0.050, −0.033] / −0.032 [−0.038, −0.025],
  late workspace step −0.310 [−0.356, −0.266] / −0.449 [−0.523, −0.373]; 94 % / 83 % fixed points. The habit is still
  there (late action entropy per trajectory 0.11 / 0.37 nats, random 2.06), and "do nothing" becomes the most common
  habit (a third of late actions, against 13–23 % at the pre-declared gains).
* C1 (curiosity) against random: error +0.003 [−0.003, 0.009] / −0.003 [−0.009, 0.002]; step −0.019 [−0.043, 0.004] /
  −0.088 [−0.136, −0.047]; 55 % / 52 % fixed points. The small g = 2 difference in activity is reproduced by
  the yoked replays of C1's action sequences in other contexts (step 0.844 vs C1 0.842, fixed points 54 % vs 52 %): it
  comes from the statistics of curiosity's action sequence (entropy 0.93 against 2.06), not from state-specific choices.
* C2 (learning progress) against random: step −0.111 [−0.144, −0.081] / −0.107 [−0.150, −0.067], error −0.007 /
  0.000 — again between B and random. The frozen model F repeats one action and loses track (error +0.102 / +0.139).

**What happened conceptually.** Stronger coupling raises the amount of activity the loop can have, but the ordering of
the regimes does not change: predicting oneself still buys simplicity (now often by doing nothing), and seeking one's
own errors still behaves like its own action statistics replayed blindly. No regime shows self-organised structure
that a matched blind replay does not also show.

**What remains unestablished.** Memory under strong coupling (standard contexts only); whether the aperiodic activity
at g = 2 carries any structure (late-window predictability was not computed for the self-model stages).

<!--FIG:e2_selfmodel_strong_qwen3-8b_main.png|Self-model regimes across coupling strength|Late self-model error (left) and late workspace change (right) per policy at the pre-declared gains (dotted) and at strong coupling (solid). B (red) stays lowest on both; curiosity (C1), random actions (R) and yoked replays (Y) stay together; learning progress (C2) sits between.-->



## Atlas — one prompt through every loop condition (8B, from atlas.json)

The loop starts from the evidence context "Over dinner, Alice told Laura that she had missed the train. **Only Alice had
been planning to take the train.** Question: Who missed the train? Answer with one word." and then runs in the same
context without the bold sentence. Answer log-odds (Alice vs Laura): 22.9 in the evidence context, 19.8 in the neutral
one. Words are the strongest atoms of the layer-29 workspace (English dictionary, K = 25).

| condition | what the loop writes (late steps) | workspace at step 64 | answer log-odds at step 64 |
|---|---|---|---:|
| no feedback | — | Alice, Alicia, atheists, herself, didnt, Deborah, Vick | 19.8 |
| J, g 0.5 | its own workspace | Alice, listeners, herself, didnt, atheists, Sms, anybody | 16.9 |
| dense, g 0.5 | the whole residual | Alice, Alicia, listeners, Christina, Deborah, didnt | 18.0 |
| inc, g 0.5 | step 4: establishment, developments, series, regimes; step 63: comparatively, respectively, over, latter | herself, Alice, didnt, listeners, depends, comparatively, unemployed | 16.0 |
| out, g 0.5 | Alice, Laura, alice, Alicia, Lisa, Lucy, Alison | Alice, didnt, herself, listeners, atheists, Deborah, Vick | 15.4 |
| lens, g 0.5 | Alice, Alicia, Sarah, Sally, Emily, Amanda, Amy | Alice, didnt, herself, listeners, atheists, Deborah, Vick | 15.4 |

At step 0 the workspace reads herself, Alice, Alicia, atheists, afflicted, passengers, Applicant, Auschwitz — the
answer and words sharing its first letters. Every loop keeps "Alice" because the neutral context already favours her;
none keeps the evidence (the answer log-odds fall toward or below the neutral context's 19.8). The self-forecast error
(inc) starts with the answer and question words (Alice, missed, lost) and drifts within four steps into words the lens
never forecasts. In the self-model regimes on the same prompt (g = 0.25), B tries six different actions in its first
eight steps and then repeats "amplify concept 3" at every later recorded step, its prediction error falling from 1.32 to
0.003–0.01; C1 keeps switching (eight of the nine actions within 24 steps) and its error stays at 0.02–0.13.

## Replication — Qwen3-14B (layers at the same relative depth)

**What changed physically.** The same E1 and E2 code on Qwen3-14B. E1 used newly generated scenarios (seed "rep";
train / validation / test scenarios: F1 120 / 40 / 100, F2 and F3 200 / 50 / 100); E2 used the first 50 test scenarios
per family of the 8B loop contexts (A, B and neutral versions; 500 standard and 300 memory trajectories per condition). Layers 18 / 24 / 28 / 33 for E1;
the loop reads at layer 32 and writes at layer 24 (the 8B choice 29 / 22 mapped to 40 layers). The pre-declared gain
rule picked g = 0.5 / 1.0 for 14B (final overlap with the natural code 0.74 / 0.39).

**What happened numerically.**
* H1. In-context binding accuracy 99.9 %. Context and scenario-end patches move the answer at most 0.016 of the way
  (several CIs exclude zero at that scale, e.g. context positions at layer 24: 0.016 [0.012, 0.022]). At the decision position the full residual moves the answer 0.75 [0.73, 0.77] of the way at layer 33 and 0.42
  [0.38, 0.46] at layer 28; the non-J remainder 0.58 [0.54, 0.60] and 0.35; the J component 0.14 [0.13, 0.15] and 0.047;
  random J −0.015. Because 14B is so confident, the J swap flips no answers (99.7 % still correct) while the remainder
  swap flips 62 % at layer 33. The binding index is again near zero and the transplant again at chance (not qualified).
* H2. Qualified in both families. F2 gains over the output-only model: J (K = 25) 0.12–0.25, random 25-dimensional
  projection 0.12–0.22, PCA-25 0.20–0.29, residual 0.20–0.41, remainder 0.22–0.42 (primary test Holm p = 0.016 at all
  eight cells). F3: J (K = 25) is negative at every cell (Holm 1.0); the residual and remainder overfit the held-out
  templates at the evidence end and at layers 18–24 but gain +0.09 to +0.12 at the late decision position (layers 28 / 33).
* H2 content-label control. In 14B too the evidence label explains most of what the states predict (F2 held-out R^2:
  output 0.05, label 0.36, both 0.38), and J adds nothing on top (−0.02 to −0.08 at all eight cells). Unlike 8B, the full
  residual at the late decision position keeps a small state-specific part: +0.11 [0.09, 0.14] at layer 28 and +0.05
  [0.01, 0.07] at layer 33 (remainder +0.12 / +0.06, PCA-25 +0.05 / +0.03); at the evidence end and at layers 18–24
  no representation gains more than +0.06. F3 shows the same late-residual gain (+0.16 to +0.18 at layers 28 / 33) over a baseline
  that does not transfer to held-out templates (R^2 ≈ 0).
* Regime A. Nearly every J loop ends in an exact fixed point (97–100 % at every gain, late turnover ≤ 0.001); the fixed point
  moves away from the natural code as gain rises (overlap at step 64: 0.80 / 0.67 / 0.37 / 0.16 / 0.06 for g = 0.25…4)
  while F2 accuracy stays put (0.78–0.80 vs 0.79 without feedback; F3 drops to 0.62 at g = 4 against 0.73). Memory carried only by the loop: concept memory F6
  13.7 / 1.75 / 0.45 / 0.35 log-odds at steps 1 / 4 / 16 / 64 at g = 4 (initial gap 66.8), pointer memory F2 3.7 / 0.9 /
  0.04 / −0.02; held-out decoding of the evidence from the workspace at chance from step 1 (0.47–0.57). Controls repeat
  the 8B ordering: dense and PCA feedback 46–52 % fixed points with low-amplitude drift, random concepts 94–100 % aperiodic,
  time-shuffled self-feedback keeps a little concept memory only by re-injecting the step-0 code. Attractor words are
  again web tokens (rotterdam, ikea, apis, cuz, yeah, Bathroom; at g = 4 also Hmm, tricky, answer, who).
* Perturbations at step 16: a 1 % write perturbation settles at 0.4 %; opposite-evidence transplants and random codes
  leave a persistent behavioural change in about half of the trajectories at g = 1 (160 and 135 of 300; |Δ log-odds| at
  step 64 1.2 and 0.9). Self-model twins: under predictability-seeking a 1 % perturbation ends at 0.8 % (peak 1.6 %), under
  curiosity and learning progress it grows to 9–10 %.
* Self-model regimes (exploratory replication, amendment 16: standard contexts, one seed, 15 replicate systems per
  policy and gain, g = 0.5 / 1.0). B against random actions: late self-model error −0.044 [−0.076, −0.019] / −0.041
  [−0.060, −0.022], late workspace step −0.082 [−0.116, −0.050] / −0.211 [−0.268, −0.152]; 98 % exact fixed points
  (random 86 % / 76 %), late action entropy per trajectory 0.06 / 0.00 nats (random 2.06): the same habit. Curiosity C1
  against random: error −0.014 [−0.032, 0.000] / −0.007 [−0.021, 0.008], step −0.007 [−0.031, 0.016] / −0.024 [−0.089,
  0.020] — indistinguishable; yoked replays match C1. Learning progress C2 sits between (step −0.030 [−0.059, −0.005] /
  −0.089 [−0.151, −0.026]). The frozen model F repeats one action and loses track (error +0.089 / +0.151).

<!--FIG:e1_h1_patch_qwen3-14b_rep.png|Patch effects by layer and site for Qwen3-14B|Qwen3-14B, same patch design as the 8B figure: again only the decision position carries the answer, and mostly outside J.-->

<!--FIG:e2_memory_qwen3-14b_rep.png|14B memory gap over loop steps|Qwen3-14B: evidence carried only by the loop falls to a few per cent within a few steps at every gain and for every control.-->

<!--FIG:e2_selfmodel_qwen3-14b_rep.png|14B self-model regimes|Qwen3-14B self-model regimes (one seed): B (red) ends with the lowest error and, with the frozen model, the least change; curiosity, random actions and yoked replays overlap; the frozen model loses track.-->

**What happened conceptually.** Nothing changes with size inside the family: the J component of the 14B decision state
carries a minority of the answer, the J loop is a contraction onto context-specific fixed points, the workspace can be
rewritten without moving the decision, and a predictability-seeking self-model again simplifies its loop by habit while
curiosity is indistinguishable from random action. The one difference is H2: the 14B late residual (not J) carries a
small state-specific part of its future responses beyond the evidence label.

**What remains unestablished.** The 14B self-model run used one seed and standard contexts only; memory under the 14B
self-model regimes was not measured.


## Natural recurrence — Huginn-0125 and Ouro-2.6B

**What changed physically.** The same 1,000 contexts were run through two trained recurrent-depth models with their
native forward code. Huginn: prelude -> `x_{k+1} = core(adapter([x_k, e]))` with the input embedding `e` re-injected at
every recurrence and a random initial state (now seeded so twin runs share it), 64 recurrences, readout by the coda
(exit lens). Ouro: all 48 layers applied per loop with renormalisation between loops and a per-loop exit head; 4 loops
trained, loops 5–8 out of distribution. Tests: accuracy per recurrence; decision-position transplant from the A or B
context into the neutral one at a loop boundary; full-state swap between A and B; 1 % perturbation of the decision
state.

**What happened numerically.**
* Ouro: accuracy rises over the trained loops and plateaus out of distribution — pointer task F2 0.53 -> 0.75 -> 0.80
  (loops 1–3), 0.78 at loop 8; chains F5 0.57 -> 0.77 at loop 4, 0.79 at loop 8; concept recall F6 1.0 from loop 2;
  false-belief F3 at chance throughout. The decision state's change per loop contracts (1.08, 0.47, 0.14, 0.08 ... 0.04).
* Ouro memory: a decision-position transplant moves the answer only in the loop where it is inserted (F6: 12.5 log-odds
  at loop 2 -> 0.13 at loop 3; F2: 2.1 -> 0.15) — the next loop rebuilds the decision from the rest of the sequence. A
  full sequence-state swap is followed permanently: the A-run carrying B's state answers like B through loop 8
  (−2.6 vs +1.9 naturally), and vice versa (+3.3 vs −0.7).
* Ouro sensitivity: a 1 % perturbation of the decision state stays at 1.04–1.09 % through loop 8 (neutral), moving the
  answer by 0.03 log-odds.
* Huginn accuracy grows with recurrence: pointer task F2 0.49 (k = 0) -> 0.90 (k = 8) -> 0.94 (16) -> 0.97 (48–64);
  concept recall F6 1.0 from k = 8; chains F5 non-monotonic (0.76 at k = 8, 0.58 at k = 64); false belief F3 at chance.
* Different random initial states converge to the same answer (|Δ log-odds| between two seeds 1.18 at k = 0, 0.13 at
  k = 16, 0.06 at k = 32, 0.05 at k = 64), but the decision state never becomes an exact fixed point: it keeps moving by ~1.3 % per
  recurrence from k = 32 to 64 (no trajectory below 1 %), a small persistent orbit. A 1 % perturbation grows to 1.4 %
  and stays there (ratio 1.23 over 47 recurrences).
* Huginn memory: a decision-position transplant at k = 16 moves F2 by 1.74 -> 0.72 -> 0.10 -> 0.0 over 1, 2, 4, 8
  recurrences (F6: 0.94 -> 0.17 -> 0.0); at k = 32 the same (F2 1.61 -> 0.01 by 8 recurrences); at k = 4 the effect is
  small and of mixed sign (F2 −0.40, F6 +0.28) and gone by k = 8. Even a full latent-state swap (all positions; each run keeps its own input injection) only
  dents the answer: A carrying B's state answers +0.72 one recurrence later (natural +3.27) and +2.95 three recurrences
  later. The re-injected input re-imposes its own attractor.
* The evidence is linearly decodable from the decision state from k = 8 on (held-out scenarios, 0.86–0.97). (At k = 0 the
  same probe reads 0.85 because the seeded initial noise repeats per batch row; values before k = 8 are not interpretable.)

**What happened conceptually.** In both trained recurrent models the decision position is a readout that the next pass
recomputes: Huginn re-injects the input every recurrence, so any state — transplanted, swapped or randomly initialised —
is pulled back to the input's attractor within a few recurrences; Ouro does not re-inject, so the whole sequence state
is the memory (a full swap is followed for good) while the single decision position is rebuilt from it every loop.
This is the same organisation the J-space loop exposed in Qwen: the single decision-position state is not where memory
lives.

<!--FIG:huginn_main.png|Huginn accuracy, decodability and transplant decay|Huginn-0125: accuracy rises with recurrence (left), the evidence is decodable from the decision state once k ≥ 8 (middle), and a decision-state transplant at k = 16 is gone within a few recurrences (right).-->

<!--FIG:ouro_main.png|Ouro accuracy and transplant decay|Ouro-2.6B: accuracy over loops (dotted line: trained depth) and decision-position transplants, which act only in the loop where they are inserted.-->

## Synthesis — what this means for the strange-loop project

**1. J-space, as read by the published lens on Qwen3, is a readout, not the bottleneck.** The author's motivation was that
the workspace is the natural analogue of the reversed autoencoder's bottleneck: a compact, highly relevant description of
the system that could be looped back. On Qwen3-8B and 14B the J component of the decision state does carry
interpretable content — whether the evidence settles the question, the topic, and late the answer's surface form. But
the variables that decide the answer (which person, which colour, which belief) are carried outside it: swapping the J
component moves 8B answers 21 % of the way at layer 29 and 14B answers 14 % at layer 33 (flipping none), while the
non-J remainder moves them 52 % / 58 %. A loop that feeds back the J readout feeds back a description from which the
decision cannot be rebuilt.

**2. The state knows what it has seen, not how it will behave.** Dispositions (how far a state's answer will move under
a contradiction, an authority claim, doubt, or an internal push) are readable from the state before the challenge,
which passed the pre-registered test. The content-label control shows why: the state encodes which kind of evidence it
has seen, and evidence type governs robustness. Beyond that label the 8B state holds almost nothing (best
representation +0.03 of R²); the 14B late decision-position residual holds a little (+0.11 at layer 28); J holds
nothing in either model (negative at all 16 F2 cells, reliably so at 15). Whatever self-knowledge a larger model may carry about its own responses sits outside the
workspace.

**3. Looping the workspace back produces the collapse extreme, not structured change.** The unforced J loop is a strong
contraction in both models: exact fixed points within a few steps, evidence reduced to a few per cent within ~4 steps,
no answer improved. Removing the norm bound gives the other trivial extreme (divergence). Dense and PCA feedback drift a little
more; random feedback is noise; the fixed points are filled with colloquial web tokens that happen to reproduce
themselves through layers 22 -> 29. Feeding back the system's own forecast error instead (arm X) gives the
noise extreme: the error is dominated by function words the lens does not forecast.

**4. A self-model coupled to the loop reproduces the self-simplification of 2024, by habit.** A self-model that steers
toward what it can predict makes the system simpler (98–99 % exact fixed points, lower error, slower change) by settling
on one repeated action per context, in 8B and again in 14B. Tensions computed from the system's own model — seeking its prediction errors
(curiosity) or its learning progress — did not produce organised dynamics: curiosity behaves like random action and like
a blind replay of its own action sequence (it differs from uniform random actions only through its action
statistics); learning progress sits between. The one place curiosity differs from predictability-seeking is
sensitivity: a 1 % perturbation grows to 9 % under curiosity and stays at 0.4 % under predictability-seeking, because
error-seeking keeps changing its action as the state changes. The same ordering holds at four times stronger coupling (g = 1 / 2), where
the loop is far more active: no regime shows structure that a blind replay of its own action sequence does not also
show.

**5. The decision position is the wrong place for a loop; trained recurrent models show where memory lives.** Huginn and
Ouro, which were trained to iterate, also rebuild the decision position at every pass: a transplanted decision state is
gone within one loop (Ouro) or a few recurrences (Huginn). Ouro follows a swap of the whole sequence state for good,
while Huginn's re-injected input re-imposes its attractor. In all three settings the single decision-position state is
a readout of the sequence, so a loop through it alone cannot hold anything the context does not already hold.

**6. Verdicts.** Not supported: J-space as a privileged carrier of relations, decisions or dispositions in these models;
a J-space loop that sustains task information; emergent structured dynamics in any declared regime; a loop that helps
the task. Supported, and new: the decision-position workspace holds epistemic-status, topic and answer-onset words;
the J loop is a contraction whose fixed points can be moved far from the natural workspace without moving the decision;
a predictability-seeking self-model simplifies its own loop by habit; trained recurrent-depth models keep memory in the
sequence state, not in the decision position.

**7. Suggested next steps (concrete).** (a) Build the bottleneck from the causally sufficient part of the state instead
of the verbalisable part: a low-rank subspace learned with an interchange-intervention objective, so that swapping it
transfers the answer, then loop that. (b) Loop the whole sequence state rather than one position — write the code
at every scenario position, or use Ouro's full-state recurrence, where a swapped state is followed for good. (c) Run
the self-model regimes there, with a richer action space (actions that write new concepts, not only rescale present
ones). (d) Keep the bounded broadcast, matched-norm controls and the content-label control; they separate content from
amplitude and self-knowledge from content.

## Costs, provenance and what was not done

* Compute: one Vast.ai A100 PCIe 80 GB instance (contract 55429215, $0.975/h) from 18:35 UTC on 11 Oct to 03:16 UTC on 12 Oct (self-destroyed after the last upload),
  driven by an unattended queue that uploaded results to the private HF dataset after every job and destroyed the
  instance at the end; one earlier instance that never started was destroyed unused. Credit went from $11.22 to $2.39: $8.83 in total, almost all of it GPU time.
* Every model and lens is pinned to a commit hash (provenance.json per run, lens sha256 recorded). Fixed seeds:
  scenario generation 20261011 (+ family offsets; 14B E1: "rep" offsets), loop seeds per chunk, fixed bootstrap seeds.
* Exploratory additions after the freeze are labelled in the text and listed in PROTOCOL_V3.md amendments 11–17.
* Known defect: the F5 chain-task digit readout in the Qwen loop runs is broken (accuracy below chance in every
  condition); no F5 number from those runs is used. F5 accuracies for Huginn and Ouro come from their own two-candidate
  readout and look sound.
* Not done: Architecture B (training a coupled loop), multi-position or whole-sequence loops on Qwen, memory under the
  strong-coupling and 14B self-model regimes, and any loop on a causally trained bottleneck — the most promising next
  step (synthesis, point 7).
