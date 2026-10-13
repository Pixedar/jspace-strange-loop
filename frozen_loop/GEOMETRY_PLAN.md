# Research v5 — geometry plan: what shape does the loop build?

Status: written 12 Oct 2026, before any geometry result was computed. Retrospective phase (existing v3 arrays, CPU only)
runs first; the prospective phase is part of PROTOCOL_V5 (§10) and runs only after qualification.

## 0. Question and why it belongs to the original programme

The question (12 Oct 2026): what internal representation, with what abstract shape, emerges when the model's
compact state is looped back (J-space recurrence) and when a self-model acts on that loop (the strange loop)? Reference
points are known representational geometries in language models: character counts on curved, rippled one-dimensional
manifolds that attention heads rotate (Gurnee et al. 2025, transformer-circuits.pub/2025/linebreaks), circular features
for days and months (Engels et al. 2024). This is the accepted proposal's Experiment 3 (meaning and geometry) and the
relational part of H1. It is descriptive science about the loop itself; no benchmark enters it.

One framing is fixed in advance because it follows from the loop's definition rather than from any data: a state
that the loop maps onto itself (read it, compress it, broadcast it, read the same thing again) is the formal
self-referential object of the system, a "diagonal" fixed point. The geometry of the set of such self-reproducing
states across contexts, and of the paths into them, is therefore the most direct answer to "what shape does the strange
loop have". Cycles and non-periodic attractors, if they exist, are the alternatives.

## 1. Objects (retrospective phase; Qwen3-8B main and Qwen3-14B replication, HF Pixedar/jspace-v3, prefix v3/e2)

| id | object | arrays |
|---|---|---|
| O1 | natural manifold: decision-position residual at the read layer, no write, one point per context (F2 interpretation, F3 false belief, F6 concept memory, F5 variable chains; tags A / B / N) | A1 `H[0]` of no-feedback rows |
| O2 | flow under J-space feedback: 64-step trajectories at gains 0.25, 0.5, 1, 2, 4 | A1 `H[t]`, `z[t]`, `idx/coef[t]` |
| O3 | control flows at the selected gains: dense residual, residual PCA, J pulled back, random words, shuffled weights, foreign code, time-shuffled code | A2 (`z` for all; `H` for J / dense) |
| O4 | self-reproducing set: states at t = 64 and their J codes; basins | A1 / A2 |
| O5 | self-model regimes: an online self-model chooses one of nine internal actions (B predictability-seeking, C1 curiosity, C2 learning progress, R random, F frozen-model curiosity, Y yoked replay) at g = 0.25, 0.5 and strong g = 1, 2 | BC, BCs (8B), BC1 (14B): `z`, `act`, `pred`, `perr`; `H` for seed 0, g1, B/C1/C2/R |
| O6 | perturbation responses at t = 16 (small noise, semantic swap, transplant of the opposite context's code, random code; self-model twins) | P, PBC |
| O7 | the loop's own variable: the 25-word J code (indices + non-negative coefficients) and the 32-d PCA of its reconstruction (`z`) | all stages |

Spaces: (i) residual at the read layer, centred by the natural mean, per-dimension standardised by natural-state
statistics (std floored at 10 % of its median), projected on a pooled PCA frame (fitted once on natural + loop states of
all gains; captured variance reported); robustness repeat in the raw centred residual; (ii) `z`, the loop's
compressed state (already standardised); (iii) the code itself (weighted-Jaccard geometry).

## 2. Measures

M1 linear dimension: participation ratio (PR) and PCA spectrum.
M2 nonlinear intrinsic dimension: TwoNN (Facco et al. 2017; linear fit on the first 90 % of the sorted ratio
distribution), computed on de-duplicated points, with scale stability (full n, n/2, n/4).
M3 topology: Vietoris–Rips persistent homology (ripser; H0, H1) on 10-d PCA of the object, farthest-point subsample
n ≤ 400. Null: Gaussian point clouds with the same covariance and size, same pipeline, 49 surrogates. A loop (H1
feature) is reported only if its persistence exceeds every surrogate's largest H1 persistence (p ≤ 0.02) in both
models; clusters (H0) are reported as the number of components that persist past the surrogate maximum.
M4 flow geometry: deviations from each trajectory's own final state; transient length; per-step contraction ratio;
angle between successive deviations (0° node, 180° flip-flop, intermediate and consistent = rotation); planarity
(variance of the transient in its top two directions); cumulative winding in that plane; a pooled linear map of
deviation transitions (ridge, PCA-64) whose eigenvalues give rotation per step and contraction; bootstrap over
contexts. Deterministic loop: no noise null is needed for direction changes, but the real-eigenvalue curved approach
(bounded winding) is distinguished from rotation (winding growing past 180°).
M5 content retention along the flow: grouped (by scenario) 5-fold logistic decoding of evidence tag A vs B from the
state at step t, per family; RSA = Spearman correlation between the context-by-context distance matrices at step 0 and
step t (does the loop keep the relational structure among contexts, compress it, or destroy it?).
M6 census of self-reproducing states: exact fixed points (`z` bit-identical from t* on), cycles (period 2–16 in the
last 32 steps), none; attractor identity = set of the five strongest code words at t = 64; number of attractors,
occupancy entropy, mutual information with family / tag / subtype (permutation p), linear predictability of the basin
from the natural state (grouped CV), the attractors' words.
M7 self-model regimes: late window t = 32–64 per context — fixed / cycle (period) / non-periodic, distinct states,
step size, action entropy and Lempel–Ziv complexity, self-model error; paired by replicate system against random
actions (R) and yoked replay (Y); visited-set geometry per policy (M1–M3).
M8 perturbation geometry: twin distance curves; fates (exact return, near return, switch to the partner's attractor,
lasting offset); slow-manifold test: alignment of the perturbed trajectory's final approach direction with the
unperturbed trajectory's, against the alignment between different contexts.
M9 coordinate meaning (descriptive only): correlation of the leading coordinates of O1 / O4 / O5 with family, tag,
subtype, answer log-odds, step, gain, action and self-model error, and with the code words present (point-biserial;
words with ≥ 5 % prevalence). Causal confirmation of any coordinate is a GPU step (prospective phase).

## 3. Declared interpretation rules

* Point-attractor regime: ≥ 90 % of trajectories reach exact fixed points. Shape claims then concern the
  self-reproducing set across contexts and the transients, not a per-trajectory attractor.
* Ring / loop: M3 significant in both models AND a circular coordinate that advances along trajectories or maps onto a
  cyclic variable. Otherwise "no loop detected".
* Spiral approach: dominant complex eigenvalue pair of the pooled map with rotation ≥ 10° per step (bootstrap CI
  excluding 10°) AND per-trajectory winding beyond 180° in ≥ 25 % of transients long enough to wind (≥ 6 steps).
* Curved low-dimensional manifold (counting-like): TwoNN dimension well below PR (ratio ≤ 0.5, stable across scales)
  AND a one-dimensional ordering (Isomap coordinate) that correlates ≥ 0.8 (Spearman) with a variable.
* Content-free collapse: tag decoding at t = 64 within chance (grouped CV, 95 % CI includes 0.5) AND RSA(0, 64) ≤ 0.1.
* Strange (non-periodic, low-dimensional) attractor: non-periodic late trajectories in ≥ 25 % of contexts with
  TwoNN dimension of the late set < 5 and stable across scales. Noise-like: non-periodic with dimension near the
  embedding dimension or growing with n.
* Every statement is made separately for 8B and 14B; "model-general" requires both. PCA pictures are never evidence on
  their own (horseshoe / arc artefacts of drifting processes are expected and are not called curvature).

## 4. Outputs

`research_v5/results/geom_<model>.json` (all numbers), figures `research_v5/results/figs/geom_*.png`, raw outputs on HF
`Pixedar/jspace-v3` prefix `v5/geom/`. Compute: one Vast instance, CPU analysis, unattended, uploads and self-destroys.

## 5. Prospective phase (v5, after qualification; summary — details in PROTOCOL_V5 §10)

Same measures on the v5 loop's `(z_t, f_t, w_t, o_t)` in a reference frame frozen on qualification data, at g = 0 and
the selected gain, for own / corrupted / restored branches; plus the object v3 could not have: the self-consistency set
(states whose self-forecast is fulfilled, |o_{t+1} − f_t| below the copy baseline's error) and its geometry relative to
the rest of the flow. Causal validation of any geometric coordinate: moving the write along it must change the
predicted downstream observable more than matched-norm random directions do.
