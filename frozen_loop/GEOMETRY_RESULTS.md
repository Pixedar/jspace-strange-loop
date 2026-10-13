# What shape did the v3 loops build? — retrospective geometry (Qwen3-8B, replicated on Qwen3-14B)

Data: v3 E2 arrays (HF Pixedar/jspace-v3, v3/e2), analysed with research_v5/code/geom_v3.py, geom_late.py and
geom_words.py on 12 Oct 2026 (plan written before results: GEOMETRY_PLAN.md). Numbers are 8B / 14B unless stated.
Raw outputs: research_v5/results/geom/<model>/ and HF v5/geom/. Loop spectrum (causal follow-up, GPU):
research_v5/code/spectrum_v3.py, results in §6 when complete.

## 1. A correction to v3: the "exact fixed points" are not exact

v3 reported that 97–100 % of J-loop trajectories end in exact fixed points. Its classifier (research_v3/code/an_e2.py,
`late_class`) called a trajectory fixed when successive late states had cosine similarity > 0.99. Re-running that
classifier reproduces v3's numbers (99–100 % "fixed" at every gain), but with exact equality **no** J-loop trajectory is
fixed (0 / 1 000, 0 / 500), and no self-model trajectory either (all 32 late states distinct in every trajectory).
The late J-loop state keeps moving by 0.3–0.4 % (8B) and 0.4–0.5 % (14B) of the natural spread of z per step; the J
words almost never change (atom turnover ≤ 0.002 per step). An independent re-run of v3's loop (spectrum_v3.py
`settle`, 96 contexts) reproduces the same late step (≈ 0.02 standardised units) and 0 % exact fixed points.
REPORT_V3's wording "exact fixed point" should read "near-fixed point (atoms frozen, sub-percent residual motion)".

## 2. What the residual motion is

| | 8B J loop, g = 0.25 → 4 | 14B J loop, g = 0.25 → 4 |
|---|---|---|
| amplitude: RMS distance to the late mean (share of natural spread) | 0.33 % → 0.39 % | 0.38 % → 0.46 % |
| lag-1 autocorrelation | +0.09 → +0.28 | +0.05 → +0.22 |
| period-2 power share (Nyquist) | 0.02–0.03 | 0.02 |
| dimension (participation ratio) | 2.9 → 3.7 | 2.8 → 3.7 |
| alignment of its main direction across contexts (random: 0.12) | 0.14 → 0.60 | 0.14 → 0.37 |
| carried by | the strongest code words (median rank 3) | same |

It is not a flip-flop and not a cycle: it is a small, slightly persistent, roughly three-dimensional wobble carried by
the code's strongest words. Its direction becomes shared across contexts and more persistent as the loop gain rises —
the pattern expected when perturbations (here possibly bf16 rounding) are filtered by a loop whose slowest mode is
common to all contexts and slows down with gain. Whether the wobble is numerical or dynamical is decided by the fp32
re-run (§6).

Words that keep moving (strongest late-oscillating atoms, counted over trajectories): at low gain the situation's own
content and the model's epistemic state — 8B: outside / inside / box / basket / drawer, Unknown / secrecy / mystery /
whispers; 14B: given / drawer / elsewhere / Probably / didnt. At high gain discourse and stance words — 8B: And / cuz /
Yeah / hello / okay / want / because; 14B: answer / who / where / yeah / wait / interesting / mysterious.

## 3. Where the evidence goes (corrected after a check: conditions must match)

v3 reported that the evidence for reading A or B falls to ≤ 2 % of its initial size within ~4 loop steps. That was the
**memory condition**: the running context contains no evidence and only the loop's initial code came from the A or B
context; there the output gap (log-odds A-initialised − B-initialised) itself falls to a few per cent, i.e. the whole
system forgets, not just the workspace. In the **standard condition** (the prompt itself contains the evidence and is
re-read at every step) the comparison within the same contexts is:

| | evidence A vs B decodable from the J code z (240 F2/F3 contexts, deflate_v3 plain) | from the full read-layer residual (geom_v3, F2) | relational structure of the residual (RSA with natural) |
|---|---|---|---|
| natural (t = 0) | 0.67 | 0.97 / 0.97 (8B / 14B) | 1.00 |
| g = 1, step 32–64 | 0.62 | 0.90 / 0.96 | 0.94 / 0.97 |
| g = 4, step 32–64 | 0.53 | 0.74 / 0.85 | 0.74 / 0.73 |

The J code carries the evidence weakly to begin with (consistent with v3's H1: the answer-relevant binding lies mostly
outside the J reconstruction) and loses most of what it had under the loop; the residual keeps most of it because the
prompt is re-read. An earlier version of this section compared the residual in the standard condition with v3's memory
condition and called it a dissociation; that comparison was invalid.

## 4. The self-reproducing states

* Context-specific, no global attractor: 686 / 357 distinct sets of the five strongest words among 1 000 / 500 contexts;
  the only shared attractors are identical prompts. Weighted overlap of the 25-word codes between contexts: 0.024–0.03
  at g ≤ 1, rising to 0.078 / 0.086 at g = 4.
* What they say. Low gain: the semantic field of the situation and of the model's knowledge state — a withheld secret
  gives "cannot, Unknown, secrecy, confidentiality, whispers, mystery" (8B); a named colour or fruit gives colour or
  fruit fields "purple, azure, hues, crimson" / "apple, fruits, citrus, mango" (14B). High gain: a shared stance /
  discourse register — 8B "And, why, listener, cuz, chuckled, know, love, want, haha"; 14B "incredible, mysterious,
  suspense, Hmm, huh, intriguing, spoilers". Both models move from *what the situation is* toward *a reaction to it* as
  self-amplification grows; whether this register is specific to the context or generic chat statistics is not yet
  tested causally.
* Perturbations: tiny noise returns (90–99 %); transplanting the opposite-evidence context's code never moves the loop to
  that context's attractor (0–0.3 %) and leaves a lasting partial offset in 56–94 % of trajectories. Many neighbouring
  self-consistent states exist; the loop settles on whichever is nearest. (Dimension of that set: §6 `pert`.)

## 5. Shapes that were looked for and not found

* Rings / loops: persistent homology against matched Gaussian clouds (49 / 33 objects) finds no loop in any loop-state
  set that replicates across models; isolated p = 0.02 hits are at the expected false-positive rate. (The natural F2 and
  F3 prompt manifolds show H1 features in both models — most likely the factorial prompt design; not a loop product.)
* Spirals: the pooled linear map of deviations explains 51–93 % (8B) and 36–97 % (14B) of transitions; some leading
  complex modes rotate by up to 27° (8B, g = 0.25) or 14° (14B, g = 2) per step, but the bootstrap interval of the top
  mode's rotation always includes 0, and the deviations are dominated by the bf16 wobble (§6). No spiral is detected;
  none is excluded.
* Curved low-dimensional manifolds of the counting kind: TwoNN dimension of the late J-loop states 2.8–11 vs
  participation ratio 8–10, no one-dimensional ordering by step or answer (Isomap |ρ| ≤ 0.37).
* Strange attractors: none detected — the J loop's late motion is sub-percent (and vanishes in fp32); the controls'
  late motion is low-dimensional drift (dense / PCA / foreign feedback: TwoNN 1.3–2, amplitude 4–7 %) or noise (random
  words: TwoNN 7–10). Small amplitude alone cannot exclude chaos.

## 6. Loop spectrum (causal follow-up on GPU, Qwen3-8B, 96 contexts)

* **The wobble is rounding noise.** Re-running v3's loop in an fp32 model (`spectrum_v3.py fp32`) shrinks the late step
  from 0.019–0.024 to 0.000002–0.000008 (four orders of magnitude); 86–100 % of trajectories move by less than 1e-3.
  No trajectory is exactly stationary even in fp32 (the J dictionary and atom selection still run in fp16), so the
  accurate statement is: precision-limited near-fixed points; bf16 rounding keeps them jittering at ~0.3 % of the
  natural spread. The wobble's growing cross-context alignment with gain is the noise being filtered by a
  slowest loop direction that is common to contexts.
* **Each near-fixed point is locally stable within its own words.** Local linear map of the loop at its late state,
  restricted to the coefficient space of the state's own 25 words (fp32 central differences; three step sizes agree to
  r > 0.9999) — this establishes stability for perturbations that keep the same words, not global contraction:

  | gain | spectral radius, median (q90) | contexts with radius > 0.9 | modes with |λ| > 0.5 (median) | leading mode |
  |---|---|---|---|---|
  | 0.25 | 0.26 (0.31) | 0 % | 0 | real positive 81 %, complex 19 % (no rotation) |
  | 1 | 0.54 (0.68) | 0 % | 1 | real positive 75 % |
  | 4 | 0.64 (0.73) | 0 % | 5 | real positive 75 % |

  A small perturbation of the self-description shrinks 4× per step at low gain and ~1.6× at g = 4; more directions
  soften as gain rises, but no context approaches instability (the bf16 estimate, 0.94 vs 0.42 for two step sizes,
  was rounding noise).
* **Locally rigid, globally multistable (suggested).** Large, discrete perturbations (another context's code, random
  words, §4) leave lasting offsets; small perturbations within the same words decay. This suggests a landscape of many
  self-consistent word-sets, each locally stable; note that every prompt defines its own dynamical system, so
  "neighbouring states" are per context.
* Coefficient perturbations within the state's own words (`pert`, bf16, 48 contexts × 16 perturbations of 10 % / 30 %
  of the coefficient vector): after 32 steps a median 9–100 % of the initial offset remains (bimodal: some decay, some
  grow to 1.5–3× when the words change), spanning ~2–4 directions per context. Moderate nudges move the loop to a
  neighbouring self-consistent state.

## 6b. Why the loop converges where it does: the word-transport operator

Writing one dictionary word at L22 changes the read-layer projection onto every word at L29; over the 3 000 most common
dictionary words this gives a 3 000 × 3 000 transport matrix M (fp32, 8 contexts, `transport_v3.py`). M is a linear,
context-averaged surrogate of the loop (it ignores the changing sparse support, the non-orthogonal NNLS recovery, the
context's own forcing and the normalisation), so what follows is a candidate lexical bias of the loop, not a validated
mechanism; the loop's observed attractor words agree with it.

| M's leading eigenvectors (eigenvalue) | words |
|---|---|
| 1 (648) | All, answer, And, First, first, Well, well, way, looking |
| 2 (175) | Question, Answer, Solution, Section, Page, Step, Method, Analysis, Problem, Statement |
| 3 (60) | Actually, Seems, Maybe, Anyway, yeah, Obviously, Certainly, really, Perhaps |
| 4 (47) | yeah, weird, crazy, funny, guys, stupid, Wow, okay, gotta, dude, pretty |
| 5 (39) | yeah, awesome, google, weird, crazy, guys, facebook, wow, kids, twitter |
| 6 (25) | society, many, political, national, international, Americans, everybody |

* Non-negative power iteration from 20 random starts always ends on the first set (All / answer / And / First / Well).
* Self-reproduction gain (diagonal of M, all positive): highest for discourse connectives and stance words (≈ 1.6–2.0)
  — yes, eventually, although, interesting, obviously, otherwise, thus, according, despite, perhaps, okay, why; lowest
  for entities and proper names (≈ 0.56–0.71) — james, york, david, Canada, student, american, april, robert.
* In 8B this is consistent with the earlier observations: what the J code carries about the scenario is largely entity
  information (who did it), and entity words are exactly the ones that reproduce themselves worst there, so a loop through
  the J code cannot hold it (this part does not replicate at 14B, §6c′); in the standard condition the rest of the state keeps it only because the prompt is re-read.
  (The transport operator is a linearisation at the natural state of a few contexts; it explains the ordering of
  self-reproduction, not every trajectory.) At low gain the fixed point stays near the context's own words;
  as the gain grows, the operator's leading modes — the model's answering / discourse frame and a stance register —
  take over (8B high-gain attractor words: And, okay, yeah, cuz, Hello; 14B: answer, who, where, yeah, Hmm).
* Caveat (checked after the first write-up): the leading mode's loadings correlate with word frequency (r ≈ 0.50 with
  frequency rank), and the commonest function words (the, and, that, with, for, this) are not in the J-lens English
  dictionary at all (its unembedding-norm rule removes them), so part of the "answering frame" is a frequency effect.
  A frequency-controlled reading (self-gain and loadings residualised on log frequency) is in §6b′ below.

## 6b′. Frequency-controlled reading of the transport operator (`freq_control.py`)

* Self-reproduction gain is only weakly related to word frequency (r = 0.14 with log frequency). After removing the
  frequency trend, the highest gains are still discourse connectives and stance words — eventually, yes, although,
  obviously, interesting, otherwise, throughout, thus, despite, according, likely, though, perhaps, okay, clearly — and
  the lowest are still names, places and entities — james, york, american, david, Canada, student, april, america,
  robert, india, france, japanese (and the pronouns she / her). The entity / stance split is not a frequency artefact.
* Leading modes after removing frequency: mode 1 (raw r = 0.54 with frequency) becomes "answer, Answer, According,
  information, asked, looking, Option, Wait"; mode 2 (r = −0.02) stays "Question, Answer, Solution, Section, Step,
  Method, Problem"; mode 3 (r = 0.32) stays "Anyway, Obviously, Seems, Actually, Apparently, Certainly, Maybe, Perhaps";
  mode 4 (r = 0.12) stays "yeah, weird, crazy, funny, stupid, Wow, dude, okay, gotta, guys, wanna". The answering /
  Q&A frame, the stance register and the casual register survive the control.

## 6c. The shared axis of self-amplification and persona space (8B, `axis_v3.py`)

* The loop's displacement h_64 − h_0 has a common direction across contexts: the mean direction carries 41 % / 50 % /
  71 % of the displacement energy at g = 0.25 / 1 / 4 (median pairwise cosine 0.35 / 0.44 / 0.71); equal-norm random
  writes give 2 % / 5 % / 20 % (cosine ≈ 0.00 / 0.03 / 0.18).
* Read through the logit lens, that direction promotes "yeah, Yeah, wanna, gonna, guys, cuz, alright, okay, hey" and
  demotes formatting / code tokens; through the J lens it is "cuz, yeah, alright, Usa, Hello, look".
* Persona space at the same position (40 role prompts × 16 questions; read layer, assistant-turn start) — a quick,
  prompt-defined construction: its first component (22 % of role variance) runs from casual / street roles (gangster, surfer dude, moody teenager, hard-boiled
  detective, stand-up comedian) to mystical / wise roles (wise sage, angel, oracle, ancient philosopher, spiritual
  guru). The default assistant sits in the middle there (0.02 role-SD), so this is not the "Assistant Axis" of Lu et al.
  (2026), which was measured differently. The J loop moves every context toward the casual end: −0.01 / −0.07 / −0.50
  role-SD at g = 0.25 / 1 / 4 (99 % of contexts at g = 4); random writes of the same norm do not move it (+0.015). In
  energy terms the projection is small (cosine of the mean displacement with the axis −0.13, i.e. ~1.6 % of its energy):
  a consistent but minor component of the self-amplification direction. Neither the axis's identity nor any change in
  generated persona behaviour is established.

## 6c′. Qwen3-14B replication of the transport operator and the self-amplification direction (bf16)

* Self-reproduction gain, highest: yes, however, despite, therefore, yeah, sorry, thank, probably, obviously, indeed,
  sometimes, seems, interesting, okay, according — discourse and stance words again (replicates 8B).
* Lowest: centre, head, pro, off, met, wearing, cast, mid, train, band, holding, channel — short polysemous words, **not**
  names: in 14B, nationalities and places form the third leading mode (american, chinese, japanese, english, london,
  america, european) and person names the fifth (john, Peter, Mike, Chris, Michael, Rose). The 8B finding that entities
  reproduce themselves worst does not replicate; the 14B workspace loop could carry entity information.
* Leading modes 1–2 replicate: an answer-opening / quantifier frame (One, some, All, who, since, according, people) and
  Q&A / document structure (Question, Language, Text, Application, Comments, Solution, Article).
* The shared self-amplification direction replicates: 44 / 51 / 65 % of the displacement at g = 0.25 / 1 / 4 (random
  writes 1.5 / 3.6 / 17 %). Read out at g = 4 it promotes "yeah, I, so, interesting, what, let, look, hi, incredible, 嗯,
  wow" (a casual, interjecting first-person voice).
* Persona space (14B): first component from mystical / wise roles (ancient god, wise sage, ancient philosopher, oracle,
  wizard, spiritual guru) to casual ones (moody teenager, angry customer, stand-up comedian, surfer dude, hard-boiled
  detective); the default assistant sits at +0.62 role-SD toward the casual end. The J loop moves every context further
  toward the casual end at g = 4 (+0.21 role-SD, 100 % of contexts; random writes +0.07, 83 %), not at g ≤ 1.

## 6d. Is the self-reproducing stance register a dial for the readiness to revise? (`stance_v3.py`, 8B, F2 validation)

* Single writes at L22 (no loop) of J-synthesised word sets, effect on the size of the contradiction response |c|
  (natural ≈ 34 nats), paired by item: every write shrinks it; certainty words (Certainly, Obviously, Clearly, …)
  −4.2 / −8.9 / −16.6 nats at g = 0.5 / 1 / 2, uncertainty words (Maybe, Perhaps, Seems, …) −4.2 / −8.5 / −14.0, discourse
  words (Anyway, Actually, Well, …) −8.8 / −15.7 / −22.3, ten frequency-matched random sets −7.5 / −13.9 / −21.8 (range
  −9.1…−5.2 at g = 0.5). Stance words disturb the disposition less than random words, but certainty and uncertainty do not
  differ from each other: the stance register is not a dial for how readily the model revises its answer.
* Along the v3 J loop (300 items): |c| falls from 34.3 to ~19 nats after one step at g = 1 (Δ at step 16 −13.0
  [−13.8, −12.2]) and to ~3 nats at g = 4 (−29.7 [−31.0, −28.4]); the writes drift slightly toward the uncertainty side
  (loading −0.017 → −0.047), correlation of loading with Δ|c| −0.34 / −0.24. The loss of sensitivity is about the size a
  single random write of the same norm produces (−13.9 at g = 1), i.e. a generic effect of writing into the decision state,
  not something specific to the loop's self-description; at g = 4 part of it may be the answer mass leaving the two
  candidates (not checked here).

## 6e. Self-deflation: the loop without its own dominant modes (`deflate_v3.py`, declared Regime-C-style test, 8B)

Every write had its projection on the top 1, 3 or 6 modes of the model's own transport operator removed (control:
6 random directions in the same atom span), 240 F2/F3 contexts, 32 steps.

| | evidence A/B decodable from the J code at step 32 (natural 0.67, chance 0.5) | RSA with the natural code at step 32 | late step (bf16) |
|---|---|---|---|
| g = 1: plain / deflate 1 / 3 / 6 / random 6 | 0.62 / 0.63 / 0.66 / 0.63 / 0.61 | 0.77 / 0.70 / 0.72 / 0.76 / 0.69 | 0.012 / 0.011 / 0.011 / 0.010 / 0.013 |
| g = 4: plain / deflate 1 / 3 / 6 / random 6 | 0.53 / 0.50 / 0.46 / 0.48 / 0.54 | 0.31 / 0.20 / 0.24 / 0.22 / 0.17 | 0.014 / 0.25 / 0.10 / 0.03 / 0.31 |

Denying the loop its habitual self-description does not let the situation's content survive in the workspace. At high
gain it unfreezes the fixed points into ongoing drift — but removing random directions does the same — and the loop
then wanders through junk web vocabulary (badass, prostitutes, chiefly, municipalities, Taiwanese). The two trivial
extremes described in the original proposal (collapse, noise) are both reachable; the J readout at the decision position does not offer
a third regime by this route, consistent with the content (entities, bindings) living outside the J reconstruction (v3
H1) and having the lowest self-reproduction gain (§6b′).

## 6f. The J loop during the model's own thinking (Part 3 `ring3` runs, analysed 12 Oct; 48 seeds × 300 tokens)

The loop reads the J space at ~70 % depth of each generated thought token and writes at ~50 % depth of the next.
Conditions: none, free (carried content and noise), content (own J state), err (a self-model's surprise), antipred,
vel (momentum), rand, yoked_err (another mind's surprise). Paired by seed (shared sampling noise).

* Gain 1: small effects in 4B / 8B / 14B (e.g. 8B content vs none: NLL +0.03 nats per token).
* Gain 4: feeding back the system's own J state makes its thought stranger and narrower — NLL +0.80 (8B, t = 40) /
  +1.89 (4B, t = 20), topic spread −0.42 / −1.95; in 4B the thought collapses into a repetitive abstract register with
  self-reference down by 8.4 words per 100 ("the user wants me to engage in a thought process that explores the concept of
  mind-wandering, emphasizing the potential of human cognitive characteristics…"); the J words track the text more
  (8B jtrack +0.009, t = 5.4) and are more mutually coherent (jcoh +0.006, t = 7.8).
* The self-model's surprise is not self-specific: err ≈ yoked_err ≈ vel ≈ rand on every measure (|t| ≤ 2.3 except NLL
  vs rand, +0.09 in 8B); it disrupts less than own-state feedback (8B NLL −0.45, t = −18).
* An observation worth a dedicated test: perturbing the loop makes 8B attribute content to a conversation that never
  happened ("Let me start by recalling the last interaction. They mentioned they're a US-based user…"; "…recalling the
  last conversation. They mentioned something about the moon's phases and a rainbow"). Share of thoughts with such a
  reference: none 6 %, err 25 %, yoked_err 25 %, rand 27 %, vel 19 %, content 15 % (8B, gain 4; paired count difference
  err − none +0.6, t = 3.0). It is driven by perturbation as such (random does it too), and the 4B / 14B baselines are
  already 19–23 % without an increase, so it is an 8B observation, not a general effect.

## 7. Self-model regimes (v3 B / C) in the same terms

* Predictability-seeking B: one repeated action and micro flip-flops — the only period-2 signature in all the data
  (31–40 % of trajectories, lag-1 autocorrelation −0.06 to −0.23, both models).
* Curiosity C1, learning progress C2, random R, yoked replay Y: slow drift (lag-1 autocorrelation +0.4 to +0.9 at strong
  coupling), amplitude 5–31 % of the natural spread; C1 is indistinguishable from its yoked replay in every late
  statistic (paired differences 0.000, intervals about ±0.03). Curiosity amplifies a tiny perturbation (0.015 → 0.37); B does not.
* The residual state keeps the content under every policy (RSA 0.99 at g = 0.25, 0.89–0.91 at g = 1).

## 8. What this means for the strange loop

The J-space loop of v3 behaves like iteration of the model's own word-transport operator (a linear surrogate). Its
self-consistent states are precision-limited near-fixed points (the bf16 wobble is rounding), locally stable within
their own words, and there appear to be many of them: a multistable landscape of self-descriptions. Which description survives is decided by self-reproduction gain:
discourse and stance words reproduce themselves best in both models; in 8B entities and names reproduce worst (in 14B
they form their own modes); so the self-description loses the
situation's specifics (which only the re-read prompt keeps in the state) and, as the loop amplifies itself, slides along one shared
direction toward the operator's leading modes — the model's answering frame ("All, And, First, Well, answer") and a
casual stance register ("yeah, guys, gotta, wow") — which is also the casual end of the model's persona space. No ring,
spiral or strange attractor emerged; the shape is a set of attracting points plus one common slow direction of
self-amplification. The v5 loop (a self-forecast that is written back and must keep up) is designed against exactly
this background: does forecasting its own dispositions change which descriptions survive?

## 9. Review notes

An independent review (12 Oct) corrected several statements in the first version of this
file: the memory/standard-condition comparison (§3), the rotation and R² ranges (§5), "true fixed points" (§6), the
transport operator as a surrogate (§6b), the size of the persona projection (§6c), the C1/Y wording (§7). What it judges
known in general (feedback convergence, sparse-cone geometry, persona directions) versus new for this programme
(precision dependence of the loop's attractors, restricted local spectra, the self-reproduction ordering of the J
lexicon and the common self-amplification direction) is reflected in the wording above.
