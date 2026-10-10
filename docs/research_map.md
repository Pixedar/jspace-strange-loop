# Research map: what else is worth testing (10 Oct 2026)

Parts 1–3 grew out of the Athena setup: a think-aloud mind, J-space content fed back into the stream, and collapse
metrics. This map starts from the literature instead. It lists what other groups have found or proposed, what is still
open, and what we can test ourselves with open Qwen3 models (1.7B–32B), the Neuronpedia Jacobian lenses and rented GPUs.

## 1. The J-space as a global workspace: the tests its own critics asked for

- **Gurnee, Lindsey et al. 2026, "Verbalizable Representations Form a Global Workspace"**
  ([paper](https://transformer-circuits.pub/2026/workspace/index.html)). The J-space is reportable, selective,
  capacity-limited and broadcast. Ignition was tested only as a blend of two country tokens, where the readout snaps to
  one endpoint from the workspace onset. Ablation removes the top-10 J-lens vectors per position over a layer band,
  with random-direction controls.
- **Dehaene & Naccache 2026, commentary**
  ([pdf](https://unicog.org/wp_2025/wp-content/uploads/2026/07/Dehaene-and-Naccache-Workspace-commentary-on-Gurnee-Lindsey-June-2026.pdf)).
  They accept the mapping to their global neuronal workspace (GNW) theory but list tests that are still missing:
  - **Ignition, "the decisive experiment".** Graded stimulus strength should switch J-space representations on with a
    threshold-like nonlinearity while earlier layers rise monotonically. At threshold there should be a bifurcation
    across trials, giving a bimodal distribution. The bottleneck should also show: holding one content should impede
    the entry of another (dual-task interference, attentional blink).
  - **Local–global test** (Bekinschtein 2009). Detecting a violation of a global sequence rule should need the
    workspace, while local surprise should not.
  - **Trace conditioning.** A cue and a target separated by a gap. Lindsey's preliminary Claude result: J-space
    ablation impairs long gaps and spares adjacent pairs.
  - **Inclusion/exclusion** (done in Claude: early-workspace ablation breaks deliberate avoidance), **error
    monitoring**, and **resting-state dynamics**. They argue autonomous recurrence, the "strange loop", is largely absent
    in LLMs.
- **Open reproductions:** [mflRevan/jspace-test](https://github.com/mflRevan/jspace-test) on Qwen3.5-4B covers the
  paper's own claims. It has no graded ignition, bimodality, dual-task, local–global, trace conditioning or
  neurofeedback.
- **[Ignition Index](https://arxiv.org/abs/2608.05160)** fits sigmoids to probe accuracy over layers in 11 models. It
  finds transformers more ignition-like than Mamba, and that Huginn ignites along its iteration axis. It uses no
  J-lens, no bimodality and no competition.
- **[Looped transformers under the J-lens](https://arxiv.org/abs/2609.01924)**: a workspace forms in Ouro and Huginn,
  but recurrence restricts access to it (per-loop reconstruction in Ouro, a two-step window in Huginn).
- **[Short horizons and sparse concepts](https://arxiv.org/pdf/2608.25347)**: the J-lens is the expectation over
  anticipated outputs.

**Open for us:** graded-strength ignition with bimodality, the central bottleneck, local–global and trace
conditioning under J-space ablation, on four model sizes. None of these has been run on open models.

## 2. Introspection and metacognition

- **Lindsey 2025, "Emergent introspective awareness"**
  ([paper](https://transformer-circuits.pub/2025/introspection/index.html)) studies concept injection.
- **Replications on open models:**
  - [Latent introspection](https://arxiv.org/abs/2602.20031): Qwen2.5-32B detects injections in its middle layers
    but says "no".
  - [Mechanisms of introspective awareness](https://arxiv.org/abs/2603.21396): a two-stage circuit that comes from
    preference optimization and is absent in base models.
  - [Content-agnostic](https://arxiv.org/html/2603.05414v2): models detect that something happened but cannot say
    what.
- **[Reality check (Singh, Linzen, Ravfogel 2026)](https://arxiv.org/abs/2605.26242).** Models cannot tell activation
  tampering from input changes. Input-only classifiers match their "self-prediction". Relabeled controls fall to
  chance. Current evidence does not establish introspection.
- **[Metacognitive neurofeedback (Ji-An et al., NeurIPS 2025)](https://arxiv.org/abs/2505.13763).** Models learn in
  context to report and to control the projection of their own activations onto a direction. Probe directions beat
  PCA directions, and the controllable space is about 32–128 dimensions. J-lens directions were never tested.

**Open for us:**
- Is the "metacognitive space" the J-space? Run neurofeedback on J-lens atoms against the same word's logit-lens and
  embedding directions and variance-matched PCs. Include a self-versus-other-model control so semantics alone cannot
  explain the result.
- Does an injection into the J-space get *identified*, while the same concept injected outside it only gets
  *detected*? This is a direct test of the content-agnostic critique.

## 3. Self-reference, self-models, attractors

- **[AE Studio: self-referential processing (Berg, de Lucena, Rosenblatt 2025)](https://arxiv.org/abs/2510.24797).**
  Sustained self-reference elicits structured experience reports. Suppressing deception and roleplay features in
  Llama-70B increases them.
- **[Assistant Axis (Lu et al. 2026)](https://arxiv.org/abs/2601.10387).** Persona drift is driven most by
  meta-reflection on the model's own processes and by emotional conversations. Activation capping stabilises it
  (Qwen3-32B, Gemma-2-27B, Llama-3.3-70B).
- **[Attractor states in multi-turn conversations (Ko & Geiping, ICML 2026)](https://arxiv.org/abs/2606.30571).**
  Self-play drifts to model-specific endpoints regardless of topic.
- **The Claude "spiritual bliss" attractor** (Claude 4 system card): self-dialogue converges to gratitude and
  consciousness themes.
- **[Attractor cycles under successive paraphrasing (Wang et al., ACL 2025)](https://arxiv.org/abs/2502.15208)**:
  repeated paraphrasing converges to 2-cycles.
- **[Repetition and self-reinforcement](https://arxiv.org/pdf/2410.13497)** (repetition neurons, induction heads) is
  the "banana" mechanism.
- **Self-models:**
  - Attention schema agents ([Farrell, Ziman & Graziano 2024](https://arxiv.org/abs/2411.00983)).
  - [Self-recognition directions in Llama-3-8B](https://arxiv.org/abs/2410.02064) and a
    [self-recognition fingerprint (ICML 2026)](https://arxiv.org/abs/2606.06315).
- **[Emotion concepts with causal function in Claude (Sofroniew et al. 2026)](https://anthropic.com/research/emotion-concepts-function).**

**Open for us:**
- What is in the J-space while a model claims experience under self-reference? Does it hold "pretend, fiction,
  roleplay" or "honest, actually"? The J-space paper scored eval awareness this way.
- Self-talk attractors traced in J-space: do they show up in the workspace before they show in the text, and what
  concepts do they hold? Your point that attractors are not bad in themselves, it depends what they represent, can
  only be tested where attractors carry content.

## 4. Criticality and dynamics

- **[Critical phase transition in LLMs (Nakaishi et al. 2024)](https://arxiv.org/abs/2406.05335).** Temperature
  drives a real transition near T ≈ 1. Correlations decay as a power law there, as in natural language. A
  [2026 follow-up](https://arxiv.org/abs/2604.00947) finds an extended critical phase.
- **[Lyapunov spectra of loop transformers (ICML 2026)](https://icml.cc/virtual/2026/79384).** Ouro-1.4B is mildly
  chaotic and Huginn converges.
- **IIT on LLM representations ([Li 2025](https://arxiv.org/abs/2506.22516))** is negative for feedforward
  transformers. Nobody has measured a recurrent latent loop.

**Open for us:** does the J-space show the transition too? The prediction is power-law decay of workspace
autocorrelation and a peak in workspace richness at the critical temperature, which ties your criticality point to a
measured phenomenon.

## 5. Substrate lessons for our own loop

- **[Filler tokens (Pfau et al. 2024)](https://arxiv.org/abs/2404.15758)** carry computation only after dense
  training. That explains why our placeholder loop held no meaning.
- **Coconut-style hidden-state feedback needs training.** Soft Thinking and SeLaR are training-free but feed back token
  mixtures.
- **Natively recurrent models (Huginn, Ouro)** are the honest substrate for a latent strange loop. They would need our
  own J-lens fit.

## Shortlist, in the order we run it

1. **Ignition battery** (Dehaene's decisive test): threshold, bifurcation at threshold, and the central bottleneck.
2. **Trace conditioning and local–global** under J-space ablation, against matched random ablation.
3. **Criticality in the J-space** across sampling temperature.
4. **Self-talk attractors** traced in the J-space.

Later candidates:

5. Neurofeedback on J-space directions, with a self-versus-other control.
6. Identification versus detection of injections inside and outside the J-space.
7. What the J-space holds during experience claims under self-reference.
8. A strange loop on a natively recurrent model (Huginn or Ouro) with a fitted lens.
