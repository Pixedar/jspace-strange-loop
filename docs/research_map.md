# Related work for the J-space loop

Only work that bears on a latent self-referential loop in the J-space.

## Latent recurrence

- **Verbalizable Representations Form a Global Workspace in Language Models** (Gurnee, Lindsey et al. 2026;
  https://transformer-circuits.pub/2026/workspace/index.html). The J-lens and the J-space. The dictionary is
  overcomplete and the sparse decomposition is not unique, so word labels are an imperfect window.
- **Looped transformers under the J-lens** (https://arxiv.org/abs/2609.01924). A workspace forms in Ouro and Huginn,
  but recurrence restricts access: Ouro rebuilds it every loop, and in Huginn it can only be reached within about two
  steps.
- **Lyapunov spectra of loop transformers** (ICML 2026; https://icml.cc/virtual/2026/79384). Ouro-1.4B is mildly
  chaotic and Huginn converges. These are natively recurrent models, the stronger substrate for a latent loop.
- **Filler tokens** ("Let's think dot by dot", Pfau et al. 2024; https://arxiv.org/abs/2404.15758). Dots carry hidden
  computation only in models trained densely to use them. This explains why the placeholder loop holds no thought.
- **Hidden-state feedback.**
  - Coconut (Hao et al. 2024) feeds the last hidden state back as the next input and needs training.
  - Soft Thinking (https://arxiv.org/abs/2505.15778) is training-free, but feeds back mixtures of token embeddings
    instead.

## Self-models and prediction about prediction

- **Attention-schema agents** (Farrell, Ziman & Graziano 2024; https://arxiv.org/abs/2411.00983). A learned model of
  the agent's own attention helps control it, and mostly helps in tasks about other agents' attention.
- **Self-modeling** (Premakumar et al. 2024). Predicting one's own states makes a network simpler and easier to
  predict. In the ring, feeding back one's own prediction did the same (collapse).
- **Metacognitive neurofeedback** (Ji-An et al., NeurIPS 2025; https://arxiv.org/abs/2505.13763). Models learn in
  context to report and steer projections of their own activations, within a "metacognitive space" of about 32–128
  dimensions.
- **Precision in predictive processing** (Feldman & Friston 2010, "Attention, uncertainty, and free-energy").
  Attention is the weighting of prediction errors by their expected reliability (precision): a prediction about
  prediction errors. This is the second rung of the "difference about difference" ladder, and the basis for the
  precision-weighted and second-order error conditions proposed in `research_report.md`.

## Prediction error as a drive

- **Curiosity from prediction error** (Pathak et al. 2017, intrinsic curiosity module). Agents are rewarded for
  states their forward model predicts badly.
- **The noisy-TV problem** (Burda et al. 2018, large-scale curiosity). Unpredictable noise produces endless surprise
  without anything to learn.
  - The ring is different. It writes the error back into the stream instead of rewarding the search for surprise.
  - The warning still applies: surprise is not meaningful novelty.
  - Normalising the error gives even noise a full-size push.

## Dimensionality, criticality, meaning

- **Effective dimensionality** of neural activity measured by the participation ratio (e.g. Gao et al. 2017, "A
  theory of multineuronal dimensionality"). High dimensionality does not by itself mean structured or useful dynamics.
- **Criticality in language models** (Nakaishi et al. 2024; https://arxiv.org/abs/2406.05335): a temperature-driven
  phase transition with power-law correlations near the critical point. A 2026 follow-up
  (https://arxiv.org/abs/2604.00947) finds an extended critical phase instead.
- **Checking meaning in latent trajectories.** Word labels from the lens cannot vouch for themselves. Use an
  independent semantic measure (embeddings), matched controls, and causal tests: perturb an apparent concept
  direction and check that a later readout changes as predicted.
