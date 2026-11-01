# Strange loop in J-space

**Learning to describe a loop’s possible changes**

Wiktor Tomasik · 2026 · [interactive experiment](https://huggingface.co/spaces/Pixedar/strange-loop-in-j-space) · [report](REPORT.md) · [data](https://huggingface.co/datasets/Pixedar/jspace-strange-loop)

A small trainable network repeatedly reads a language model’s internal activity. What it reads sets its next state, and that state changes what it reads next. This repository tests what happens when the network is also trained to describe how this feedback loop responds to changes in its own state.

The central comparison holds the text fixed and tries several starting states. Does the loop go to the same next state every time, or does its starting state make a difference? Some trained loops develop a strong dependence on their starting state. A control trained to describe another loop’s recorded responses remains close to one destination. Cutting the feedback into the reader largely removes the movements in the strongest case.

![A small reader summarizes language-model activity, sets the next control state, and reads again. Its attention can also bias the language model.](figures/self_describing_attention.png)

## How the experiment works

The language model’s weights stay fixed. Around it, three learned components form a loop:

1. A **reader** uses the current control state to weight the model’s activity at recent text positions.
2. A **summary** compresses that reading into 32 numbers. This is a learned vector, not a verbal description or an image.
3. A **state decoder** turns the summary into eight control numbers, which set the next read.

During training, a separate prediction network receives the summary and several candidate starting states. It learns to distinguish the loop’s possible updates from those of parallel copies. These copies share learned weights but read different text streams.

There is a shortcut in that task. If every starting state leads to one destination, identifying the destination is enough. The **centered-field objective** removes that common destination from the target. What remains is the pattern of differences between next states. Training can change both the description and the loop it describes. The prediction network shapes this learning; it is not used to choose the next state during operation.

## Results

**The training target changes the dynamics that develop.** In an initial comparison on a 0.6-billion-parameter model, 2 of 5 self-description runs sustained substantial state dependence without settling into a two-state oscillation. Three further runs starting from previously trained readers reached the state-dependent criterion during training. Controls describing another loop’s recordings remained near a common destination. Two runs on an 8-billion-parameter model also developed state dependence. These are small, related samples: the outcome depends on initialization and training history.

![Different starting states lead to nearly one next state in the control and to separated next states in a self-description run.](figures/state_dependence.png)

**The feedback through the reader causes most of the large movements in the strongest run.** Holding the reader’s query fixed reduces large steps from **26.6% to 1.2%** of evaluated steps. Disabling the separate route into the language model’s attention leaves **26.4%**. A large step is a distance greater than 0.5 between successive unit control vectors. The [interactive comparison](https://huggingface.co/spaces/Pixedar/strange-loop-in-j-space) shows the starting-state tests and these path interventions.

**The reader can affect the language model’s predictions.** Applying the reader’s attention weights as a bias inside the model changes **11–16% of its highest-probability next-token predictions** across five self-description runs. A token is a word or part of a word. Prediction loss changes by −0.14% to +1.51%, with negative values indicating improvement. The control describing another loop also changes predictions. These effects establish influence on processing; they do not establish improved performance caused by self-description.

**Feedback through the language model remains weak.** Interventions isolate two routes: the state changes the small reader, or it changes the language model’s processing. The second route produces less than 1% of the full starting-state-dependent variation in the five runs. Most of the measured dependence therefore travels through the small reader. A representation of the consequences of its own attention has not been demonstrated.

The [report](REPORT.md) gives every final-run result, measurement definitions and causal tests. Its strongest case has a normalized state-dependence score of **0.65**, compared with **0.002** for the control. The score compares two sums of squared variation; it is not a probability or a percentage of movement caused by the state. Predicted and measured response patterns correlate at **+0.81**, but the predictions have large scale errors. Pattern agreement is stronger than absolute forecasting accuracy here.

## Interpretation and open questions

The evidence supports a specific result: training a summary to distinguish the loop’s own possible updates can help produce dynamics that depend on its state. It does not yet show reliable development across starts, accurate forecasts used during operation, or lasting individuality among copies. Copies with one shared reader can converge when given the same text. Short changes in generated wording do not establish a stable personality. None of these measurements tests subjective experience.

A [text, image and audio experiment](followups/README.md) tests the same training objective with Qwen2.5-Omni. Readers given sampled image, audio and text positions develop state dependence, while readers restricted to recent text positions do not. However, changing the spoken direction affects their next states roughly 80–250 times more than reversing the image. State dependence alone is insufficient evidence that information from the modalities has been integrated. This experiment uses cached model activity and does not reproduce the full live feedback system.

## Other experiments in this repository

The name J-space comes from the project’s word-based reading of internal model states using the Jacobian lens. The attention-feedback system above learns its own numerical summary. The following experiments use different mechanisms and address distinct questions.

| System | Question and finding | Details |
|---|---|---|
| Frozen J-space feedback | What survives repeated reading and rewriting of a model’s state? Discourse and stance words persist, and the feedback favors a casual persona. Perturbed reasoning can invent a conversational past. | [Frozen loop](frozen_loop/) |
| State passed between reasoning segments | How does a self-prediction task shape memory? Distinguishing moments produces a rotating time code; distinguishing trajectories produces a different code. A forecast written back into the model can shape the state it predicts. | [Trained loop](trained_loop/), [stages of a thought](figures/stages_of_a_thought.png) |
| State controlling evidence selection | Can a compact state steer what is read? Editing the state redirects evidence selection when access is restricted; giving the model full text access changes the learned solution. | [Selection loop](selection_loop/) |

The rotating code across reasoning segments and the token-by-token two-state oscillation are different phenomena. The [research history](HISTORY.md) compares the mechanisms, records unsuccessful causal-model experiments and explains which interpretations the controls support.

## Explore and reproduce

- [Interactive experiment](https://huggingface.co/spaces/Pixedar/strange-loop-in-j-space): change a starting-state comparison and inspect measured responses and path cuts.
- [Report](REPORT.md): experimental design, results, limits and related work.
- [Methods and code](self_reference/): configurations, numerical settings, data provenance and reproduction instructions.
- [Behavior and multimodal experiments](followups/): prediction changes, generated text, perception controls and input interventions.
- [Data](https://huggingface.co/datasets/Pixedar/jspace-strange-loop): numeric recordings and learned interface weights. Base-model weights remain in their upstream repositories.
- [Archived report](history/REPORT_20261014.md): detailed results for the frozen, reasoning-segment and evidence-selection systems.

Code: **AGPL-3.0-only**. Author-owned text, figures and observations introduced in release 4: **CC BY-SA 4.0**. Earlier grants and third-party terms remain valid; see [LICENSING.md](LICENSING.md). Citation: [CITATION.cff](CITATION.cff). File checksums: `SHA256SUMS`.
