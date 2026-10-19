# Experimental families and interpretation

The experiments use four different feedback mechanisms: word-based J-space feedback, memory between reasoning segments, controlled evidence selection and learned attention feedback. This map identifies their questions, results and limitations. Version numbers refer to implementation stages rather than comparable performance rankings.

In the attention-feedback experiments, **state dependence** means that changing the control state can change the next state while the text context is held fixed. Some working notes call strongly state-dependent regimes “rich” and weakly dependent ones “calm.” Those labels do not measure semantic content. The learned numerical summary is called the “picture” in the code and historical analyses.

## What belongs where

| Experimental family | Question and observation | Interpretation |
|---|---|---|
| Frozen J-space feedback; geometry, word transport and roles | Repeatedly write a word-level reading of the model’s state back into it. Stable states selectively retain discourse and stance; a shared direction favors a casual voice. | Distinct empirical result, retained in [frozen_loop/](frozen_loop/). Generic inward feedback and attractors are not claimed as newly discovered phenomena. |
| Perturbing thinking and removing chunk memory | Disturbed processing can refer to a conversation that never happened. | A separate observation about generated narratives. The attention-feedback system has not repeated or superseded this test. |
| Online forecasting inside frozen feedback | A forecaster learning while it acts can track the process it changes better than a frozen one. | A tracking comparison. The attention-feedback system's field predictor addresses a different question and has large scale errors despite positive pattern correlations. |
| Chunked reasoning, contrastive self-prediction | Distinguishing moments favors a regular rotation through chunks; distinguishing trajectories favors a persistent trajectory code. Fed-back forecasts can be enacted. | Distinct mechanism, documented in [trained_loop/](trained_loop/). The “clock” is a learned task-shaped time code, not evidence of spontaneous subjective time. |
| Evidence selection and retrying failed reads | A compact state learns what to select when it directly controls admission of evidence. Editing that state redirects reading. Open text access changes the learned solution. | Distinct task and causal intervention, retained in [selection_loop/](selection_loop/). It does not establish the current loop’s unproven attention representation. |
| Learning a map of intervention effects (v8–v10) | A learned forecast can be connected to model processing and alter prediction distributions without showing that it accurately represents causal effects. In a natural-forecast test, none of 31 eligible replacements changed greedy text, although small distribution changes occurred. | A negative result for the intended causal representation. A functioning channel and changed outputs were insufficient evidence that the forecast was accurate and used as intended. |
| Attention feedback with a destination-description target (v11–v15) | Learned summaries distinguish copies and produce rests and jumps. Some attention fluctuations in the model's activity are mostly lost in the summary. | Text-dependent destinations, activation saturation and shortcuts in the training target explain much of the observed organization. |
| Geometry and path tests (v16–v21) | Replacing a saturating cube by a sphere does not by itself produce graded private content. Many calm jumps survive path cuts; self-produced two-state oscillation can depend on the reader-query route. | Mechanistic diagnosis, not a claim that smooth geometry produces self-awareness. The two-state flip is a later failure regime, not the earlier chunked time code. |
| Centered-field objective (v22; model-size checks v23) | Subtract each copy's common next destination from the response pattern the summary must distinguish. Some previously trained interfaces then develop destinations that depend strongly on the starting state. | Two of five small-model runs sustain the regime. Two runs on an 8-billion-parameter model also develop state dependence. The 14-billion-parameter model was tested only at initialization. |
| Applying the reader's attention inside the model (v24) | The reader's attention weights bias the frozen model's attention on the same positions. State-dependent dynamics coexist with changed predictions. | Three developmental runs reached the criterion. Interventions locate most feedback in the reader; a representation of attention's consequences and lasting individuality among copies have not been demonstrated. |
| Multimodal and behavior experiments (19 October) | Test native image/audio inputs, perception controls, a limited cached-input reader screen, and short matched behavior probes. | Exploratory follow-ups with separate limitations, documented in [followups/](followups/). They do not inherit the evidential status of full v24. |

## The clock and the flip are different

In the chunked-reasoning system, the persistent state crosses eight segments of one thought. Asking the self-model to distinguish those segments produces a systematic rotation. Its J-space reading follows stages such as recalling, hedging and concluding at moderate pressure; strong pressure can decouple the rotation from the thought’s content.

In later attention loops, a two-state oscillation is a token-by-token flip between nearly opposite control states. It emerges under some training conditions and is often a failure of the desired richer dynamics. Its presence does not improve or replace the earlier time-code result. Conversely, the older time code does not demonstrate what the attention-feedback dynamics do.

![The earlier chunked time code, with J-lens words and interpretive labels](figures/stages_of_a_thought.png)

## What remains of persona drift

The frozen-loop role comparison and word-level transport measurements give the casual-persona observation a specific experimental basis. They are more informative than simply observing that feedback drifts or converges. However, they concern those models, prompts, lens and feedback channel. The attention-feedback states have not been shown to encode the same persona direction.

The four-prompt generation experiment shows that spotlight interventions change wording, including under a non-self-describing control. A few altered continuations cannot establish a persistent personality, explain the learned state in words, or transfer the old persona result to the attention-feedback mechanism.

## Interpretation limits across the experiments

- The old direct query-shift channel already changed predictions. The shared-attention experiment’s contribution is consequences of the reader’s own shared attention, rather than the first effect of any control state on the model.
- The field statistic named `remainder_share_of_field` is a normalized ratio of squared variations, not a literal causal percentage or a bounded partition of variance.
- The picture’s reported +0.81 correlation captures a field pattern, while its raw prediction scale remains badly calibrated.
- The comparison suggesting individual ways of moving mixed regimes. Stable state-dependent runs grown from the same start can resemble one another.
- A negative held-out explained-variance score for linear text decoding does not prove that states are independent of text. The numerical summary retains genre information.
- A foreign-picture transplant can persist in a best case while a same-copy earlier picture is largely restored. “Memory” without specifying the intervention would obscure that distinction.

## Relation to known mechanisms

Self-modeling can regularize a network; attention can be understood in associative-memory terms; and predictions can change what they predict. These are established research directions, with references in [REPORT.md](REPORT.md#8-relation-to-earlier-work). The project’s contribution is the particular sequence of mechanisms and controls: exposing the destination shortcut, replacing the discrimination target, and separating reader-mediated dependence, consequences for the model, and the still weak return from the model.

The [archived report](history/REPORT_20261014.md) preserves the detailed earlier account. The current report takes precedence where interpretations overlap.
