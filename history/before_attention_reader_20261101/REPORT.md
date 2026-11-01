# Self-description in a language-model feedback loop

Wiktor Tomasik · 19 October 2026

## Abstract

A trainable reader repeatedly summarizes a frozen language model’s internal activity. Its summary sets a control state, which changes the reader’s next query. We train a second network to describe how this loop would update from alternative control states. Subtracting the common destination from its training target removes a shortcut: the description must distinguish differences between possible updates rather than merely identify where a copy goes.

Starting from previously trained interfaces, some runs develop substantial dependence on the control state. A control describing another run’s recorded responses remains near a common destination. In the strongest developmental run, holding the reader’s query fixed reduces large steps from 26.6% to 1.2%. Giving the reader a channel into the language model’s attention changes predictions, but feedback through model processing contributes little to the loop’s state dependence. The results concern learned feedback dynamics. They do not establish calibrated self-prediction, persistent individuality or subjective experience.

## 1. A reader that changes its next reading

A **frozen model** is a language model whose weights are held fixed. Here it processes ordinary external text, one token—a word or part of a word—at a time. A small trainable interface reads its internal activity.

The interface has an **eight-dimensional control state**: eight numbers normalized to length one. A learned **reader** turns that state into a query over the model’s activity at the last 16 text positions. Its attention is a set of weights specifying how much to read from each position. The weighted reading becomes a **summary**, a 32-number vector normalized to length √32. A decoder maps the summary to the next control state, completing the loop.

Four copies run in parallel. They share the same interface weights and frozen model but maintain separate states, attention caches and text streams. A training run is one history of these shared interface weights. The four copies are not four independently trained systems.

![The reader, summary and control state form a feedback loop around a frozen language model.](figures/self_describing_attention.png)

### Describing possible updates

At a fixed text context, evaluate the interface from several candidate control states. The resulting map from starting states to next states is its **response field**. This is a measured function, not a spatial field or an interpretation of what a state means.

A separate prediction network receives the summary and a candidate starting state. It predicts the corresponding update. Training asks the summary to distinguish its copy’s field from fields measured in other copies. The distance-based discrimination loss can change both the description and the interface that produces the target field. A control instead describes another run’s recorded field, which its own learning cannot change.

An uncentered field allows a shortcut. If every starting state leads to one destination, that destination can identify the copy even though the next state scarcely depends on the starting state. The **centered-field objective** subtracts each copy’s mean destination across candidate states. It compares only the differences that remain, centering the predicted field in the same way.

The prediction network affects training. During operation, the summary decoder determines the next state directly; the field predictions are not consulted. Thus the experiment tests how self-description shapes development, rather than whether a running system uses explicit forecasts to select its actions.

### Applying the reading inside the language model

The reader’s attention weights can also bias the frozen model’s attention on the same recent text positions. We call this optional channel the **shared spotlight**. A position receiving more weight from the reader is favored inside the model as well. This makes the reader relevant to the model’s next-token prediction, allowing prediction loss to train the interface through that channel.

For R retained positions and reader probability p, the additive attention-score bias is 1.5 × clip(log(Rp), −log R, log R). In the 0.6-billion-parameter model it is applied in layers 12–18, using zero-based indices. The direct attention-query shift used in separate experiments is disabled. The biased layers calculate attention scores explicitly in float32 because the fused kernel did not support the required gradient. Model weights remain fixed. Exact dimensions, cache handling and precision are in [METHODS.md](self_reference/METHODS.md).

## 2. Comparisons and sampling

The centered-field comparison without the shared spotlight contains five self-description runs on Qwen3-0.6B, two controls describing another run and one uncentered-objective control. Two of the five self-description runs sustain substantial state dependence without persistent two-state oscillation. Two self-description runs on Qwen3-8B also develop state dependence, with an uncentered-objective control for comparison. Initialization checks on 0.6B, 8B and 14B models all show little starting-state dependence. Development at 14B was not tested.

These successful runs begin from interfaces trained with the uncentered position objective. At random initialization, destinations barely vary across starting states, leaving little signal for the centered task. Success from a trained start should not be generalized to success from scratch.

The shared-spotlight batch contains nine 3,000-token continuations on Qwen3-0.6B: two state-dependent runs each continued with and without the spotlight; three developmental runs from previously trained interfaces with the spotlight; one continuation receiving a neighboring copy’s attention; and one interface describing another run. Two jobs that exhausted memory were rerun to completion. The completed records, rather than the infrastructure failures, enter the analysis.

All three developmental runs reached the operational state-dependence criterion at some checkpoint. This does not imply that every continuation maintained that behavior. One continuation partly entered a two-state oscillation. The samples share starting interfaces and are too small to estimate a general success rate.

Frozen evaluations hold the learned weights fixed and use matched external text streams. Field-description summaries evaluate 900 tokens, probing every five tokens in the last two thirds. The separate fixed-probe recordings used in the interactive comparison contain 12 candidate states, tested every ten tokens from positions 300–890, for all four copies. Spotlight comparisons use 600-token streams and summarize after token 30. These analyses use different sampling and normalization; their statistics are not interchangeable.

Tokens and checkpoints within a run are dependent observations. Many token measurements do not constitute many independent training replications. The displayed developmental run was selected for its strong result; the table below includes the other final runs.

## 3. What state dependence means here

Let s be a candidate control state and F(s) its measured next state. Its displacement is D(s) = F(s) − s. Subtracting the mean destination over candidates gives F(s) − mean_s F(s).

The saved statistic `remainder_share_of_field`, called **normalized state dependence** here, is:

    sum ||F(s) − mean_s F(s)||² / sum ||D(s) − mean D||²

The sums cover sampled tokens, copies and probes. The numerator centers destinations across probes separately for each token and copy. The denominator centers displacements using their global mean. This ratio measures starting-state-dependent destination variation relative to displacement variation. Because the centering differs, it is not a bounded variance decomposition. A value of 0.65 does not mean that 65% of steps or 65% of the future is caused by the current state.

The strongest developmental run scores 0.6519; the recorded-other control scores 0.00205. Correlation between the predicted and measured centered fields is +0.8114 in that run. This indicates agreement about the response pattern. The unscaled explained-variance score is −2609.4: large squared errors remain because the predictions are badly scaled. Positive correlation therefore does not establish accurate vector forecasts.

![Measured next states from matched candidate starting states.](figures/state_dependence.png)

The [interactive experiment](https://huggingface.co/spaces/Pixedar/strange-loop-in-j-space) avoids projecting these states onto a plane. For each of the 12 candidates, it shows the distance of the next state from a selected reference candidate’s next state. Distances use all eight control coordinates and range from 0 to 2. The selected context is an illustration; the recordings provide 60 contexts and four copies for inspection.

## 4. Final-run results

“Developed A/B” denotes a previously trained starting interface; the seed distinguishes continuations from that start. “Continued A/B” begins from a state-dependent interface. The recorded-other control predicts another run’s responses. Each row below applies the evaluated interface’s own attention bias, including the control.

| Run | Normalized dependence | Pattern correlation | Unscaled explained variance | Top prediction changed | Loss change |
|---|---:|---:|---:|---:|---:|
| Developed A / seed 0 | 0.6519 | +0.811 | -2609.4 | 12.28% | +0.605% |
| Developed A / seed 2 | 0.5387 | +0.764 | -2562.5 | 10.92% | +0.296% |
| Developed B / seed 1 | 0.1634 | +0.479 | -7983.8 | 11.97% | -0.141% |
| Continued A | 0.3562 | +0.674 | -47976.6 | 16.01% | +1.513% |
| Continued B | 0.4685 | +0.719 | -18014.7 | 12.68% | +0.588% |
| Recorded-other description | 0.0021 | -0.033 | -97354.8 | 8.55% | -0.259% |

Prediction change counts positions where the highest-probability next token differs from the no-spotlight comparison. Both evaluations receive the same preceding text, so later differences are not caused by accumulated generated-text divergence. Loss change is the relative change in next-token prediction loss; negative values are better.

Across the five self-description runs, top predictions change at 10.92–16.01% of positions and loss changes by −0.141% to +1.513%. The recorded-other control changes 8.55% of predictions and lowers loss by 0.259%. Influencing predictions is therefore possible without developing substantial state dependence. These tests do not establish a performance advantage caused by self-description.

![State dependence, prediction changes and prediction loss measured separately.](figures/consequences.png)

## 5. Causal interventions: where the loop closes

A large step has Euclidean distance greater than 0.5 between successive unit control states. In the strongest developmental run, such steps occur at 26.64% of evaluated positions. Holding the state supplied to the reader’s query constant reduces this to 1.18%. Cutting the state’s route into the language model’s attention leaves 26.41%. These interventions keep trained weights fixed; they isolate the reader route as the main cause of these large movements.

Replacing a summary with one from another copy can also affect subsequent summaries. The normalized difference from the unperturbed trajectory is 0.588 after one token, 0.590 after five, 0.571 after twenty and 0.260 after forty in this run. Restricting the replacement to the reader route gives similar results; the model-only route produces differences around 0.001–0.003. Replacing a summary with the same copy’s earlier summary is largely undone. Persistence depends on both the run and the intervention and does not by itself demonstrate semantic memory.

![Path cuts and the persistence of a summary replacement.](figures/causal_paths.png)

An independent response-field comparison also separates the two routes. The ratio of model-only to full starting-state-dependent variation is below 0.007 in each of the five self-description runs. This compares intervention magnitudes; the routes are not an additive partition of causal responsibility.

The weak return can be followed through the interface. In the strongest run, changing the state changes the model’s read-layer activity by 0.1036 relative to its baseline norm, the summary by 0.00113, and the next control state by 0.000163. These are relative changes in different vector spaces. They show attenuation along this measured route, rather than an information-conservation calculation.

## 6. Meaning, attention and individuality

Decoders test which observable features can be recovered from recorded states. They are trained on three copies and tested on the fourth, with time-shifted controls. The [meaning table](self_reference/results/meaning/table.md) identifies the recordings and dynamical regimes used for each analysis.

States in loops with little dependence on their own starting state carry text-genre information. In the initial state-dependent small-model runs, a linear predictor from text features has negative held-out explained variance for the control state: it fails to generalize to the held-out copy. This does not establish independence from text. Genre information remains recoverable from the summary, and jumps are more likely after line breaks.

The reader can control how concentrated the model’s attention is. However, the summary generally adds little predictive information about attention beyond the text and control state. These analyses have not demonstrated a representation of the effects of the loop’s own attention—an attention schema in the sense discussed below.

Different training histories can produce different dynamics, but the evidence for persistent individuality is limited. A comparison that suggested distinct identities mixed dynamical regimes; two stable state-dependent runs from the same trained start correlate about +0.69. Copies sharing an interface can converge on the same text.

A separate test compares the per-token loss changes induced by the spotlight. Median correlation within one run across checkpoints 500 tokens apart is 0.614; between runs it is 0.304. The predefined twofold gap is narrowly missed, by about 0.0052, and a control without self-description gives a similar result. This does not establish behavioral individuality caused by self-description.

The [generation experiment](followups/README.md) compares three saved interfaces with an unbiased model on matched prose and four prompts. Predictions and generated wording change under both self-description and control interfaces. The sample does not support a stable-personality claim, and generated self-reports are not treated as measurements of hidden states.

## 7. Limits of the interpretation

- State dependence concerns a defined response-field statistic; it is not a causal percentage.
- Predicted and measured response patterns can correlate despite severe errors in prediction scale. The running loop does not consult the field predictions.
- The reader’s shared attention affects model predictions. A separate direct query-shift channel already changed predictions in other experiments, often at about a fifth of positions. The shared-attention channel tests a different mechanism.
- Differences between runs do not establish individuality when regimes differ. Negative linear decoding does not establish independence from text.
- A summary replacement can persist without demonstrating a semantic memory. The effect varies with the source of the replacement and the receiving run.
- Shared starting interfaces, small samples and dependence between observations limit generalization. None of the measures tests consciousness or subjective experience.

## 8. Relation to earlier work

The project asks how a representation changes the process it represents. Self-modeling, associative-memory dynamics and predictions that alter their targets all have substantial prior literatures. The specific experiment here removes a destination shortcut from a field-description task, then distinguishes feedback through a learned reader from effects on—and feedback through—a frozen language model.

- The Jacobian lens and J-space come from [Gurnee et al., *Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html). The earliest experiments use that lens; the attention-feedback system’s learned 32-number summary is learned independently and should not inherit the lens’s interpretive guarantees.
- Attention already has connections to associative-memory and fixed-point dynamics, including [Ramsauer et al., *Hopfield Networks Is All You Need*](https://arxiv.org/abs/2008.02217). Resting places or attractors alone are therefore weak evidence of a distinctive reflexive mechanism. The earlier feedback’s word-level selection and persona bias remain separate empirical observations.
- Learning to predict oneself can simplify the system, as studied by [Premakumar et al., *Unexpected Benefits of Self-Modeling in Neural Systems*](https://arxiv.org/abs/2407.10188). Predictability or successful discrimination alone need not imply a faithful self-description.
- A prediction changing its target is a broader phenomenon; [Perdomo et al., *Performative Prediction*](https://arxiv.org/abs/2002.06673) formalize such feedback in a different setting. The earlier fed-back forecast results are related in this sense, rather than proof that influencing one’s predicted future is unique to this project.
- [Wilterson and Graziano’s attention-schema agent](https://pmc.ncbi.nlm.nih.gov/articles/PMC8379943/) explicitly distinguishes a model of attention from attention control. The attention-feedback loop has not demonstrated the corresponding representational role.


The [research history](HISTORY.md) explains the separate frozen-feedback persona results, invented-past behavior, rotating code across reasoning segments, evidence selection and unsuccessful causal-map experiments. The attention-feedback system does not repeat all of those tests.

## 9. Multimodal transfer and reproduction

A separate experiment applies the centered-field objective to cached Qwen2.5-Omni activity. Two readers sampling image, audio and text positions reach normalized dependence about 0.51 on held-out colors. Matched readers restricted to recent text and controls describing another run remain close to a common destination. Yet reversing image panels changes next states only about 0.4–1.3% as much as swapping the spoken direction. The development of state dependence is not evidence of balanced use of the modalities.

This experiment omits live prediction-loss training and recurrent feedback through the native model. The [multimodal report](followups/README.md) gives perception controls, the failed beep task, matched reader comparisons, input interventions and a one-step attention-bias test. It is a limited transfer experiment, not a reproduction of the complete live attention-feedback system.

The [reproduction guide](self_reference/README.md) covers code, configurations, numerical settings, interface checkpoints, recordings and corpus provenance. Figures and the interactive comparison are generated from saved measurements. Some field-prediction statistics were saved as summaries without complete prediction arrays; verification distinguishes recomputed measurements from checked summaries. Missing source artifacts and the scope of the training smoke test are stated in the guide.
