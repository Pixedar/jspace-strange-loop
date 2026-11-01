# Why this architecture: decisions, failures and unresolved choices

This report explains the design of the preserved c8 reference. It does not propose replacing the active architecture-development line. Its evidence comes from the recorded architecture history, the 29 and 31 October syntheses, and the independent controls in this consolidation. Findings from different configurations are kept separate.

## The original obstacle was forgetting, not a missing label for identity

The early loop repeatedly compressed what it read into a new state. That sounds recurrent, but recurrence alone did not make the current state depend meaningfully on its history. When each new reading replaced the stance, the shared input rapidly pulled copies into the same operation. The state behaved more like a changing report of the text than an enduring contributor to the next reading.

The turned-stance update addresses this specific failure. The reading produces a direction, and the change in that direction rotates the existing stance. The update carries the previous stance forward instead of replacing it. This gives a later state a route of dependence on its earlier state.

Why not stop there? Preserved state can be a geometric consequence without an interesting learned organization. Tests without the intended self-reading path could preserve differences too. A trajectory can also settle into a fixed point or a simple alternation. The design therefore needs evidence that history builds content-sensitive differences, not just that a recurrent variable continues moving.

## Why add a habit

The habit stores a slower consequence of the copy's actual selection. When a stance and a selected pattern occur together, the stance-to-query connection changes slightly. The resulting habit changes what the same stance will select later. A copy's history therefore affects the *rule for reading*, not only its current state.

For c8, this made distinctive reading several times more persistent and more responsive to content than the turned stance alone. The update was capped so that accumulated habit could not grow without bound and dominate every other input. That cap is an engineering constraint; it does not itself explain what the habit comes to represent.

The alternative reader d8 is an instructive failure of generalization. Its pre-existing dynamics were already differently organized, and the same habit update competed with that organization instead of helping it. Thus “add slow plasticity” is not a universal recipe for richer behavior. Its benefit depends on the instability it is correcting in the particular reader. This is why the release reference specifies the exact c8 weights rather than advertising only a block diagram.

## Why add an attention schema instead of another memory variable

A habit can build a way of reading without representing that way as a prediction. The schema adds a different relationship: it predicts the copy's selected-minus-average key content from the current window, then uses that prediction in the selection query. This makes a learned account of the system's operation part of the operation being described.

The decisive tests were not the schema's size or the fact that it could be trained. A learned-but-unused schema could predict attention without affecting development. A schema trained on another copy's attention reduced distinctiveness. An exchange produced a directional change beyond a size-matched random intervention, and restoration often repaired the relation to an unchanged twin. Mixing one's own and another's map produced graded convergence. These tests establish an operational role for the represented information.

They also reveal a limitation: the schema mostly **keeps or strengthens differences that are already present**. It does not reliably create individual organization from an otherwise undifferentiated system. That is why an apparent self-amplification mechanism should not be described as spontaneous identity creation. It can reinforce the current selection rule, including a mundane preference for sentence boundaries.

The consolidation's running-average control makes this limit sharper. A private average retains much of persistence and also gives an own-versus-other effect. Context-sensitive prediction does improve the changing part of the forecast, but that extra predictive information has not yet been shown necessary for the full persistence benefit. The next useful question is when the richer predictor supports a control function that the private average cannot support.

## Why stability is more than a large persistence number

There are at least four different possibilities behind a stable trajectory: a fixed state, a repeating cycle, a learned filter that keeps responding to the input, or a more elaborate evolving organization. Their persistence scores can overlap. A publishable description must distinguish them.

The freezing study shows that the learned matrices can keep working without continued parameter updates. The context-change study shows that improving prediction fit need not increase persistence. Actual passage comparisons reveal that some strong cases are partly explained by formatting. Phase analysis warns that rapid two-reading motion should not be counted as richness. Together these findings shift the question from “how much movement or stability is there?” to “what relationship is maintained, across which inputs, and through which mechanism?”

The most stable individual is therefore not automatically the most informative one. Seed 10 offers strong persistence; seed 12 offers stronger results on limited word-group tests; fresh seed 27 is valuable partly because ordinary surface predictors explain a substantial part of its attention. A coherent release can preserve these as complementary cases rather than ranking them by a single mental-capacity scale.

## Why the action carries content

Early routes changed where the language model attended. In the tested forms, almost all of the resulting influence was common to copies. They could change overall attention or model predictions without carrying what was particular to a developed copy. Some apparent “world” differences also came from the sampled text diverging by itself, after which the copies merely followed different inputs.

The content route addresses both problems. It constructs an action from the difference between the copy's selected activity and a stance-free reference selection. The action therefore encodes what that selection adds. It is injected below the read layer so the model processes the intervention before its consequences return to the controller.

The complete live milestone combined stance, habit and schema with this route. Its criteria held together on the first text, while later texts showed partial replication and weaker organization. This is different from earlier failed communication tests that omitted habit and schema. A failure in a partial route cannot be used as a verdict on a complete configuration that was not running in that test. Conversely, success of a communication route alone does not establish the full developmental mechanism.

## Why J-space became relevant—and what its name does not solve

The prior J-part intervention asks which part of a content action carries a measured effect on the model's predictions. A sparse decomposition using J-lens-derived vocabulary directions isolated a relatively small part of the action that carried much of the copy-related meaning score. Removing that part greatly reduced that score; random-dictionary comparisons did not reproduce the same selective pattern. This makes the representation a promising communication interface.

It does not show that every vector in that space is a thought, that the space is a soul, or that it implements all functions of a global workspace. Nor does it show that a new learned adapter will preserve the old sparse intervention's operation. The new revision therefore treats a rank-limited J-derived basis and a trained action adapter as a new experiment with its own fidelity and behavioral controls.

The adapter's supervision is the reference action projected into that basis. No desired character, emotion or belief is supplied as a training target. The model backbone is frozen. The experiment asks whether an interface from stance, reading and predicted attention can learn to carry the reference signal on held-out situations. If removing the attention prediction leaves the same fit, we should say that the adapter did not demonstrate a necessary role for that input. A good reconstruction alone would still not establish workspace-wide access or improved self-modeling.

## Why a model of one's own influence is the next difficult component

Acting changes the input that comes back. Some prior influence predictors learned that actions tend to confirm a copy's current reading. Using the predictor reduced that self-confirmation in the reading summary but did not transform the continuing organization. This suggests the predictor learned a real, relatively steady consequence without capturing enough of the changing influence to regulate it strongly.

The key-versus-value counterfactual locates a more specific problem. Removing the actual action-induced change from what attracts attention recovers much of action's cost to consistency. Removing it from the selected values does not. The useful target is therefore not “subtract every consequence of my action” but “predict how my action will change what captures my attention.” Perfect counterfactual access was supplied by the experimenter; the learned key-level predictor has not yet reached that capability.

This is a concrete architectural lesson. A self-model can be accurate about a shared or steady effect while missing the private, changing part needed for control. A high average score can conceal the failure that matters. New predictors should therefore be evaluated on the specific consequence they are meant to regulate, against ordinary memory and shared-reader baselines.

## Why multimodal integration needs a different success criterion

Past multimodal tests could produce organized shifts while external progress memory accounted for them even when the neural self-state was removed. The mistake would be to rename those shifts an integrated presence without making the proposed mechanism necessary.

The new native audiovisual scenes improve the task: the answer depends jointly on a physical event and whether a character observed it. But the native model can already solve that relation. A useful added-controller result must show a contribution beyond native competence, sensory salience and ordinary memory. It should preserve the relation under paraphrase and altered surface cues, and show what is lost when a specific part of the added mechanism is removed or exchanged.

The present cross-model bridge failed its functional qualification. Continuing with that bridge would conflate a changed reader operation with an effect of multimodality. A newly trained native reader or a substantially better transfer is required. This is unfinished work, not a negative answer to whether multisensory systems can support richer organization.

## Why the stated width is not the used capacity

The revision's summary analysis finds that one direction accounts for most of the 32-number reading's variation in the selected cases. A causal follow-up replaces that summary with one of two shared earlier prototypes. Much of the measured organization survives in two cases, while a third is more affected. The prototype table is shared across copies, so it does not encode a private identity per copy. The other modules still retain substantial history.

This qualifies the architectural interpretation: a 32-number interface is not evidence for 32 independently organized meanings, and a persistent attention score does not require every intermediate representation to remain rich. A next design should demonstrate a function that the reduced controller loses, rather than increase a dimension count and infer increased mental richness. The complete findings and random-distortion comparison are in the new-results report.

## What this design is ready to claim

The preserved design provides an inspectable test bed for history-dependent selection, partial prediction of attention, causal use of that prediction, and recurrent influence on a language model. Its most informative findings include the conditions under which these properties separate. It is not yet a demonstrated humanlike mind, a task-general personality, or a combined implementation of every property discussed in its history.

The architecture remains the reference because its causal interventions are clearer than its alternatives, not because every score is largest or every earlier problem has been solved. The companion requirement audit names the remaining work. Any handoff to the active design line should contain only the relevant measured findings and limitations, rather than silently replacing its current direction.

### Source map

The source snapshots and SHA-256 manifest are in `provenance/`. Principal records: `research_v12/ARCHITECTURE_HISTORY.md`, `SYNTHESIS_29OCT.md`, `SYNTHESIS_31OCT.md`; sections 41–59 cover habit/schema development and controls, 60–64 early routes, 65–76 complete live and J-part tests, and 77–80 influence/echo tests. Consolidation evidence: `../results/mean_analysis.json`, `schema_baseline_analysis.json`, `freeze_analysis.json`, `transfer_analysis.json`, `phase_diagnostic.json` and `native_analysis.json`. Section numbers refer to the source research log, not new experiments in this revision.


## What the recurrent comparison changes about the design rationale

A learned attention target is the distinguishing feature of the forecast, not recurrence itself. The new 64-state random recurrent comparator responds to the same context and writes through the same reader interface. Preserving much of the language effect when its direction replaces the forecast shows that the remaining stance/habit selector must receive part of the explanatory credit. Reference magnitude streams are retained in that control, so the result specifically concerns detailed directional information.

In the counterbalanced fact-choice assay, the learned selector mainly softens already strong answer margins while changing few choices. Most of that pattern survives recurrent replacement of the forecast direction. Whose forecast is used changes the strength more subtly; it does not become an automatically superior reasoning policy. Together with the key-return intervention, this motivates a precise next question: can a model of the controller's own impact regulate information use in a way a private average or arbitrary recurrent direction cannot? That is a proposed test, not a silent change to the active architecture.

The longer unusual case also needs only a small set of earlier reading-summary prototypes for good reconstruction. This limits claims made from the 32-number summary's visual motion, while leaving the other memory-bearing components to be tested in their own causal roles.


---
Wiktor Tomasik · measurements: 1 November 2026 · publication updated: 2026-11-01T23:52:27+01:00
