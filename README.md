# Strange loop in J-space

**Can a system’s description of itself help create the way it changes—and can that description matter to what a language model does?**

Wiktor Tomasik · 2026 · [interactive study](https://huggingface.co/spaces/Pixedar/strange-loop-in-j-space) · [full report](REPORT.md) · [data](https://huggingface.co/datasets/Pixedar/jspace-strange-loop)

The newest experiments build a small self-describing attention loop around a language model whose weights stay fixed. Its current state changes how it reads the model’s activity; that reading forms a picture, which becomes its next state. During development, the picture learns to distinguish the loop’s own possible moves from other copies’ moves. The description can reshape what it describes.

The important change was to remove an easy answer. Earlier pictures could identify a copy simply by naming where it would end up. The new game subtracts that destination and asks about **how the destination changes with the starting state**. From an already organized start, some loops develop substantial dependence on their own state, and their pictures capture its pattern. Giving the same reading a channel into the model’s attention then makes it affect the model’s predictions.

![The reader forms a picture and a control state, which changes the next read. A shared spotlight also focuses the frozen model’s attention.](figures/self_describing_attention.png)

## What the newest experiments establish

- **Removing the destination shortcut changes what develops.** In the initial small-model comparison, 2 of 5 self-describing runs sustained the state-dependent regime. In the later spotlight batch, all three developmental starts reached it. Recorded-other-description controls stayed close to one destination. Two 8B runs also developed state dependence; 14B was checked only at initialization. At initialization, all tested sizes had almost no signal for the new game.
- **The dependence runs through the reader.** In the strongest later run, cutting the state’s influence on the reader reduced large jumps from 26.6% to 1.2% of steps. Cutting the model-attention route left them at 26.4%. A transplanted picture could still influence later pictures, with about 57% of the normalized difference remaining after 20 tokens in this run.
- **The reading now has measurable consequences.** A shared spotlight changes 11–16% of top next-token predictions in the five self-describing runs trained with their own spotlight. Next-token loss changes by −0.14% to +1.51% on the held-out texts. A control describing another run also changes predictions and slightly improves this loss: acting on the model is not exclusive to self-description.
- **The loop still scarcely reads back what its attending did.** The route through the model carries less than 1% of the full probe-dependent variation in those five runs. The loop controls attention without a demonstrated representation of the consequences of that control.

![Measured state dependence, top-prediction changes and next-token-loss changes are separate quantities.](figures/consequences.png)

The strongest developmental run has a normalized state-dependence measure of **0.65**, compared with **0.002** for the recorded-other control, and a picture–field pattern correlation of **+0.81**. The first number is a ratio of squared variations, **not a probability or a fraction of steps**. The correlation describes a pattern, **not a calibrated prediction**: the field head’s unscaled squared errors remain large. Definitions and every final-run result are in the [report](REPORT.md) and [result tables](self_reference/results/verified_summary.json).

## What the states mean—and what remains missing

Earlier calm loops carried information about text genre. In the state-dependent loops, a simple linear decoder of text features generalizes poorly to held-out copies; this does not establish independence from the text, and the picture still carries genre information. The loop’s dynamics have become a larger part of what is represented.

The evidence for individuality is weaker than it first appeared: two stable rich runs from the same start move similarly, while copies sharing one interface can converge on a shared text. Short follow-up generations change under the spotlight, but the non-self-describing control changes them too. No stable personality unique to self-description has been established.

The descriptive field head shapes the interface during learning; it is not used online to choose a move. Useful online use of those predicted possibilities, a representation of the effects of its own attention, and persistent individuality within one system remain open. These are functional experiments; none measures subjective experience.

A bounded [native text–image–audio follow-up](followups/README.md) transfers the centered-field mechanism when the reader can sample modality positions. Its state dependence reaches about 0.51 in two seeds, while recent-text-only readers remain nearly constant. Yet speech drives the resulting fields far more than the image does. A rich loop is therefore not, by itself, evidence of multimodal integration. The failed beep controls, matched comparison and native input interventions are all retained.

## Earlier results still worth keeping

The project began with J-space: reading model states as weighted mixtures of word directions using the Jacobian lens. The newest learned picture is a different representation. The older systems answer questions the new one has not replaced:

| Earlier system | Distinct result | Where to read it |
|---|---|---|
| Frozen J-space feedback | The feedback favors self-reproducing discourse and stance words; contexts drift toward a casual persona. Perturbed thinking can invent a conversational past. | [Frozen loop](frozen_loop/) |
| A state carried between reasoning chunks | Discriminating moments produces a rotating time code; discriminating trajectories produces a different code. A fed-back forecast can shape the state it predicts. | [Trained loop](trained_loop/), [stages of a thought](figures/stages_of_a_thought.png) |
| A state that selects evidence | With a direct reading channel, editing the state redirects which item is read; open text access changes what the state learns. | [Selection loop](selection_loop/) |

The rotating code belongs to the chunked reasoning experiment. It is not the two-state flip seen in some later attention loops. The [research history](HISTORY.md) explains the branches, negative results, corrections, and relation to earlier literature.

## Code, evidence and reproduction

- [`self_reference/`](self_reference/): the current mechanism, methods, portable training and analysis code, final-run measurements, and artifact provenance.
- [`interactive/`](interactive/): the recorded-data viewer used by the [Hugging Face study](https://huggingface.co/spaces/Pixedar/strange-loop-in-j-space).
- [`followups/`](followups/): bounded multimodal and behavior investigations, with their controls and limitations.
- [`history/REPORT_20261014.md`](history/REPORT_20261014.md): the earlier detailed report; the new findings and corrections are in the current [REPORT.md](REPORT.md).

Start with the [reproduction guide](self_reference/README.md). Numeric recordings and learned interface weights are hosted in the linked dataset; the frozen language-model weights remain at their original model repositories. The figures and viewer are derived from saved measurements.

Code: **AGPL-3.0-only**. New author-owned text, figures and numeric observations: **CC BY-SA 4.0**. Earlier releases and third-party material retain their applicable terms; see [LICENSING.md](LICENSING.md). Cite: [CITATION.cff](CITATION.cff). `SHA256SUMS` records the published files’ hashes.
