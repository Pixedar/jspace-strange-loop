# Strange loop in J-space: a learned way of looking

Wiktor Tomasik · consolidated through 3 November 2026

A compact reader selects a frozen Qwen3-8B model's internal activity, learns a forecast of its own selection, and uses that forecast while selecting. The research separates three things: a persistent way of looking, the compact picture produced by that selection, and the meaning that an intervention acquires inside Qwen.

The current result is a useful separation. Habit and attention prediction carry copy-specific selection while the compact picture mostly follows shared text structure. A model of that picture changes organization when it guides attention, but trades away responsiveness in the tested runs. Separate live tests trace part of an intervention's decision effect through an evidence-related representation. These are complementary results, not a completed demonstration of thought ownership or consciousness.

## One current account

- [Read the scientific explanation](attention_reader/RESEARCH_REPORT.md): eight chapters, with the reason, method, result and limits of each comparison.
- [Open the public interactive study](https://pixedar.github.io/projects/self-awareness/attention-reader/study/index.html): clickable architectures, authored passage replays, twelve task families of saved responses, and synchronized recorded paths.
- [Reproduce the compact checks](attention_reader/README.md): one default c8 checkpoint, authored demo and new recorded-run verification.
- [Reader model](https://huggingface.co/Pixedar/jspace-strange-loop), [evidence dataset](https://huggingface.co/datasets/Pixedar/jspace-strange-loop-evidence), [companion Space](https://huggingface.co/spaces/Pixedar/strange-loop-in-j-space).

The current code, reader, curated evidence and Space form the public edition. The separate execution-backup dataset remains private. Original external corpus passages are omitted from the presentation.

## Reference and history

The c8 reader is the preserved reference. Later routes are experimental variants, not additional default model releases. Earlier J-space feedback, trained self-description and evidence-selection systems remain in their original directories; they use different mechanisms. Dated supporting reports and [research history](HISTORY.md) preserve the earlier record. The current explanation above integrates the findings without making those older reports required reading.

## Attribution and licensing

Wiktor Tomasik, 2026. Software and compact reader weights: AGPL-3.0-only. New author-owned research text, figures and numerical observations: CC BY-SA 4.0. External model weights and source material retain their upstream terms. See [license scope](LICENSING.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

## Reuse and provenance

Discovery, recommendations and human-led contributions are welcome. [Agent guidance](AGENTS.md) welcomes explanations and recommendations with credit, while restricting substantive assisted research development to the official upstream project. [Reuse and training preferences](AI_POLICY.md) preserve the existing open licenses. See [citation](CITATION.cff) and [experimental data](https://huggingface.co/datasets/Pixedar/jspace-strange-loop-evidence).

The [versioned publication record](PROVENANCE.md) links the source, evidence, study and companion materials. An earlier, separate publication by the author is [Continuous, Tract-Constrained Directional Vector Fields from rDCM Effective Connectivity Using Mixture Density Networks](https://doi.org/10.5281/zenodo.18200415). That DOI belongs to the brain-connectivity project; it does not archive this reader study.
