# One reader, many histories

A small reader selects information inside a frozen language model, learns a prediction of its own attention, and uses that prediction in subsequent selection. In live operation, selected content can change the language model and return through its later activity.

Start with [the interactive explorer](explorer/index.html) or [the main research report](RESEARCH_REPORT.md). The explorer provides separate recorded/live architectures, clickable module details, query-route diagrams for the recurrent controls, guided response comparisons, and synchronized measured paths through the reader and language model.

The research centers on a specific question: which effects depend on modeling this copy's attention, beyond private memory, structured steering or arbitrary recurrence? Exchanges of learned predictions reveal an ownership-sensitive relation in continuing attention. Matched recurrent controls and fact-paired tests then separate that relation from the language effects of the retained selector. The report explains the function of each comparison and what its result means.

There is one default c8 reader checkpoint. Saved histories and the untrained recurrent comparator are experimental inputs, not additional trained model versions. Qwen3-8B is downloaded separately for live experiments.

## Run the small recorded demo

Install PyTorch for your hardware and NumPy. The authored demo needs no 8B model:

```powershell
python reproduce.py --states demo/states.npz --out results/demo --steps 468 --warmup 0
```

It supplies identical histories to four copies. They should stay identical; this checks symmetry rather than demonstrating individuality. The recorded runner also supports supplied layer-24 observations, own/other prediction, exchange, frozen learning and resumption. See [the demo](demo/README.md) and [condition definitions](evidence/CONDITIONS.md).

`python recheck_results.py` recomputes the central information-test aggregates from the supplied recorded contrasts, without a GPU or a model download. [Publication and rerun notes](PUBLICATION_PLAN.md) explain how to assemble the live recurrent comparison using the supplied snapshots and pinned model loader.

## Follow the scientific explanation

- [Main report](RESEARCH_REPORT.md): the mechanism, questions, comparisons, figures and implications.
- [Attention model versus recurrence](RECURRENCE_REPORT.md): what differs from a generic information-driven recurrent system, including counterbalanced information use.
- [Actual response changes](PATTERN_BEHAVIOR.md): decisions, character information, continuity, causal claims, imagination and planning.
- [Why this architecture](ARCHITECTURE_CHOICES.md): which earlier observations motivated each component.
- [Detailed studies](NEW_RESULTS.md), [longer-run methods](SPECIFICITY_REPORT.md), and [coverage](REQUIREMENTS_AUDIT.md): source evidence, qualification outcomes and open aims.

The figures retain their data and include explanations of axes, comparisons and inference. The 120-concept catalogue is a set of defined questions, not 120 discovered traits. The three-dimensional axes are display projections of measured signals, with retention percentages and synchronized time; they are not named mental faculties.

## Release and provenance

Wiktor Tomasik · measurements: 1 November 2026 · publication updated: 2026-11-01T23:52:27+01:00

- [Public interactive explorer](https://pixedar.github.io/projects/self-awareness/attention-reader/explorer/index.html)
- [Reader model](https://huggingface.co/Pixedar/jspace-strange-loop)
- [Numerical evidence](https://huggingface.co/datasets/Pixedar/jspace-strange-loop)
- [Companion Space](https://huggingface.co/spaces/Pixedar/strange-loop-in-j-space)

GitHub and the three Hugging Face repositories are private at the owner's request at publication time; the website is public. External corpus passages are omitted from this explorer. Authored situations, generated responses and numerical measurements remain inspectable. Numerical histories retain their original provenance. See [license scope](LICENSING.md), [third-party sources](THIRD_PARTY_NOTICES.md), and [publication notes](PUBLICATION_NOTES.md).
