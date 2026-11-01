# Strange loop in J-space: one reader, many histories

Wiktor Tomasik · measurements: 1 November 2026 · publication updated: 2026-11-01T23:52:27+01:00

A compact reader selects a frozen language model's recent internal activity, updates a persistent state, and predicts its own attention. That prediction helps shape what it reads next. In live mode, selected content can also change the language model and return through its later activity.

The current study asks a specific question: **which effects depend on modeling this copy's attention, beyond private memory, structured steering and arbitrary recurrence?** Exchanging learned predictors changes developed reading habits. Matched recurrent controls and fact-paired response tests separate this ownership-sensitive relationship from the influence of the retained selector. See the [main report](attention_reader/RESEARCH_REPORT.md) for why each comparison was performed and what it means.

## Explore the current study

- [Public interactive explorer](https://pixedar.github.io/projects/self-awareness/attention-reader/explorer/index.html): separate recorded/live architectures, module details, guided examples and synchronized measured paths.
- [Research report](attention_reader/RESEARCH_REPORT.md): the questions, methods, figures and interpretations.
- [Attention model versus recurrence](attention_reader/RECURRENCE_REPORT.md): controls for generic recurrent influence and information-conditioned behavior.
- [Response analysis](attention_reader/PATTERN_BEHAVIOR.md) and [architecture rationale](attention_reader/ARCHITECTURE_CHOICES.md).
- [Reproducibility package](attention_reader/README.md): one default checkpoint, an authored CPU demo, recorded numerical evidence and live rerun instructions.
- [Reader on Hugging Face](https://huggingface.co/Pixedar/jspace-strange-loop), [evidence dataset](https://huggingface.co/datasets/Pixedar/jspace-strange-loop), and [companion Space](https://huggingface.co/spaces/Pixedar/strange-loop-in-j-space).

GitHub and the three Hugging Face repositories are private at publication time. The personal website hosts the sanitized public explorer independently of their access settings.

## What the comparisons establish

A learned prediction can participate in a persistent way of reading. Content-aligned actions influence the model more than equally sized noise or rotations. Much organization and live influence also survive simpler private-memory and recurrent controls. Counterbalanced information tests show mostly softer existing answer preferences rather than new decisions. The contribution is a controlled separation of these mechanisms; persistent individuality, general personality and subjective experience are not established.

## Earlier systems

The earlier frozen J-space feedback, memory-between-reasoning-segments, evidence-selection and self-description training studies remain in [frozen_loop/](frozen_loop/), [trained_loop/](trained_loop/), [selection_loop/](selection_loop/), [self_reference/](self_reference/) and [followups/](followups/). They use different mechanisms and are not extra versions of the current default reader. The [research history](HISTORY.md) and [previous release account](history/before_attention_reader_20261101/REPORT.md) preserve their findings.

## Attribution and licenses

Wiktor Tomasik, 2026. Software and the compact reader weights are AGPL-3.0-only; new author-owned research text, figures and numerical observations are CC BY-SA 4.0. Earlier grants remain valid. External model weights and source texts retain their upstream terms. The public explorer omits external corpus passages. See [license scope](LICENSING.md), [third-party sources](THIRD_PARTY_NOTICES.md) and [publication scope](attention_reader/PUBLICATION_NOTES.md).
