---
license: agpl-3.0
language:
- en
base_model: Qwen/Qwen3-8B
tags:
- interpretability
- attention-schema
- experimental
---

# c8 history-dependent attention reader

**Wiktor Tomasik · 2026-11-01**

[Code and report](https://github.com/Pixedar/jspace-strange-loop) · [Public explorer](https://pixedar.github.io/projects/self-awareness/attention-reader/explorer/index.html) · [Evidence dataset](https://huggingface.co/datasets/Pixedar/jspace-strange-loop)

This is one small reader and its online state, attached to a frozen Qwen3-8B backbone. It is a research mechanism for investigating how history changes selection, how a prediction of attention can influence attention, and what those internal differences do—or do not—carry into generated behavior.

The reader observes a 16-position window of layer-24 activity. An eight-number stance changes its selection query. The selected activity is compressed into a 32-number summary that determines a direction for turning the stance. A slower habit updates the stance-to-query map. An attention schema predicts selected-minus-average key content from a 16-number context description and adds that prediction to the query. The 256 query/key/value features are arranged as four heads of 64; they are not a list of 256 identified concepts.

The initial reader was trained earlier to distinguish copy dynamics. During the reported development lives, its weights and the language model's weights are fixed. Habit and schema updates have no desired personality labels. Different warm-up histories precede shared recorded input. In live operation, selected content can act before model layer 16, with the processed consequence read after layer 24.

The download contains reader weights, not an 8B language model. A developed controller snapshot separately contains stance, recent observations, habit, prediction statistics, calibration state and random-generator state. It does not contain a saved language-model context/cache. The seed-10/12 snapshots used in the expanded generation study are at reading 3,000. These are numerical observations derived from external text, not copies of its original passages; see `PUBLICATION_NOTES.md` for scope.

The reference has exact checked agreement with a fresh preserved-original replay and checked save/reload continuation. Causal studies compare own and other prediction maps, exchange/restoration, matched random interventions, frozen learning, context change and a simpler private running-average controller. The latter preserves much organization, so own-versus-other effects alone do not identify a rich self-model.

The studies distinguish the attention model's own-copy relationship from ordinary private memory, structured writing and arbitrary recurrence. They include longer sixteen-copy lives, same-context action interventions, recurrent controls through the same interface, fact-paired single-label tests, and guided response analysis. The central functional claim and its limits are explained in RESEARCH_REPORT.md and RECURRENCE_REPORT.md. Detailed interface attempts remain in the experimental appendices rather than additional default models.

One compact reader checkpoint is supplied. Saved histories, including the four selected longer-run states, and the untrained recurrent comparator are experimental inputs. They are not extra trained model versions. The release includes an offline explorer, an authored recorded demo, standard-library aggregate verification and a layout assembler for the live recurrent comparison. The model is a research mechanism; task-general personality, recognized causal ownership and subjective experience are not established. Software and reader weights use AGPL-3.0-only; new author-owned text, figures and numerical observations use CC BY-SA 4.0. External sources retain their own terms.


---
Wiktor Tomasik · measurements: 1 November 2026 · publication updated: 2026-11-01T23:52:27+01:00
