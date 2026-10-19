# Third-party sources

- [Hugging Face Transformers](https://github.com/huggingface/transformers), Apache-2.0. The Qwen layer computations use its model definitions and rotary-position helpers. Upstream software retains its license.
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), and other Qwen checkpoints identified in the experimental configurations. No base weights are distributed in this code repository.
- [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B), Qwen Research license; used only in the bounded multimodal follow-up. Consult that model’s license separately.
- Jacobian-lens method: Gurnee et al. (2026), [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html). The earlier systems use lenses from [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens), with the revisions and attribution in the archived report.
- Corpus sources: [WikiText](https://huggingface.co/datasets/Salesforce/wikitext), [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), and [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum). These texts are downloaded separately and are not relicensed or redistributed by this release.
- Earlier reasoning prompts use [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), under its upstream terms.

Figures use Matplotlib; numeric processing uses NumPy/SciPy; model execution uses PyTorch. The interactive viewer uses standard browser APIs without bundling a third-party renderer. Dependency licenses remain applicable. The software and data license split is specified in [LICENSING.md](LICENSING.md).
