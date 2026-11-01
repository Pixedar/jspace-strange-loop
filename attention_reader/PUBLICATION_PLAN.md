# One reference, three ways to inspect it

The release contains one default c8 reader, not a collection of competing trained versions. Saved controller histories and the untrained recurrent comparator are experimental inputs. The final raw reader tensors match the original reference exactly; unused historical training components are omitted from the compact checkpoint.

The GitHub edition provides the source, portable authored recorded demo, main scientific report, architecture rationale, complete figure explanations and compact measurements. Hugging Face can host the same checkpoint and model card. The portable explorer can be used as a static Space from the same version of the evidence. No account, API key or rental is required to view it. The release locations are linked from README.md.

## Read or reproduce without a large model

Open `explorer/index.html`. Run the root recorded demo as documented in README. `recheck_results.py` recomputes the central family-paired information-test means from stored contrasts using only Python's standard library. Numerical measurements, authored situations and the output texts remain inspectable in `evidence/revision/`.

## Rerun the live recurrent comparison

`prepare_recurrent_rerun.py --out D:/your_separate_run --minutes 90` assembles the executed research layout and sets a finite run deadline. It does not rent hardware, install dependencies, execute a model or upload anything. Invoke it immediately before a rerun. The layout must be mounted at `/workspace/consolidation`, with a writable cache at `/workspace/hf`.

The measured stack uses PyTorch 2.7.1 with CUDA 12.8, Transformers 5.8.1, Hugging Face Hub 1.15.0, Tokenizers 0.22.2 and Safetensors 0.7.0, plus their ordinary runtime dependencies. Keep the qualified CUDA PyTorch installation when installing the smaller dependencies. The original runs used a 48 GB GPU. Qwen3-8B downloads roughly 16 GB of model weights; allow additional space for the runtime and cache. Model revisions are pinned in the supplied loader.

From that layout, `python cloud_queue.py` runs the sixteen paired-context and repeated-response comparisons. `python inputs/information_choice.py` runs the counterbalanced single-label assay. Run them sequentially on one GPU. The deadline is an experiment limit; it does not shut down a manually rented machine. The private research execution used additional provider-side and local collectors with independent destruction guards, intentionally excluded from the publication package because they are account operations.

The reference snapshot and recurrent state allow these runs without redistributing the large recorded-corpus archive. They contain numerical observations derived from that history. License scope and source attribution are documented in LICENSING.md and THIRD_PARTY_NOTICES.md. External corpus strings are omitted from the published explorer; authored prompts and generated responses are retained. The full private evidence archive remains separate.

The model card and coverage audit identify the claims supported by this reference and the open research aims. A workspace adapter or cross-model bridge is not silently substituted for the reference. Routine engineering attempts remain in the methods appendix; the main report centers on informative mechanisms and their meaning.

## Hosting layouts

The GitHub attention_reader/ directory, Hugging Face dataset attention_reader/ directory, model checkpoint and static Space use the same measured version. The public website serves the sanitized explorer independently of private repository access.

Wiktor Tomasik · measurements: 1 November 2026 · publication updated: 2026-11-01T23:52:27+01:00
