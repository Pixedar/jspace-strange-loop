# Self-describing attention

This is the current experimental family: a learned reader and picture around a frozen Qwen3 model, with a centered-field description objective and an optional shared spotlight into the model’s attention. The [main report](../REPORT.md) explains the findings and limitations.

## What is in the package

- `code/llm_loop.py`: frozen Qwen3 runner and trainable interface.
- `code/loop12.py`: training, centered-field identity objective, sphere picture, channel controls and shared spotlight. Earlier controlled variants remain available as explicit switches; the current configuration disables private body fluctuations and plastic weights.
- Analysis scripts: probe-field decomposition, fixed-probe comparisons, picture transplants, path cuts, shared-text convergence, spotlight consequences, attention and meaning decoders.
- `results/`: compact JSON measurements, the meaning table, verified summary and source/export hashes.
- Numeric recordings and learned interface weights: the linked Hugging Face dataset, fetched by the hash-verifying helper below. These are interface weights, not redistributed Qwen model weights.

## Inspect the evidence without a GPU

Create a separate environment with Python 3.10 or newer. Install `requirements-current.txt`; choose the official PyTorch wheel appropriate to your CPU/GPU. `RUNTIME.json` records the environment qualified for this release.

From the repository root:

```bash
python self_reference/code/fetch_artifacts.py --output .
python self_reference/code/verify_evidence.py
python figures/make_current.py
```

The downloader uses the immutable dataset revision and SHA-256 values in `ARTIFACTS.json`. It downloads only the selected current-family artifacts, rather than the entire historical dataset. The viewer in `interactive/` is a replay of saved measurements; serve it with an ordinary local HTTP server.

The final result JSONs are preserved measurements, with private workstation paths removed. The export manifest records source hashes and exported hashes. Some statistics, including the original field-head correlation, were saved as summaries rather than full predicted-field arrays; the release verifies their code definitions and recorded values rather than pretending to recompute unsaved predictions.

## Reconstruct the text corpus

The combined corpus contains WikiText-2, AG News, TinyStories, DialogSum and generated causal situations. **The upstream texts are not redistributed or relicensed here.** They have different terms, including noncommercial ShareAlike terms for DialogSum and an unknown license label for AG News. See `corpus_sources.json` and the upstream cards before downloading or reusing them.

```bash
python self_reference/code/build_corpus.py data
python self_reference/code/check_corpus.py data/corpus.json
```

The constructor pins upstream dataset revisions. The checker compares the reconstructed combined corpus with the original corpus’s SHA-256. If it differs, that is an input-reproduction failure, not evidence against the reported loop. A different corpus may be useful for exploration, but does not reproduce these measurements.

## Run a current continuation

The two organized v22 starting interfaces and selected complete v24 interfaces are included in the data release. `examples/development_spotlight.json` contains the current 0.6B configuration with portable paths and an explicit model revision.

```bash
python self_reference/code/loop12.py self_reference/examples/development_spotlight.json
```

The example writes to `artifacts/new_development`; it does not overwrite the published recordings. The saved historical training configurations did not record an immutable model revision. The example pins the revision independently confirmed in the local Qwen3-0.6B cache and in the follow-up run; it is a transparent reproduction choice, rather than a claim that the old configuration already contained that provenance.

The original runner uses BF16 model weights, float32 interface states and explicit float32 biased attention. Its cache keeps four initial attention-sink positions and recent positions within a 256-token window. The trained 0.6B configuration uses band layers 12–18 (zero-based), read layer 19, four copies, eight fresh probes plus current states, control dimension 8, picture dimension 32, reader window 16, learning rate 0.002 and 8-token optimization chunks. The original centered-field continuations begin from organized token-2,000 interfaces and run for 3,000 further tokens. Spotlight strength is 1.5; the direct query shift is disabled.

The exported example was checked with a real eight-token training smoke run, producing a checkpoint and a completion record. This verifies the portable entry point, not a repeat of the research outcome. Historical configurations have private output, corpus and target-record locations removed or shortened. The package supports a new self-describing continuation and frozen re-evaluation from saved interfaces; it is not a complete rerun bundle for every development history. In particular, the other-description control's separately recorded `twin_s1s1` training target and the pre-continuation rich interfaces are not included in the selected release. Recreating those training histories requires their corresponding source artifacts; do not substitute a different target record and call it the original control.

Changing precision, kernel, batch arrangement or context handling can change sensitive recurrent trajectories. Establish replay compatibility before treating a changed backend as a reproduction. The new multimodal pilot uses a different model and FP16 native inference on a T4 and is not numerically interchangeable with these runs.

## Frozen analyses

The scripts take a writable run directory containing a configuration and a checkpoint path. Set `JSPACE_CORPUS` to the reconstructed corpus. Use a working copy of an artifact directory, so the scripts’ result files do not overwrite the downloaded evidence.

Examples:

```bash
python self_reference/code/self_model_residual.py WORKING_RUN IFACE_PT 900
python self_reference/code/self_formation.py WORKING_RUN IFACE_PT
python self_reference/code/life_fields.py record WORKING_RUN IFACE_PT 900
python self_reference/code/spotlight_check.py OUTPUT_JSON 1.5:nocalib NAME=WORKING_RUN::IFACE_PT
```

The older metric key `remainder_share_of_field` is retained to preserve recorded-file compatibility. It is the normalized ratio defined in [REPORT.md](../REPORT.md#3-what-state-dependence-means-here), not a bounded share of causal responsibility. A positive pattern correlation and an accurate vector forecast are different tests.
