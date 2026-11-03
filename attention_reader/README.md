# The current attention-reader study

Wiktor Tomasik · 3 November 2026

Start with [the unified research explanation](RESEARCH_REPORT.md) or the [public chaptered study](https://pixedar.github.io/projects/self-awareness/attention-reader/study/index.html). The study includes architecture details, measured passages and responses, and synchronized recorded paths. It runs no language model in the browser.

The c8 reader remains the default reference. Its tensors are identical to the reader used in the new recorded experiments. Picture reductions, influence comparators and interpretation-to-attention feedback are distinct variants; the newest recorded variants have not been tested for live language behavior. No successful full fine-tuned J-workspace fusion is supplied.

## Reproduction

Install PyTorch for your hardware and NumPy. Run the authored symmetry demo:

```bash
python reproduce.py --states demo/states.npz --out results/demo --steps 468 --warmup 0 --device cpu
```

All four copies receive identical histories and should remain identical. This checks symmetry, not the emergence of individuality. Base Qwen weights are a separate upstream download for live experiments.

Reaggregate the 64 new recorded runs without a GPU. From the repository root, download the numerical archive from the public curated evidence dataset:

```bash
hf download Pixedar/jspace-strange-loop-evidence attention_reader/evidence/current/current_runs.npz --repo-type dataset --local-dir .
python attention_reader/evidence/current/recheck_current.py
```

`evidence/current/` contains run configurations, expected aggregates, source snapshots and the archive checksum; the large unchanged-field archive is hosted on the evidence dataset. `study/data/` contains display exports and actual generated replies. The [methods chapter](https://pixedar.github.io/projects/self-awareness/attention-reader/study/index.html#methods) explains experimental units, reference checks and interpretive limits. Earlier code/results remain available as dated supporting evidence rather than alternative current reports.

The completed late-evening extension is in `evidence/late/`. From the repository root:

```bash
hf download Pixedar/jspace-strange-loop-evidence attention_reader/evidence/late/late_runs.npz --repo-type dataset --local-dir .
python attention_reader/evidence/late/recheck_late.py
```

This checks 72 matched records, including 24 overlapping reference comparators. Those records must not be counted as 72 independent replications.
