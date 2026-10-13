# The frozen J-space loop

Qwen3-8B (Qwen3-14B where marked). The model's state at the answer position of a short scenario is read at layer 29,
described as 25 J-space words, written back at layer 22, and read again; nothing is trained. What the loop converges
to, why, and whether a self-model inside it can know anything private. Summary and figures:
[../REPORT.md](../REPORT.md), section 3.

## Code (`code/`)

| file | what it measures |
|---|---|
| `spectrum_v3.py` | settled states in 16- and 32-bit arithmetic; local stability (contraction factors); perturbations |
| `transport_v3.py`, `freq_control.py`, `check_words.py` | the write-to-read word transport matrix, self-gain, leading patterns, frequency control |
| `axis_v3.py` | the shared self-amplification direction and the persona axis (40 role prompts) |
| `geom_v3.py`, `geom_late.py`, `geom_words.py` | geometry of the loop trajectories: end states, residual motion, searches for rings, spirals and attractors |
| `stance_v3.py`, `deflate_v3.py` | stance words and readiness to revise; the loop without its own leading patterns |
| `confab.py` | references to a previous conversation in the model's thinking under the loop |
| `engine.py`, `qual.py`, `qual_mem.py` | the private-state test (own state vs a content-matched swap, planted positive control) |
| `phase_p.py`, `phase_e.py` | the forecaster whose predictions are written into the loop; online learning (the 2024 tension) |
| `figs_geom.py`, `figs_qual.py` | figures |

The loop itself and the scenario generator are in [`../common`](../common) (`e2.py`, `tasks.py`). `fetch_v3.py`
downloads the loop trajectories that the geometry scripts analyse.

## Results (`results/`)

Compact JSON per analysis (`spec/`, `transport/`, `axis/`, `geom/`, `stance/`, `deflate/`, `speech/`, `q/`, `qm/`,
`p/`, `e/`) and figures (`figs/`). Large arrays (trajectories, transport matrices, persona projections) are in the
Hugging Face dataset under `v5/`.

## Working notes

Dated documents written during the experiments, kept as they were:

* [GEOMETRY_PLAN.md](GEOMETRY_PLAN.md) — measures and interpretation rules, written before the geometry results
* [GEOMETRY_RESULTS.md](GEOMETRY_RESULTS.md) — full geometry results, with the corrections made along the way
* [PROTOCOL.md](PROTOCOL.md) — protocol of the self-forecast loop
* [SELF_FORECAST_GATES.md](SELF_FORECAST_GATES.md) — the private-state test and the exploratory forecaster runs

Names used in these notes: "v3" is the first implementation of this loop (`../common/e2.py`) and its result arrays;
paths such as `research_v3/…` or `research_v5/…` refer to the original working tree (now `../common` and this folder);
"F2" and "F3" are the two scenario families (interpretation under evidence, false belief); "Phase Q" is the
private-state test, "G2" its main contrast and "G3" the planted positive control; "Phase P" and "Phase E" are the
forecaster experiments (frozen and online); "gain" is the loop strength g; "Δc" is the change of the readiness to
revise and "Δlo" the change of the answer's log-odds.
