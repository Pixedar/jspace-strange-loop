# What the frozen J loop converges to, and why self-forecasts cannot be self-specific there

Qwen3-8B (14B where marked), re-running the decision-position loop of `../v3_decision_loop`.

* [GEOMETRY_PLAN.md](GEOMETRY_PLAN.md) — the measures and interpretation rules, written before the results
* [GEOMETRY_RESULTS.md](GEOMETRY_RESULTS.md) — fixed points (bf16 vs fp32), local spectra, the write→read word-transport
  operator, the shared self-amplification direction and persona space, stance, self-deflation, the loop during the
  model's own thinking
* [PROTOCOL.md](PROTOCOL.md) and [SELF_FORECAST_GATES.md](SELF_FORECAST_GATES.md) — the self-forecast loop and its
  qualification gates (no private state in frozen-context loops; positive controls)
* [LESSONS.md](LESSONS.md) — eight negative results turned into design constraints for the trained loop
* `code/` — `spectrum_v3.py` (fixed points, Jacobians), `transport_v3.py` + `freq_control.py` (word transport),
  `axis_v3.py` (persona space), `geom_*.py` (retrospective geometry), `stance_v3.py`, `deflate_v3.py`, `confab.py`,
  `qual.py` / `qual_mem.py` (self-forecast gates), `phase_p.py`, `phase_e.py`
* `results/` — compact JSON results and figures; geometry and transport arrays on Hugging Face (`v5/`)

Summary in [../REPORT.md](../REPORT.md) §3.4–3.5.
