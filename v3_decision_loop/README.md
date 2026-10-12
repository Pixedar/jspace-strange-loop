# The J-space loop at a decision position

Qwen3-8B (Qwen3-14B replication), plus the natively recurrent Huginn-0125 and Ouro-2.6B. Scenarios with evidence for
one of two answers; the J code at the decision position is decomposed into 25 words and broadcast back into the model.

* [PROTOCOL.md](PROTOCOL.md) — the frozen protocol (sections 1–8 written before any data; dated amendments)
* [REPORT.md](REPORT.md) — full report: relational structure (H1), processing dispositions (H2), what the workspace
  holds, regimes A (collapse), B (self-simplification), C (intrinsic tension), feedback of the forecast error, natively
  recurrent models
* `code/` — `common.py` (model, Jacobian lens, batched J-space decomposition, residual hooks; also used by the later
  phases), `e1.py` / `e2.py` (experiments), `an_*.py` (analyses), `huginn.py` / `ouro.py`, `run_*.sh` (unattended runs)
* `results/` — compact JSON results and figures; the raw arrays (21 GB) are kept separately and available on request

Summary in [../REPORT.md](../REPORT.md) §3.3.
