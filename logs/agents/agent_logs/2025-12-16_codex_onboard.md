# Agent log: onboarding

- Timestamp: 2025-12-16T19:20:26-08:00
- Agent: Codex CLI
- Model: GPT-5.2

## What I checked

- Read `agents.md`, `README.md`, `Makefile`, `pyproject.toml`, `prd.md`, `todo.md`, and key modules under `src/xgb_complex_features/`.
- Mapped the CLI entrypoint in `src/xgb_complex_features/__main__.py` and the core pipeline flow:
  - `runner.execute.run_experiment()` → writes `runs/<name>/results.(parquet|csv)` + metadata.
  - `reporting.aggregate.aggregate_runs()` → writes `runs.parquet`, `deltas.parquet`, and summary tables.
  - `reporting.report_md.build_report()` → writes a Markdown report and plots.

## Quick sanity checks

- `python -m xgb_complex_features -h` works via `python3 -m xgb_complex_features -h`.
- Runtime deps (e.g. `pandas`) are not installed in this environment yet, so `aggregate`/`report` fail until you run the install step (ideally in a venv).

## Notes for the next task

- Preferred workflow is `python -m xgb_complex_features workflow --config configs/<...>.yaml` or `make smoke` / `make exp_default`.
- In environments without a `python` shim, use `python3` and/or `make PYTHON=python3 ...`.
