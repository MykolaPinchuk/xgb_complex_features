# Agent log: onboarding (refresh)

- Timestamp: 2025-12-20T16:22:05-08:00
- Agent: Codex CLI
- Model: GPT-5.2

## Repo purpose

This repo implements a synthetic benchmark to measure how well XGBoost learns ratio/product structure from raw variables, compared to “oracle” feature views that include DGP-derived coordinates and/or the true signal.

## Primary workflow

- CLI entrypoint: `python -m xgb_complex_features {run,aggregate,report,workflow,smoke}`
- Make shortcuts: `make smoke`, `make exp_default`
- Configs live in `configs/*.yaml` and define tasks, regimes, oracle modes, XGB configs, diagnostics, and output locations.

Pipeline:

1) `run`: generate datasets and train models; write `runs/<exp>/results.(parquet|csv)` + metadata/config snapshots.
2) `aggregate`: read results and compute deltas vs `raw_only`; write `runs.parquet`, `deltas.parquet`, and summary tables.
3) `report`: generate a Markdown report with tables and plots under `<aggregate_dir>/plots/`.

## Code map (where to look)

- `src/xgb_complex_features/dgp/`
  - `dataset.py`: raw feature generation, deterministic splits, test-time shifts (`none`, `naive`, `preserve`), label calibration.
  - `tasks.py`: task definitions (ratio/product, interactions, nonmonotone, gated) and the `TaskTransform` feature construction.
  - `label.py`: prevalence calibration (binary search on `beta0`) and Bernoulli sampling.
- `src/xgb_complex_features/runner/execute.py`: experiment grid execution (joblib parallelism) + result row assembly.
- `src/xgb_complex_features/models/xgb.py`: `XGBClassifier` training, prediction, metrics (PRAUC primary).
- `src/xgb_complex_features/diagnostics/`: invariance and dominance shortcut diagnostics.
- `src/xgb_complex_features/reporting/`: aggregation + Markdown report generation + plotting.

## Quick sanity checks I ran

- `python -m xgb_complex_features -h` works.
- `python -c "import numpy, pandas, sklearn, xgboost, pyarrow, yaml"` works in this environment.
- `python -c "import xgb_complex_features; print(xgb_complex_features.__version__)"` works; repo supports `python -m xgb_complex_features` from root without `pip install -e .` via `xgb_complex_features/__init__.py`.

## Notes / gotchas

- `configs/smoke.yaml` sets `output.overwrite: true`, so re-running the smoke pipeline will overwrite artifacts under `runs/smoke/`.
- The `shift_preserve` regime uses per-task diagnostics to rescale raw columns on the test split while keeping the “true” ratio/product invariant (an identifiability / shortcut-stress device).
- `nonmonotone` tasks wrap a base ratio/product coordinate and apply either a U-shape/peak (`-((u-mu)/delta)^2`) or a Gaussian band-pass.
- Aggregation outputs under `runs/<exp>/aggregate/` now also include benchmark-panel stability columns in `summary_by_level.*` / `summary_by_regime_family.*` (win rate `%{ΔPRAUC>0}` with Wilson interval, plus leave-one-task-out median ranges).

## Ready for the next task

I’m fully oriented on the run → aggregate → report flow and where to extend tasks/regimes/diagnostics or extract tables/plots for the paper.
