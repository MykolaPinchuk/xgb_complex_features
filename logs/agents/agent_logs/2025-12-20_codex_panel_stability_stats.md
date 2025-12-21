# Agent log: benchmark-panel stability stats (win-rate + LOO)

- Timestamp: 2025-12-20T17:05:36-08:00
- Agent: Codex CLI
- Model: GPT-5.2

## Goal

Add lightweight “inference-like” stability summaries over the benchmark panel (no new training):

- Win rate: fraction of run instances with paired `ΔPRAUC > 0`, plus a Wilson interval.
- Leave-one-task-out (LOO) stability: min/max of the median `ΔPRAUC` when excluding each `task_id` in turn (where applicable).

## What changed

- Updated `src/xgb_complex_features/reporting/aggregate.py` to compute and merge:
  - `delta_prauc_pos_rate_frac_pos`
  - `delta_prauc_pos_rate_ci_lo`
  - `delta_prauc_pos_rate_ci_hi`
  - `delta_prauc_pos_rate_n_eff`
  - `delta_prauc_median_loo_task_min`
  - `delta_prauc_median_loo_task_max`
  - `delta_prauc_median_loo_task_n_loo`

These columns are added to summary tables where they make sense:

- `summary_by_task.*` (win rate only)
- `summary_by_regime_family.*` (win rate + LOO across tasks)
- `summary_by_level.*` (win rate + LOO across tasks within each level)
- `summary_by_level_regime_family.*` (win rate only)

## Commands run

- `python -m xgb_complex_features aggregate --input runs/exp_default --output runs/exp_default/aggregate`
- `python -m xgb_complex_features report --input runs/exp_default/aggregate --output runs/exp_default/report.md`

## Outputs to use in the paper

- `runs/exp_default/aggregate/summary_by_regime_family.csv`
- `runs/exp_default/aggregate/summary_by_level.csv`
- `runs/exp_default/aggregate/summary_by_task.csv`

