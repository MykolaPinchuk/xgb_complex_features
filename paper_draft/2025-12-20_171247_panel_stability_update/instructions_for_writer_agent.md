# Instructions for writer agent: add benchmark-panel stability stats to v7_fixed

Target draft: `paper_draft/2025-12-14_110315/xgb_invariant_features_updated_v7_fixed.pdf`

This bundle adds lightweight “inference-like” stability summaries computed **from existing runs** (no new training). The goal is to strengthen credibility without changing the main story or requiring new experiments.

## What “stability over the benchmark panel” means (1 short paragraph to add)

Add a short paragraph (Methods or at start of Results) defining:

- **Experimental unit**: a run instance `(task_id, regime_id, seed, n, xgb_config_id)` with a paired delta `Δ = PRAUC_oracle − PRAUC_raw`.
- **Panel interpretation**: reported medians and p10–p90 summarize **heterogeneity across benchmark instances**; the new “win rate” summarizes how often the paired delta is positive across the panel (not classical i.i.d. sampling inference).

Suggested text (tight, v7 style):

> Each configuration produces paired outcomes for raw and oracle features on the same synthetic dataset instance. I treat each `(task, regime, seed)` instance as one experimental unit and summarize the paired effect `ΔPRAUC`. Percentile ranges describe heterogeneity across instances. I additionally report the fraction of instances with `ΔPRAUC > 0` as a stability check over the benchmark panel.

## Table 1 update (regime-family breakdown)

Current Table 1 in v7_fixed shows raw/oracle medians, median Δ, p10–p90, runs. Add one column:

- **Win rate**: `% runs with Δ>0` (optionally add a 95% Wilson interval in parentheses).

Use baseline model and oracle coords only. Source:

- `paper_draft/2025-12-20_171247_panel_stability_update/tables/exp_default__summary_by_regime_family.csv`
- Filter: `xgb_config_id == baseline` and `oracle_mode == oracle_coords_only`.

Columns to use:

- `delta_prauc_pos_rate_frac_pos` (multiply by 100 for percent)
- `delta_prauc_pos_rate_ci_lo`, `delta_prauc_pos_rate_ci_hi` (optional)

Numbers (already filtered; copy/paste):

| regime_family | median ΔPRAUC | p10 | p90 | runs | win rate (Δ>0) |
| --- | --- | --- | --- | --- | --- |
| tail_corr | 0.006558 | -0.005747 | 0.037327 | 108 | 76.9% (68.1–83.8) |
| mixture_extremes | 0.015986 | 0.002924 | 0.046060 | 36 | 97.2% (85.8–99.5) |
| shift_naive | 0.015509 | -0.010090 | 0.094778 | 36 | 72.2% (56.0–84.2) |
| shift_preserve | 0.146383 | 0.017371 | 0.392228 | 36 | 97.2% (85.8–99.5) |

Optional (footnote or appendix): LOO stability of the regime-family median when dropping each task in turn:

- Columns: `delta_prauc_median_loo_task_min`, `delta_prauc_median_loo_task_max`

For `shift_preserve` (baseline/coords): median Δ ranges from 0.128 to 0.157 under leave-one-task-out across the 12 tasks.

## Table 2 update (by level)

Add a “win rate (Δ>0)” column for both oracle variants (coords and signal), or add it only for coords to keep the table compact.

Source:

- `paper_draft/2025-12-20_171247_panel_stability_update/tables/exp_default__summary_by_level.csv`
- Filter: `xgb_config_id == baseline`, `level in {1..6}`

Columns to use:

- same `delta_prauc_pos_rate_*` fields.

Baseline + oracle coords only (Levels 1–6):

| level | median ΔPRAUC | runs | win rate (Δ>0) |
| --- | --- | --- | --- |
| 1 | 0.006043 | 36 | 72.2% (56.0–84.2) |
| 2 | 0.011333 | 36 | 77.8% (61.9–88.3) |
| 3 | 0.028011 | 36 | 83.3% (68.1–92.1) |
| 4 | 0.035380 | 36 | 88.9% (74.7–95.6) |
| 5 | 0.049608 | 18 | 94.4% (74.2–99.0) |
| 6 | 0.006231 | 36 | 88.9% (74.7–95.6) |

Baseline + oracle signal (Levels 1–6):

| level | median ΔPRAUC | runs | win rate (Δ>0) |
| --- | --- | --- | --- |
| 1 | 0.007966 | 36 | 69.4% (53.1–82.0) |
| 2 | 0.014138 | 36 | 88.9% (74.7–95.6) |
| 3 | 0.046745 | 36 | 94.4% (81.9–98.5) |
| 4 | 0.051545 | 36 | 97.2% (85.8–99.5) |
| 5 | 0.062122 | 18 | 100.0% (82.4–100.0) |
| 6 | 0.007924 | 36 | 88.9% (74.7–95.6) |

Optional: add a brief note that Level 4 is heterogeneous across its two tasks; LOO range for the median (baseline/coords) is wide:

- Level 4 LOO median Δ range: 0.012 to 0.106 (from `delta_prauc_median_loo_task_min/max`).

## Where to cite the stability results (minimal edits)

- Add one sentence after Table 1: “The paired gap is positive in 97% of preserve-shift instances (Table 1).”
- Add one sentence after Table 2: “Win rates increase for interaction motifs, consistent with larger median gaps.”

## Files in this bundle

- Tables:
  - `paper_draft/2025-12-20_171247_panel_stability_update/tables/exp_default__summary_by_regime_family.csv`
  - `paper_draft/2025-12-20_171247_panel_stability_update/tables/exp_default__summary_by_level.csv`
  - `paper_draft/2025-12-20_171247_panel_stability_update/tables/exp_default__summary_by_task.csv`
  - `paper_draft/2025-12-20_171247_panel_stability_update/tables/exp_default__summary_by_level_regime_family.csv`
  - `paper_draft/2025-12-20_171247_panel_stability_update/tables/exp_default__deltas.csv`
- Report snapshot:
  - `paper_draft/2025-12-20_171247_panel_stability_update/reports/exp_default__report.md`

