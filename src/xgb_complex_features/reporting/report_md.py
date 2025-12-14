from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from xgb_complex_features.reporting.plots import (
    plot_delta_boxplots,
    plot_delta_heatmap,
    plot_delta_vs_dominance,
    plot_delta_vs_dominance_by_level,
    plot_delta_vs_invariance,
    plot_delta_vs_invariance_by_level,
    plot_delta_vs_n,
)


def _md_table(df: pd.DataFrame, *, max_rows: int = 30) -> str:
    if df.empty:
        return "_(empty)_\n"

    show = df.head(max_rows).copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda v: f"{v:.4g}" if pd.notna(v) else "")
    cols = list(show.columns)
    header = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    rows = []
    for _, r in show.iterrows():
        rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    return header + sep + "".join(rows) + ("\n" if len(df) > max_rows else "")


def build_report(*, input_dir: str, output_path: str) -> None:
    in_dir = Path(input_dir)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runs_path = in_dir / "runs.parquet"
    deltas_path = in_dir / "deltas.parquet"
    summary_task_path = in_dir / "summary_by_task.parquet"
    summary_regime_path = in_dir / "summary_by_regime_family.parquet"
    summary_task_n_path = in_dir / "summary_by_task_n.parquet"
    summary_level_path = in_dir / "summary_by_level.parquet"
    summary_level_regime_path = in_dir / "summary_by_level_regime_family.parquet"

    runs = pd.read_parquet(runs_path) if runs_path.exists() else pd.DataFrame()
    deltas = pd.read_parquet(deltas_path) if deltas_path.exists() else pd.DataFrame()
    summary_task = pd.read_parquet(summary_task_path) if summary_task_path.exists() else pd.DataFrame()
    summary_regime = pd.read_parquet(summary_regime_path) if summary_regime_path.exists() else pd.DataFrame()
    summary_task_n = pd.read_parquet(summary_task_n_path) if summary_task_n_path.exists() else pd.DataFrame()
    summary_level = pd.read_parquet(summary_level_path) if summary_level_path.exists() else pd.DataFrame()
    summary_level_regime = (
        pd.read_parquet(summary_level_regime_path) if summary_level_regime_path.exists() else pd.DataFrame()
    )

    meta_path = in_dir / "run_metadata.json"
    run_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    plots_dir = out_path.parent / "plots"
    boxplot = plot_delta_boxplots(deltas, plots_dir)
    inv_plot = plot_delta_vs_invariance(deltas, plots_dir)
    dom_plot = plot_delta_vs_dominance(deltas, plots_dir)
    n_plot = plot_delta_vs_n(summary_task_n, plots_dir)
    heatmap = plot_delta_heatmap(summary_level_regime, plots_dir)
    level_inv_plot = plot_delta_vs_invariance_by_level(summary_level, plots_dir)
    level_dom_plot = plot_delta_vs_dominance_by_level(summary_level, plots_dir)

    rel_box = boxplot.relative_to(out_path.parent)
    rel_inv = inv_plot.relative_to(out_path.parent)
    rel_dom = dom_plot.relative_to(out_path.parent)
    rel_n = n_plot.relative_to(out_path.parent)
    rel_heat = heatmap.relative_to(out_path.parent)
    rel_level_inv = level_inv_plot.relative_to(out_path.parent)
    rel_level_dom = level_dom_plot.relative_to(out_path.parent)

    uniq = lambda series: sorted(pd.Series(series).dropna().unique().tolist()) if not runs.empty else []  # noqa: E731
    tasks = uniq(runs.get("task_id"))
    regimes = uniq(runs.get("regime_id"))
    regime_families = uniq(runs.get("regime_family"))
    oracle_modes = uniq(runs.get("oracle_mode"))
    xgb_cfgs = uniq(runs.get("xgb_config_id"))
    n_vals = uniq(runs.get("n"))
    seeds = uniq(runs.get("seed"))

    def _overview_lines() -> list[str]:
        if runs.empty:
            return ["- No `runs.parquet` found.\n"]
        lines = [
            f"- Run rows: {len(runs)}\n",
            f"- Delta rows: {len(deltas)}\n" if not deltas.empty else "- Delta rows: 0\n",
            f"- Tasks ({len(tasks)}): {', '.join(map(str, tasks))}\n",
            f"- Regimes ({len(regimes)}): {', '.join(map(str, regimes))}\n",
            f"- Oracle modes ({len(oracle_modes)}): {', '.join(map(str, oracle_modes))}\n",
            f"- XGB configs ({len(xgb_cfgs)}): {', '.join(map(str, xgb_cfgs))}\n",
            f"- n values ({len(n_vals)}): {', '.join(map(str, n_vals))}\n",
            f"- seeds ({len(seeds)}): {', '.join(map(str, seeds))}\n",
        ]
        if run_meta:
            duration = run_meta.get("duration_seconds")
            start_time = run_meta.get("start_time")
            end_time = run_meta.get("end_time")
            if duration is not None:
                line = f"- Execution time: {duration:.2f}s"
                if start_time and end_time:
                    line += f" (start {start_time} → end {end_time})"
                lines.append(line + "\n")
        return lines

    def _regime_table() -> pd.DataFrame:
        if runs.empty:
            return pd.DataFrame()
        cols = ["regime_id", "regime_family", "sigma", "rho", "shift_type", "mixture_json"]
        present = [c for c in cols if c in runs.columns]
        return runs[present].drop_duplicates().sort_values(["regime_id"]).reset_index(drop=True)

    def _table_subset(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if df.empty:
            return df
        present = [c for c in cols if c in df.columns]
        if not present:
            return df
        return df[present]

    def _relative_drop_table(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        required = {"prauc_raw_median", "prauc_oracle_median"}
        if not required.issubset(df.columns):
            return pd.DataFrame()
        cols = [c for c in keys if c in df.columns]
        view = df[cols + sorted(required)].copy()
        denom = view["prauc_oracle_median"].replace(0.0, np.nan)
        view["relative_drop_pct"] = ((view["prauc_oracle_median"] - view["prauc_raw_median"]) / denom) * 100.0
        view = view.replace([np.inf, -np.inf], np.nan).dropna(subset=["relative_drop_pct"])
        return view.sort_values("relative_drop_pct", ascending=False)

    def _top_bottom_deltas() -> tuple[pd.DataFrame, pd.DataFrame]:
        if deltas.empty:
            return pd.DataFrame(), pd.DataFrame()
        d = deltas.copy()
        # Single diagnostic axis for quick scanning (ratio if present else product).
        d["invariance_raw"] = d["ratio_scale_invariance_raw"].where(
            d["ratio_scale_invariance_raw"].notna(), d["product_comp_invariance_raw"]
        )
        d["invariance_oracle"] = d["ratio_scale_invariance_oracle"].where(
            d["ratio_scale_invariance_oracle"].notna(), d["product_comp_invariance_oracle"]
        )
        d["invariance_improvement"] = d["invariance_raw"] - d["invariance_oracle"]
        cols = [
            "task_id",
            "level",
            "regime_id",
            "seed",
            "n",
            "oracle_mode",
            "xgb_config_id",
            "prauc_raw",
            "prauc_oracle",
            "delta_prauc",
            "invariance_raw",
            "invariance_oracle",
            "invariance_improvement",
            "dominance_train_mean",
            "dominance_test_mean",
            "shift_type",
        ]
        cols = [c for c in cols if c in d.columns]
        top = d.sort_values("delta_prauc", ascending=False)[cols].head(25).reset_index(drop=True)
        bot = d.sort_values("delta_prauc", ascending=True)[cols].head(25).reset_index(drop=True)
        return top, bot

    def _diagnostics_summary() -> pd.DataFrame:
        if deltas.empty:
            return pd.DataFrame()
        d = deltas.copy()
        d["invariance_raw"] = d["ratio_scale_invariance_raw"].where(
            d["ratio_scale_invariance_raw"].notna(), d["product_comp_invariance_raw"]
        )
        d["invariance_oracle"] = d["ratio_scale_invariance_oracle"].where(
            d["ratio_scale_invariance_oracle"].notna(), d["product_comp_invariance_oracle"]
        )
        d["invariance_improvement"] = d["invariance_raw"] - d["invariance_oracle"]
        d["iso_raw"] = d["iso_var_ratio_raw"].where(d["iso_var_ratio_raw"].notna(), d["iso_var_product_raw"])
        d["iso_oracle"] = d["iso_var_ratio_oracle"].where(d["iso_var_ratio_oracle"].notna(), d["iso_var_product_oracle"])
        d["iso_improvement"] = d["iso_raw"] - d["iso_oracle"]
        g = (
            d.groupby(["oracle_mode", "xgb_config_id"], dropna=False)[
                ["invariance_raw", "invariance_oracle", "invariance_improvement", "iso_raw", "iso_oracle", "iso_improvement"]
            ]
            .median(numeric_only=True)
            .reset_index()
            .sort_values(["oracle_mode", "xgb_config_id"])
        )
        return g

    top, bot = _top_bottom_deltas()
    diag_summary = _diagnostics_summary()
    level_rank = (
        summary_level.sort_values("delta_prauc_median", ascending=False).reset_index(drop=True)
        if not summary_level.empty
        else pd.DataFrame()
    )
    level_regime_rank = (
        summary_level_regime.sort_values("delta_prauc_median", ascending=False).reset_index(drop=True)
        if not summary_level_regime.empty
        else pd.DataFrame()
    )
    rel_task = _relative_drop_table(
        summary_task,
        ["task_id", "level", "oracle_mode", "xgb_config_id"],
    )
    rel_level = _relative_drop_table(
        summary_level,
        ["level", "oracle_mode", "xgb_config_id"],
    )
    rel_regime = _relative_drop_table(
        summary_regime,
        ["regime_family", "oracle_mode", "xgb_config_id"],
    )

    md = []
    md.append(f"# XGB Complex Features Report\n\n")
    md.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    md.append(f"- Input: `{in_dir}`\n\n")

    md.append("## Overview\n\n")
    md.extend(_overview_lines())
    md.append("\n")

    md.append("## Experiment setup\n\n")
    if runs.empty:
        md.append("No run metadata available; rerun the pipeline to populate `runs.parquet`.\n\n")
    else:
        md.append(
            "This benchmark trains gradient-boosted trees (XGBoost) on synthetic ratio/product tasks to test whether "
            "a model trained on **raw variables only** can rediscover the **constructed true feature(s)** used by the "
            "data-generating process (DGP).\n\n"
            "DGP sketch (per row):\n\n"
            "1) Generate positive raw variables `x` (lognormal by default, with configurable correlation).\n"
            "2) Compute constructed true feature(s) `u` from `x` (e.g., log-ratios, log-products, sum-normalizations).\n"
            "3) Combine them into a final signal `s` (e.g., `s=u`, `s=u1-u2`, `s=u1*u2`, or a non-monotone transform).\n"
            "4) Sample label `y ~ Bernoulli(sigmoid(beta0 + a*s + noise))`.\n\n"
            "Important: **we never provide `y` to the model as a feature**. The oracle modes only add deterministic "
            "functions of `x` that the DGP itself uses.\n\n"
            "Feature views (oracle modes):\n\n"
            "- `raw_only`: only the generated raw columns `x_i`.\n"
            "- `oracle_coords_only`: raw columns plus the constructed true feature coordinates (`u`, `u1`, `u2`, "
            "gate indicator, etc.), but **not** the final signal `s`.\n"
            "- `oracle_s_only`: raw columns plus the ground-truth signal `s` (a deterministic feature used to generate `y`).\n\n"
        )
        md.append(
            f"Runs cover **{len(tasks)} task(s)** ({', '.join(tasks) if tasks else 'n/a'}), "
            f"**{len(regime_families)} regime family(ies)** ({', '.join(regime_families) if regime_families else 'n/a'}), "
            f"and dataset size(s) **n = {', '.join(map(str, n_vals)) if n_vals else 'n/a'}** with seeds {', '.join(map(str, seeds)) if seeds else 'n/a'}.\n\n"
        )
        md.append(
            "The latent prevalence is calibrated before any train/val/test split, then deterministic splits (default 60/20/20) "
            "feed the model. Regime metadata captures tail-heaviness, correlations, and scale-shift stress tests.\n\n"
        )

    md.append("## What we measure\n\n")
    md.append(
        "- **Primary metric**: PRAUC (Average Precision) on the test split. We report the generalization gap "
        "Δ = PRAUC_oracle − PRAUC_raw for each oracle mode.\n"
        "- **Invariance diagnostics**: perturbations that should leave the true ratio/product invariant "
        "(scaling numerator/denominator together or compensating product transforms). We report the mean absolute "
        "prediction change, so smaller is better.\n"
        "- **Iso-coordinate variance**: prediction variance when keeping ratio/product constant while varying magnitudes.\n"
        "- **Dominance stats**: `max(x_i)/sum(x_i)` for each sum used in ratio/product-of-sums tasks to detect "
        "shortcut opportunities.\n"
        "- **Variance summaries**: Δ PRAUC standard deviation and IQR across regimes/seeds/configs help flag brittle "
        "levels even when medians look healthy.\n\n"
    )

    md.append("## Regimes\n\n")
    md.append(_md_table(_regime_table(), max_rows=50))

    md.append("## Summary by task\n\n")
    md.append(_md_table(summary_task, max_rows=50))
    if not rel_task.empty:
        md.append("### Relative Δ% (oracle vs raw) by task\n\n")
        md.append(_md_table(rel_task, max_rows=50))

    md.append("## Summary by level\n\n")
    md.append(_md_table(summary_level, max_rows=50))
    if not rel_level.empty:
        md.append("### Relative Δ% (oracle vs raw) by level\n\n")
        md.append(_md_table(rel_level, max_rows=50))

    if not level_rank.empty:
        level_rank_table = _table_subset(
            level_rank,
            [
                "level",
                "oracle_mode",
                "xgb_config_id",
                "delta_prauc_median",
                "delta_prauc_p90",
                "delta_prauc_p10",
                "delta_prauc_std",
                "delta_prauc_iqr",
                "prauc_raw_median",
                "prauc_oracle_median",
            ],
        )
        md.append("## Highest Δ levels (ranked)\n\n")
        md.append(_md_table(level_rank_table, max_rows=25))

    md.append("## Summary by regime family\n\n")
    md.append(_md_table(summary_regime, max_rows=50))
    if not rel_regime.empty:
        md.append("### Relative Δ% (oracle vs raw) by regime family\n\n")
        md.append(_md_table(rel_regime, max_rows=50))

    if not summary_level_regime.empty:
        md.append("## Summary by level × regime family\n\n")
        md.append(_md_table(summary_level_regime, max_rows=80))
        md.append("### Heatmap: Δ PRAUC median (level × regime)\n\n")
        md.append(f"![]({rel_heat.as_posix()})\n\n")

    if not level_regime_rank.empty:
        lr = _table_subset(
            level_regime_rank,
            [
                "level",
                "regime_family",
                "oracle_mode",
                "xgb_config_id",
                "delta_prauc_median",
                "delta_prauc_p90",
                "delta_prauc_p10",
                "delta_prauc_std",
            ],
        )
        md.append("## Highest Δ level × regime combos\n\n")
        md.append(_md_table(lr, max_rows=40))

    if not summary_task_n.empty and "n" in summary_task_n.columns and summary_task_n["n"].nunique(dropna=True) > 1:
        md.append("## Δ vs n (summary)\n\n")
        md.append(_md_table(summary_task_n, max_rows=80))

    md.append("## Diagnostics (median)\n\n")
    md.append(_md_table(diag_summary, max_rows=50))

    md.append("## Largest Δ runs\n\n")
    md.append(_md_table(top, max_rows=25))

    md.append("## Smallest Δ runs\n\n")
    md.append(_md_table(bot, max_rows=25))

    md.append("## Plots\n\n")
    md.append(f"### Δ PRAUC distribution\n\n![]({rel_box.as_posix()})\n\n")
    md.append(f"### Δ vs n\n\n![]({rel_n.as_posix()})\n\n")
    md.append(f"### Invariance vs Δ\n\n![]({rel_inv.as_posix()})\n\n")
    md.append(f"### Dominance vs Δ\n\n![]({rel_dom.as_posix()})\n\n")
    md.append(f"### Per-level Δ vs invariance\n\n![]({rel_level_inv.as_posix()})\n\n")
    md.append(f"### Per-level Δ vs dominance\n\n![]({rel_level_dom.as_posix()})\n\n")

    out_path.write_text("".join(md), encoding="utf-8")
