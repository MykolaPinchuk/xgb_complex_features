from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from xgb_complex_features.utils import ensure_dir

logger = logging.getLogger(__name__)


def _find_result_files(input_dir: Path) -> list[Path]:
    parquet = sorted(input_dir.rglob("results.parquet"))
    if parquet:
        return parquet
    csv = sorted(input_dir.rglob("results.csv"))
    return csv


def _read_results(path: Path) -> pd.DataFrame:
    if path.name.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.name.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported results file: {path}")


def _std0(series: pd.Series) -> float:
    return float(series.std(ddof=0))


def _iqr(series: pd.Series) -> float:
    return float(series.quantile(0.75) - series.quantile(0.25))


def _wilson_interval(k: int, n: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    if k < 0 or k > n:
        raise ValueError("k must be in [0, n]")

    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = (z * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi


def _win_rate_table(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    value_col: str,
    out_prefix: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                *group_cols,
                f"{out_prefix}_frac_pos",
                f"{out_prefix}_ci_lo",
                f"{out_prefix}_ci_hi",
                f"{out_prefix}_n_eff",
            ]
        )

    def _one(g: pd.DataFrame) -> pd.Series:
        v = pd.to_numeric(g[value_col], errors="coerce").dropna()
        n_eff = int(v.shape[0])
        if n_eff == 0:
            return pd.Series(
                {
                    f"{out_prefix}_frac_pos": float("nan"),
                    f"{out_prefix}_ci_lo": float("nan"),
                    f"{out_prefix}_ci_hi": float("nan"),
                    f"{out_prefix}_n_eff": 0,
                }
            )
        k = int((v > 0).sum())
        frac = float(k / n_eff)
        lo, hi = _wilson_interval(k, n_eff)
        return pd.Series(
            {
                f"{out_prefix}_frac_pos": frac,
                f"{out_prefix}_ci_lo": lo,
                f"{out_prefix}_ci_hi": hi,
                f"{out_prefix}_n_eff": n_eff,
            }
        )

    return df.groupby(group_cols, dropna=False).apply(_one, include_groups=False).reset_index()


def _loo_median_range_table(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    loo_col: str,
    value_col: str,
    out_prefix: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[*group_cols, f"{out_prefix}_min", f"{out_prefix}_max", f"{out_prefix}_n_loo"])

    def _one(g: pd.DataFrame) -> pd.Series:
        loo_vals = pd.Series(g[loo_col]).dropna().unique().tolist()
        medians: list[float] = []
        for v in loo_vals:
            vv = pd.to_numeric(g.loc[g[loo_col] != v, value_col], errors="coerce").dropna()
            if vv.empty:
                continue
            medians.append(float(vv.median()))
        if not medians:
            return pd.Series(
                {
                    f"{out_prefix}_min": float("nan"),
                    f"{out_prefix}_max": float("nan"),
                    f"{out_prefix}_n_loo": int(len(loo_vals)),
                }
            )
        return pd.Series(
            {
                f"{out_prefix}_min": float(min(medians)),
                f"{out_prefix}_max": float(max(medians)),
                f"{out_prefix}_n_loo": int(len(loo_vals)),
            }
        )

    return df.groupby(group_cols, dropna=False).apply(_one, include_groups=False).reset_index()


def aggregate_runs(*, input_dir: str, output_dir: str) -> None:
    in_dir = Path(input_dir)
    out_dir = ensure_dir(output_dir)

    files = _find_result_files(in_dir)
    if not files:
        raise FileNotFoundError(f"No results.parquet or results.csv found under {in_dir}")

    dfs = []
    for p in files:
        df = _read_results(p)
        df["source_path"] = str(p)
        dfs.append(df)
    runs = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d rows from %d files", len(runs), len(files))

    # Compute deltas for each oracle mode vs raw_only.
    keys = ["task_id", "regime_id", "seed", "n", "xgb_config_id"]
    raw = runs[runs["oracle_mode"] == "raw_only"].copy()
    oracle = runs[runs["oracle_mode"] != "raw_only"].copy()

    raw = raw.rename(
        columns={
            "prauc": "prauc_raw",
            "rocauc": "rocauc_raw",
            "logloss": "logloss_raw",
            "ratio_scale_invariance": "ratio_scale_invariance_raw",
            "product_comp_invariance": "product_comp_invariance_raw",
            "iso_var_ratio": "iso_var_ratio_raw",
            "iso_var_product": "iso_var_product_raw",
        }
    )
    oracle = oracle.rename(
        columns={
            "prauc": "prauc_oracle",
            "rocauc": "rocauc_oracle",
            "logloss": "logloss_oracle",
            "ratio_scale_invariance": "ratio_scale_invariance_oracle",
            "product_comp_invariance": "product_comp_invariance_oracle",
            "iso_var_ratio": "iso_var_ratio_oracle",
            "iso_var_product": "iso_var_product_oracle",
        }
    )

    deltas = oracle.merge(
        raw[
            keys
            + [
                "prauc_raw",
                "rocauc_raw",
                "logloss_raw",
                "ratio_scale_invariance_raw",
                "product_comp_invariance_raw",
                "iso_var_ratio_raw",
                "iso_var_product_raw",
            ]
        ],
        on=keys,
        how="inner",
    )
    deltas["delta_prauc"] = deltas["prauc_oracle"] - deltas["prauc_raw"]
    deltas["delta_rocauc"] = deltas["rocauc_oracle"] - deltas["rocauc_raw"]
    deltas["delta_logloss"] = deltas["logloss_oracle"] - deltas["logloss_raw"]

    # Write machine-readable outputs.
    runs.to_parquet(Path(out_dir) / "runs.parquet", index=False)
    runs.to_csv(Path(out_dir) / "runs.csv", index=False)
    deltas.to_parquet(Path(out_dir) / "deltas.parquet", index=False)
    deltas.to_csv(Path(out_dir) / "deltas.csv", index=False)

    def q10(s: pd.Series) -> float:
        return float(s.quantile(0.1))

    def q90(s: pd.Series) -> float:
        return float(s.quantile(0.9))

    # Summaries.
    agg_specs = {
        "delta_prauc_median": ("delta_prauc", "median"),
        "delta_prauc_p10": ("delta_prauc", q10),
        "delta_prauc_p90": ("delta_prauc", q90),
        "delta_prauc_std": ("delta_prauc", _std0),
        "delta_prauc_iqr": ("delta_prauc", _iqr),
        "prauc_raw_median": ("prauc_raw", "median"),
        "prauc_oracle_median": ("prauc_oracle", "median"),
        "ratio_scale_invariance_raw_median": ("ratio_scale_invariance_raw", "median"),
        "ratio_scale_invariance_oracle_median": ("ratio_scale_invariance_oracle", "median"),
        "product_comp_invariance_raw_median": ("product_comp_invariance_raw", "median"),
        "product_comp_invariance_oracle_median": ("product_comp_invariance_oracle", "median"),
        "iso_var_ratio_raw_median": ("iso_var_ratio_raw", "median"),
        "iso_var_ratio_oracle_median": ("iso_var_ratio_oracle", "median"),
        "iso_var_product_raw_median": ("iso_var_product_raw", "median"),
        "iso_var_product_oracle_median": ("iso_var_product_oracle", "median"),
        "dominance_train_mean_median": ("dominance_train_mean", "median"),
        "dominance_train_p90_median": ("dominance_train_p90", "median"),
        "dominance_test_mean_median": ("dominance_test_mean", "median"),
        "dominance_test_p90_median": ("dominance_test_p90", "median"),
        "n_runs": ("delta_prauc", "size"),
    }

    by_task = (
        deltas.groupby(["task_id", "level", "oracle_mode", "xgb_config_id"], dropna=False)
        .agg(**agg_specs)
        .reset_index()
    )
    by_regime_family = (
        deltas.groupby(["regime_family", "oracle_mode", "xgb_config_id"], dropna=False)
        .agg(**agg_specs)
        .reset_index()
    )
    by_task_n = (
        deltas.groupby(["task_id", "level", "oracle_mode", "xgb_config_id", "n"], dropna=False)
        .agg(**agg_specs)
        .reset_index()
    )
    by_level = (
        deltas.groupby(["level", "oracle_mode", "xgb_config_id"], dropna=False)
        .agg(**agg_specs)
        .reset_index()
        .sort_values(["level", "oracle_mode", "xgb_config_id"])
    )
    by_level_regime_family = (
        deltas.groupby(["level", "regime_family", "oracle_mode", "xgb_config_id"], dropna=False)
        .agg(**agg_specs)
        .reset_index()
        .sort_values(["level", "regime_family", "oracle_mode", "xgb_config_id"])
    )

    # Benchmark-panel stability summaries (computed from existing deltas; no new training).
    win_prefix = "delta_prauc_pos_rate"
    win_task = _win_rate_table(
        deltas,
        group_cols=["task_id", "level", "oracle_mode", "xgb_config_id"],
        value_col="delta_prauc",
        out_prefix=win_prefix,
    )
    by_task = by_task.merge(win_task, on=["task_id", "level", "oracle_mode", "xgb_config_id"], how="left")

    win_regime = _win_rate_table(
        deltas,
        group_cols=["regime_family", "oracle_mode", "xgb_config_id"],
        value_col="delta_prauc",
        out_prefix=win_prefix,
    )
    by_regime_family = by_regime_family.merge(win_regime, on=["regime_family", "oracle_mode", "xgb_config_id"], how="left")

    win_level = _win_rate_table(
        deltas,
        group_cols=["level", "oracle_mode", "xgb_config_id"],
        value_col="delta_prauc",
        out_prefix=win_prefix,
    )
    by_level = by_level.merge(win_level, on=["level", "oracle_mode", "xgb_config_id"], how="left")

    win_level_regime = _win_rate_table(
        deltas,
        group_cols=["level", "regime_family", "oracle_mode", "xgb_config_id"],
        value_col="delta_prauc",
        out_prefix=win_prefix,
    )
    by_level_regime_family = by_level_regime_family.merge(
        win_level_regime, on=["level", "regime_family", "oracle_mode", "xgb_config_id"], how="left"
    )

    # Leave-one-task-out stability range for the median delta within each summary cell.
    loo_prefix = "delta_prauc_median_loo_task"
    loo_level = _loo_median_range_table(
        deltas,
        group_cols=["level", "oracle_mode", "xgb_config_id"],
        loo_col="task_id",
        value_col="delta_prauc",
        out_prefix=loo_prefix,
    )
    by_level = by_level.merge(loo_level, on=["level", "oracle_mode", "xgb_config_id"], how="left")

    loo_regime = _loo_median_range_table(
        deltas,
        group_cols=["regime_family", "oracle_mode", "xgb_config_id"],
        loo_col="task_id",
        value_col="delta_prauc",
        out_prefix=loo_prefix,
    )
    by_regime_family = by_regime_family.merge(loo_regime, on=["regime_family", "oracle_mode", "xgb_config_id"], how="left")

    by_task.to_parquet(Path(out_dir) / "summary_by_task.parquet", index=False)
    by_task.to_csv(Path(out_dir) / "summary_by_task.csv", index=False)
    by_regime_family.to_parquet(Path(out_dir) / "summary_by_regime_family.parquet", index=False)
    by_regime_family.to_csv(Path(out_dir) / "summary_by_regime_family.csv", index=False)
    by_task_n.to_parquet(Path(out_dir) / "summary_by_task_n.parquet", index=False)
    by_task_n.to_csv(Path(out_dir) / "summary_by_task_n.csv", index=False)
    by_level.to_parquet(Path(out_dir) / "summary_by_level.parquet", index=False)
    by_level.to_csv(Path(out_dir) / "summary_by_level.csv", index=False)
    by_level_regime_family.to_parquet(Path(out_dir) / "summary_by_level_regime_family.parquet", index=False)
    by_level_regime_family.to_csv(Path(out_dir) / "summary_by_level_regime_family.csv", index=False)

    meta_src = Path(input_dir) / "run_metadata.json"
    if meta_src.exists():
        shutil.copy(meta_src, Path(out_dir) / "run_metadata.json")

    logger.info("Wrote aggregated outputs to %s", out_dir)
