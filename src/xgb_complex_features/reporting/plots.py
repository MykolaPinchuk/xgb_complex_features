from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xgb_complex_features.utils import ensure_dir


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_delta_boxplots(deltas: pd.DataFrame, out_dir: str | Path) -> Path:
    out_dir = ensure_dir(out_dir)
    path = Path(out_dir) / "delta_prauc_boxplot.png"

    if deltas.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No delta rows", ha="center", va="center")
        ax.axis("off")
        _save(fig, path)
        return path

    # One box per task per oracle_mode.
    deltas = deltas.copy()
    deltas["label"] = deltas["task_id"].astype(str) + " | " + deltas["oracle_mode"].astype(str)
    labels = sorted(deltas["label"].unique())
    data = [deltas.loc[deltas["label"] == lab, "delta_prauc"].values for lab in labels]

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(labels)), 4))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Δ PRAUC (oracle − raw) by task and oracle mode")
    ax.set_ylabel("Δ PRAUC")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    _save(fig, path)
    return path


def plot_delta_vs_invariance(deltas: pd.DataFrame, out_dir: str | Path) -> Path:
    out_dir = ensure_dir(out_dir)
    path = Path(out_dir) / "delta_vs_invariance.png"

    if deltas.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No delta rows", ha="center", va="center")
        ax.axis("off")
        _save(fig, path)
        return path

    fig, ax = plt.subplots(figsize=(7, 5))
    for oracle_mode, g in deltas.groupby("oracle_mode"):
        x = g["ratio_scale_invariance_raw"].fillna(g["product_comp_invariance_raw"])
        y = g["delta_prauc"]
        ax.scatter(x, y, s=18, alpha=0.7, label=str(oracle_mode))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Invariance error (raw-only; ratio or product)")
    ax.set_ylabel("Δ PRAUC (oracle − raw)")
    ax.set_title("Invariance vs generalization gap")
    ax.legend(fontsize=8)
    _save(fig, path)
    return path


def plot_delta_vs_dominance(deltas: pd.DataFrame, out_dir: str | Path) -> Path:
    out_dir = ensure_dir(out_dir)
    path = Path(out_dir) / "delta_vs_dominance.png"

    if deltas.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No delta rows", ha="center", va="center")
        ax.axis("off")
        _save(fig, path)
        return path

    fig, ax = plt.subplots(figsize=(7, 5))
    for oracle_mode, g in deltas.groupby("oracle_mode"):
        x = g["dominance_train_mean"]
        y = g["delta_prauc"]
        ax.scatter(x, y, s=18, alpha=0.7, label=str(oracle_mode))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Dominance mean (train; max(x)/sum(x) for sum-groups)")
    ax.set_ylabel("Δ PRAUC (oracle − raw)")
    ax.set_title("Sum dominance vs generalization gap")
    ax.legend(fontsize=8)
    _save(fig, path)
    return path


def plot_delta_vs_n(summary_by_task_n: pd.DataFrame, out_dir: str | Path) -> Path:
    out_dir = ensure_dir(out_dir)
    path = Path(out_dir) / "delta_vs_n.png"

    if summary_by_task_n.empty or summary_by_task_n["n"].nunique(dropna=True) <= 1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No n-sweep data", ha="center", va="center")
        ax.axis("off")
        _save(fig, path)
        return path

    df = summary_by_task_n.copy()
    df = df.sort_values(["task_id", "oracle_mode", "n"])

    fig, ax = plt.subplots(figsize=(8, 5))
    for (task_id, oracle_mode), g in df.groupby(["task_id", "oracle_mode"]):
        ax.plot(g["n"], g["delta_prauc_median"], marker="o", linewidth=1.5, label=f"{task_id} | {oracle_mode}")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("n")
    ax.set_ylabel("Median Δ PRAUC (oracle − raw)")
    ax.set_title("Δ vs n (median across regimes/seeds)")
    ax.legend(fontsize=8)
    _save(fig, path)
    return path


def plot_delta_heatmap(summary_level_regime: pd.DataFrame, out_dir: str | Path) -> Path:
    out_dir = ensure_dir(out_dir)
    path = Path(out_dir) / "delta_heatmap.png"

    if summary_level_regime.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No level × regime summary", ha="center", va="center")
        ax.axis("off")
        _save(fig, path)
        return path

    df = summary_level_regime.copy()
    df = df.sort_values(["oracle_mode", "level", "regime_family"])
    modes = df["oracle_mode"].dropna().unique().tolist()
    if not modes:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No oracle modes for heatmap", ha="center", va="center")
        ax.axis("off")
        _save(fig, path)
        return path

    vmin = float(df["delta_prauc_median"].min())
    vmax = float(df["delta_prauc_median"].max())

    nrows = len(modes)
    fig, axes = plt.subplots(nrows, 1, figsize=(8, max(3, 3 * nrows)), sharex=False)
    if nrows == 1:
        axes = [axes]  # type: ignore[list-item]

    for ax, mode in zip(axes, modes):
        sub = df[df["oracle_mode"] == mode]
        pivot = sub.pivot_table(index="level", columns="regime_family", values="delta_prauc_median")
        if pivot.empty:
            ax.text(0.5, 0.5, f"No data for {mode}", ha="center", va="center")
            ax.axis("off")
            continue
        levels = pivot.index.tolist()
        regimes = pivot.columns.tolist()
        im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(regimes)))
        ax.set_xticklabels(regimes, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(levels)))
        ax.set_yticklabels([f"L{int(l)}" if pd.notna(l) else "nan" for l in levels])
        for (i, j), val in np.ndenumerate(pivot.values):
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color="black")
        ax.set_title(f"{mode}: Δ PRAUC median")
    fig.colorbar(im, ax=axes, shrink=0.6, label="Δ PRAUC median")
    _save(fig, path)
    return path


def plot_delta_vs_invariance_by_level(summary_by_level: pd.DataFrame, out_dir: str | Path) -> Path:
    out_dir = ensure_dir(out_dir)
    path = Path(out_dir) / "delta_vs_invariance_level.png"

    if summary_by_level.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No level summary rows", ha="center", va="center")
        ax.axis("off")
        _save(fig, path)
        return path

    df = summary_by_level.copy()
    df["invariance_raw"] = df["ratio_scale_invariance_raw_median"].where(
        df["ratio_scale_invariance_raw_median"].notna(), df["product_comp_invariance_raw_median"]
    )
    df = df.dropna(subset=["invariance_raw", "delta_prauc_median"])
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No invariance metrics available", ha="center", va="center")
        ax.axis("off")
        _save(fig, path)
        return path

    fig, ax = plt.subplots(figsize=(7, 5))
    for oracle_mode, g in df.groupby("oracle_mode"):
        ax.scatter(g["invariance_raw"], g["delta_prauc_median"], s=35, alpha=0.8, label=str(oracle_mode))
        for _, row in g.iterrows():
            ax.text(
                row["invariance_raw"],
                row["delta_prauc_median"],
                f"L{int(row['level'])}" if pd.notna(row["level"]) else "L?",
                fontsize=7,
                alpha=0.7,
            )
    ax.set_xlabel("Per-level invariance error (raw median)")
    ax.set_ylabel("Median Δ PRAUC (oracle − raw)")
    ax.set_title("Δ vs invariance (per level)")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.legend(fontsize=8)
    _save(fig, path)
    return path


def plot_delta_vs_dominance_by_level(summary_by_level: pd.DataFrame, out_dir: str | Path) -> Path:
    out_dir = ensure_dir(out_dir)
    path = Path(out_dir) / "delta_vs_dominance_level.png"

    if summary_by_level.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No level summary rows", ha="center", va="center")
        ax.axis("off")
        _save(fig, path)
        return path

    df = summary_by_level.dropna(subset=["dominance_train_mean_median"])
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No dominance metrics available", ha="center", va="center")
        ax.axis("off")
        _save(fig, path)
        return path

    fig, ax = plt.subplots(figsize=(7, 5))
    for oracle_mode, g in df.groupby("oracle_mode"):
        ax.scatter(g["dominance_train_mean_median"], g["delta_prauc_median"], s=35, alpha=0.8, label=str(oracle_mode))
        for _, row in g.iterrows():
            ax.text(
                row["dominance_train_mean_median"],
                row["delta_prauc_median"],
                f"L{int(row['level'])}" if pd.notna(row["level"]) else "L?",
                fontsize=7,
                alpha=0.7,
            )
    ax.set_xlabel("Median dominance (train)")
    ax.set_ylabel("Median Δ PRAUC (oracle − raw)")
    ax.set_title("Δ vs dominance (per level)")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.legend(fontsize=8)
    _save(fig, path)
    return path
