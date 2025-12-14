from __future__ import annotations

import json
from typing import Any

import numpy as np


def _dominance_ratio(x: np.ndarray, cols: tuple[int, ...]) -> np.ndarray:
    block = x[:, cols]
    s = block.sum(axis=1)
    m = block.max(axis=1)
    out = np.divide(m, s, out=np.zeros_like(m, dtype=np.float64), where=s > 0)
    return out


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "p90": float(np.quantile(v, 0.9)),
    }


def compute_dominance(
    *,
    x_raw: np.ndarray,
    idx: np.ndarray,
    sum_groups: tuple[tuple[int, ...], ...],
) -> dict[str, Any]:
    if not sum_groups:
        return {
            "dominance_mean": float("nan"),
            "dominance_median": float("nan"),
            "dominance_p90": float("nan"),
            "dominance_groups_json": None,
        }

    x = np.asarray(x_raw, dtype=np.float64)[idx]
    all_ratios = []
    per_group = []
    for g in sum_groups:
        r = _dominance_ratio(x, cols=g)
        all_ratios.append(r)
        per_group.append({"cols": list(g), **_summary_stats(r)})

    merged = np.concatenate(all_ratios) if all_ratios else np.array([], dtype=np.float64)
    merged_stats = _summary_stats(merged)
    return {
        "dominance_mean": merged_stats["mean"],
        "dominance_median": merged_stats["median"],
        "dominance_p90": merged_stats["p90"],
        "dominance_groups_json": json.dumps(per_group, sort_keys=True),
    }
