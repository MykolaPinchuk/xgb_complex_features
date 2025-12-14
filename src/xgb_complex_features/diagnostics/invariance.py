from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from xgboost import XGBClassifier

from xgb_complex_features.dgp.dataset import OracleMode
from xgb_complex_features.dgp.tasks import FittedTask
from xgb_complex_features.models.xgb import predict_proba_positive


@dataclass(frozen=True)
class InvarianceConfig:
    n_diag: int = 2000
    m_c: int = 5
    c_loguniform_low: float = 0.5
    c_loguniform_high: float = 2.0


@dataclass(frozen=True)
class IsoVarianceConfig:
    n_base: int = 500
    m: int = 10
    c_loguniform_low: float = 0.5
    c_loguniform_high: float = 2.0


def _loguniform(rng: np.random.Generator, low: float, high: float, size: tuple[int, ...]) -> np.ndarray:
    if low <= 0 or high <= 0 or high <= low:
        raise ValueError("loguniform bounds must be positive with high > low")
    return np.exp(rng.uniform(np.log(low), np.log(high), size=size))


def _features_from_raw(x_raw: np.ndarray, task: FittedTask, oracle_mode: OracleMode) -> np.ndarray:
    if oracle_mode == "raw_only":
        return x_raw.astype(np.float32, copy=False)
    tf = task.transform(x_raw)
    parts = [x_raw.astype(np.float32, copy=False)]
    if oracle_mode in {"oracle_coords_only", "oracle_all"}:
        parts.append(tf.coords.astype(np.float32, copy=False))
    if oracle_mode in {"oracle_s_only", "oracle_all"}:
        parts.append(tf.s.astype(np.float32, copy=False))
    return np.concatenate(parts, axis=1)


def ratio_scale_invariance(
    *,
    model: XGBClassifier,
    oracle_mode: OracleMode,
    task: FittedTask,
    x_test_raw: np.ndarray,
    rng: np.random.Generator,
    cfg: InvarianceConfig,
) -> float:
    if not task.diagnostics.ratio_coords:
        return float("nan")
    n = min(int(cfg.n_diag), x_test_raw.shape[0])
    if n <= 0:
        return float("nan")

    idx = rng.choice(x_test_raw.shape[0], size=n, replace=False)
    x0 = x_test_raw[idx].astype(np.float64, copy=True)
    p0 = predict_proba_positive(model, _features_from_raw(x0, task, oracle_mode))

    cs = _loguniform(rng, cfg.c_loguniform_low, cfg.c_loguniform_high, size=(n, int(cfg.m_c)))
    deltas = []
    for j in range(cs.shape[1]):
        x = x0.copy()
        c = cs[:, j]
        for rc in task.diagnostics.ratio_coords:
            cols = list(set(rc.numerator_cols) | set(rc.denominator_cols))
            x[:, cols] *= c[:, None]
        pj = predict_proba_positive(model, _features_from_raw(x, task, oracle_mode))
        deltas.append(np.abs(pj - p0))
    return float(np.mean(np.concatenate(deltas)))


def product_comp_invariance(
    *,
    model: XGBClassifier,
    oracle_mode: OracleMode,
    task: FittedTask,
    x_test_raw: np.ndarray,
    rng: np.random.Generator,
    cfg: InvarianceConfig,
) -> float:
    if not task.diagnostics.product_coords:
        return float("nan")
    n = min(int(cfg.n_diag), x_test_raw.shape[0])
    if n <= 0:
        return float("nan")

    idx = rng.choice(x_test_raw.shape[0], size=n, replace=False)
    x0 = x_test_raw[idx].astype(np.float64, copy=True)
    p0 = predict_proba_positive(model, _features_from_raw(x0, task, oracle_mode))

    cs = _loguniform(rng, cfg.c_loguniform_low, cfg.c_loguniform_high, size=(n, int(cfg.m_c)))
    deltas = []
    for j in range(cs.shape[1]):
        x = x0.copy()
        c = cs[:, j]
        for pc in task.diagnostics.product_coords:
            x[:, pc.group_a_cols] *= c[:, None]
            x[:, pc.group_b_cols] *= (1.0 / c)[:, None]
        pj = predict_proba_positive(model, _features_from_raw(x, task, oracle_mode))
        deltas.append(np.abs(pj - p0))
    return float(np.mean(np.concatenate(deltas)))


def iso_coordinate_variance_ratio(
    *,
    model: XGBClassifier,
    oracle_mode: OracleMode,
    task: FittedTask,
    x_test_raw: np.ndarray,
    rng: np.random.Generator,
    cfg: IsoVarianceConfig,
) -> float:
    if not task.diagnostics.ratio_coords:
        return float("nan")
    n = min(int(cfg.n_base), x_test_raw.shape[0])
    if n <= 0:
        return float("nan")

    idx = rng.choice(x_test_raw.shape[0], size=n, replace=False)
    x0 = x_test_raw[idx].astype(np.float64, copy=True)

    cs = _loguniform(rng, cfg.c_loguniform_low, cfg.c_loguniform_high, size=(n, int(cfg.m)))
    vars_ = []
    for i in range(n):
        preds = []
        for j in range(cs.shape[1]):
            x = x0[i : i + 1].copy()
            c = float(cs[i, j])
            for rc in task.diagnostics.ratio_coords:
                cols = list(set(rc.numerator_cols) | set(rc.denominator_cols))
                x[:, cols] *= c
            preds.append(float(predict_proba_positive(model, _features_from_raw(x, task, oracle_mode))[0]))
        vars_.append(float(np.var(preds)))
    return float(np.mean(vars_))


def iso_coordinate_variance_product(
    *,
    model: XGBClassifier,
    oracle_mode: OracleMode,
    task: FittedTask,
    x_test_raw: np.ndarray,
    rng: np.random.Generator,
    cfg: IsoVarianceConfig,
) -> float:
    if not task.diagnostics.product_coords:
        return float("nan")
    n = min(int(cfg.n_base), x_test_raw.shape[0])
    if n <= 0:
        return float("nan")

    idx = rng.choice(x_test_raw.shape[0], size=n, replace=False)
    x0 = x_test_raw[idx].astype(np.float64, copy=True)

    cs = _loguniform(rng, cfg.c_loguniform_low, cfg.c_loguniform_high, size=(n, int(cfg.m)))
    vars_ = []
    for i in range(n):
        preds = []
        for j in range(cs.shape[1]):
            x = x0[i : i + 1].copy()
            c = float(cs[i, j])
            for pc in task.diagnostics.product_coords:
                x[:, pc.group_a_cols] *= c
                x[:, pc.group_b_cols] *= 1.0 / c
            preds.append(float(predict_proba_positive(model, _features_from_raw(x, task, oracle_mode))[0]))
        vars_.append(float(np.var(preds)))
    return float(np.mean(vars_))


def compute_all_invariance(
    *,
    model: XGBClassifier,
    oracle_mode: OracleMode,
    task: FittedTask,
    x_test_raw: np.ndarray,
    rng: np.random.Generator,
    cfg: dict[str, Any],
) -> dict[str, float]:
    inv_cfg = cfg.get("invariance", {}) or {}
    iso_cfg = cfg.get("iso_variance", {}) or {}

    invariance = InvarianceConfig(
        n_diag=int(inv_cfg.get("n_diag", 2000)),
        m_c=int(inv_cfg.get("m_c", 5)),
        c_loguniform_low=float(inv_cfg.get("c_loguniform_low", 0.5)),
        c_loguniform_high=float(inv_cfg.get("c_loguniform_high", 2.0)),
    )
    iso = IsoVarianceConfig(
        n_base=int(iso_cfg.get("n_base", 500)),
        m=int(iso_cfg.get("m", 10)),
        c_loguniform_low=float(inv_cfg.get("c_loguniform_low", 0.5)),
        c_loguniform_high=float(inv_cfg.get("c_loguniform_high", 2.0)),
    )

    return {
        "ratio_scale_invariance": ratio_scale_invariance(
            model=model,
            oracle_mode=oracle_mode,
            task=task,
            x_test_raw=x_test_raw,
            rng=rng,
            cfg=invariance,
        ),
        "product_comp_invariance": product_comp_invariance(
            model=model,
            oracle_mode=oracle_mode,
            task=task,
            x_test_raw=x_test_raw,
            rng=rng,
            cfg=invariance,
        ),
        "iso_var_ratio": iso_coordinate_variance_ratio(
            model=model,
            oracle_mode=oracle_mode,
            task=task,
            x_test_raw=x_test_raw,
            rng=rng,
            cfg=iso,
        ),
        "iso_var_product": iso_coordinate_variance_product(
            model=model,
            oracle_mode=oracle_mode,
            task=task,
            x_test_raw=x_test_raw,
            rng=rng,
            cfg=iso,
        ),
    }
