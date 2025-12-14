from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from xgb_complex_features.dgp.label import calibrate_beta0, sample_labels
from xgb_complex_features.dgp.latent import CorrelationSpec, make_correlation_matrix, sample_latent_normal
from xgb_complex_features.dgp.marginals import make_positive_features
from xgb_complex_features.dgp.tasks import FittedTask, TaskTransform, fit_task
from xgb_complex_features.utils import make_rng


OracleMode = Literal["raw_only", "oracle_s_only", "oracle_coords_only", "oracle_all"]


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class Dataset:
    x_raw: np.ndarray
    y: np.ndarray
    splits: SplitIndices
    task: FittedTask
    task_transform: TaskTransform
    beta0: float
    p_true: np.ndarray
    metadata: dict[str, Any]


def _split_indices(n: int, splits: dict[str, Any], rng: np.random.Generator) -> SplitIndices:
    train_frac = float(splits.get("train", 0.6))
    val_frac = float(splits.get("val", 0.2))
    test_frac = float(splits.get("test", 0.2))
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-8:
        raise ValueError("splits must sum to 1.0")

    perm = rng.permutation(n)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)
    train = perm[:n_train]
    val = perm[n_train : n_train + n_val]
    test = perm[n_train + n_val :]
    return SplitIndices(train=train, val=val, test=test)


def _apply_shift_naive(
    x: np.ndarray,
    test_idx: np.ndarray,
    *,
    c: float,
    subset_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    d = x.shape[1]
    frac = float(subset_fraction)
    if not (0.0 < frac <= 1.0):
        raise ValueError("shift.naive subset_fraction must be in (0,1]")
    k = max(1, int(round(frac * d)))
    cols = np.sort(rng.choice(d, size=k, replace=False))
    x = x.copy()
    x[np.ix_(test_idx, cols)] *= float(c)
    return x, {"shift_cols": cols.tolist()}


def _apply_shift_preserve(
    x: np.ndarray,
    test_idx: np.ndarray,
    *,
    c: float,
    task: FittedTask,
) -> tuple[np.ndarray, dict[str, Any]]:
    d = x.shape[1]
    scales = np.ones(d, dtype=np.float64)

    # Ratio invariance: scale numerator and denominator groups by c.
    for rc in task.diagnostics.ratio_coords:
        cols = set(rc.numerator_cols) | set(rc.denominator_cols)
        for j in cols:
            scales[j] *= float(c)

    # Product invariance: scale group A by c and group B by 1/c.
    for pc in task.diagnostics.product_coords:
        for j in pc.group_a_cols:
            scales[j] *= float(c)
        for j in pc.group_b_cols:
            scales[j] *= 1.0 / float(c)

    x = x.copy()
    x[test_idx, :] *= scales[None, :]
    affected = np.where(np.abs(scales - 1.0) > 1e-12)[0]
    return x, {"shift_cols": affected.tolist(), "shift_scales": scales[affected].tolist()}


def _generate_raw_features(
    n: int,
    *,
    d_total: int,
    d_signal_max: int,
    distractors: dict[str, Any],
    corr_cfg: dict[str, Any],
    regime: dict[str, Any],
    marginal_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_duplicates = int(distractors.get("n_duplicates", 0))
    dup_sigma = float(distractors.get("duplicate_log_noise_sigma", 0.02))
    base_d = d_total - n_duplicates
    if base_d <= 0:
        raise ValueError("d_total must be > n_duplicates")

    n_blocked = int(corr_cfg.get("n_blocked_features", 0))
    block_size = int(corr_cfg.get("block_size", 5))
    blocks = corr_cfg.get("blocks")
    corr_spec = CorrelationSpec(block_size=block_size, n_blocked_features=n_blocked, blocks=blocks)
    rho = float(regime.get("rho", 0.0))
    corr = make_correlation_matrix(base_d, rho=rho, spec=corr_spec)
    z = sample_latent_normal(n, corr=corr, rng=rng)

    marginal_kind = str(regime.get("marginal_kind") or marginal_cfg.get("kind", "lognormal"))
    sigma = float(regime.get("sigma", marginal_cfg.get("sigma", 0.7)))
    mixture = regime.get("mixture")
    beta_spec = regime.get("beta") or marginal_cfg.get("beta")
    x_base = make_positive_features(
        z,
        marginal_kind=marginal_kind,  # type: ignore[arg-type]
        sigma=sigma,
        mixture=mixture,
        beta_spec=beta_spec,
        rng=rng,
    )

    if n_duplicates > 0:
        src = rng.choice(base_d, size=n_duplicates, replace=False if n_duplicates <= base_d else True)
        log_noise = rng.normal(0.0, dup_sigma, size=(n, n_duplicates))
        x_dup = x_base[:, src] * np.exp(log_noise)
        x = np.concatenate([x_base, x_dup], axis=1)
        dup_info = {"duplicate_sources": src.tolist(), "duplicate_log_noise_sigma": dup_sigma}
    else:
        x = x_base
        dup_info = {}

    if x.shape[1] != d_total:
        raise AssertionError(f"Expected d_total={d_total} columns, got {x.shape[1]}")
    if d_signal_max > d_total:
        raise ValueError("d_signal_max must be <= d_total")

    return x.astype(np.float32, copy=False), {"corr_rho": rho, "corr_spec": corr_cfg, **dup_info}


def generate_dataset(
    *,
    n: int,
    seed: int,
    task_cfg: dict[str, Any],
    regime_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    label_cfg: dict[str, Any],
    splits_cfg: dict[str, Any],
) -> Dataset:
    exp_seed = int(seed)
    d_total = int(data_cfg.get("d_total", 120))
    d_signal_max = int(data_cfg.get("d_signal_max", 10))
    distractors = data_cfg.get("distractors", {})
    corr_cfg = data_cfg.get("correlation", {})
    marginal_cfg = data_cfg.get("marginals", {"kind": "lognormal"})

    epsilon_rel = float(label_cfg.get("epsilon_rel", 1e-3))
    nonmono_cfg = label_cfg.get("nonmonotone", {}) or {}
    nonmonotone_mu = float(nonmono_cfg.get("mu", 0.0))
    nonmonotone_delta = float(nonmono_cfg.get("delta", 1.0))
    gating_cfg = label_cfg.get("gating", {}) or {}
    gating_q = float(gating_cfg.get("threshold_quantile", 0.7))

    # Raw features for the full dataset (unshifted).
    rng_x = make_rng(exp_seed, task_cfg["id"], regime_cfg["id"], n, "x")
    x_base, raw_meta = _generate_raw_features(
        n,
        d_total=d_total,
        d_signal_max=d_signal_max,
        distractors=distractors,
        corr_cfg=corr_cfg,
        regime=regime_cfg,
        marginal_cfg=marginal_cfg,
        rng=rng_x,
    )

    # Fit task-level constants (epsilon, gate threshold) on full base dataset.
    task = fit_task(
        task_cfg,
        x_base,
        d_signal_max=d_signal_max,
        epsilon_rel=epsilon_rel,
        nonmonotone_mu=nonmonotone_mu,
        nonmonotone_delta=nonmonotone_delta,
        gating_threshold_quantile=gating_q,
    )

    # Calibrate beta0 using base (unshifted) signal.
    base_tf = task.transform(x_base)
    rng_noise = make_rng(exp_seed, task_cfg["id"], regime_cfg["id"], n, "noise")
    eps_noise = rng_noise.normal(0.0, float(label_cfg.get("sigma_eps", 0.5)), size=n).astype(np.float64)
    beta0 = calibrate_beta0(
        base_tf.s_total,
        eps_noise,
        target_prevalence=float(label_cfg.get("target_prevalence", 0.05)),
        a=float(label_cfg.get("a", 2.0)),
        component_weight=float(label_cfg.get("component_weight", 1.0)),
    )

    # Deterministic split indices.
    rng_split = make_rng(exp_seed, task_cfg["id"], regime_cfg["id"], n, "split")
    splits = _split_indices(n, splits=splits_cfg, rng=rng_split)

    # Apply test-only shift if configured, then recompute task transform on observed features.
    shift = regime_cfg.get("shift") or {}
    shift_type = str(shift.get("type", "none"))
    shift_meta: dict[str, Any] = {"shift_type": shift_type}
    x_obs = x_base
    if shift_type == "none":
        pass
    elif shift_type == "naive":
        x_obs, meta = _apply_shift_naive(
            x_obs,
            splits.test,
            c=float(shift.get("c", 5.0)),
            subset_fraction=float(shift.get("subset_fraction", 0.2)),
            rng=make_rng(exp_seed, task_cfg["id"], regime_cfg["id"], n, "shift_naive"),
        )
        shift_meta.update(meta)
    elif shift_type == "preserve":
        x_obs, meta = _apply_shift_preserve(
            x_obs,
            splits.test,
            c=float(shift.get("c", 5.0)),
            task=task,
        )
        shift_meta.update(meta)
    else:
        raise ValueError(f"Unknown shift.type: {shift_type}")

    task_tf = task.transform(x_obs)

    # Labels from observed signal, using beta0 calibrated on base signal.
    rng_y = make_rng(exp_seed, task_cfg["id"], regime_cfg["id"], n, "y")
    y, p_true = sample_labels(
        task_tf.s_total,
        eps_noise,
        rng=rng_y,
        beta0=beta0,
        a=float(label_cfg.get("a", 2.0)),
        component_weight=float(label_cfg.get("component_weight", 1.0)),
    )

    def _prev(idx: np.ndarray) -> float:
        return float(y[idx].mean()) if idx.size else float("nan")

    meta = {
        "task_id": task.task_id,
        "task_kind": task.kind,
        "task_level": task.level,
        "component_count": task.component_count,
        "regime_id": str(regime_cfg["id"]),
        "regime_family": str(regime_cfg.get("family", "")),
        "sigma": float(regime_cfg.get("sigma", float("nan"))),
        "rho": float(regime_cfg.get("rho", float("nan"))),
        "mixture": regime_cfg.get("mixture"),
        **raw_meta,
        **shift_meta,
        "beta0": beta0,
        "prevalence_train": _prev(splits.train),
        "prevalence_val": _prev(splits.val),
        "prevalence_test": _prev(splits.test),
    }

    return Dataset(
        x_raw=np.asarray(x_obs, dtype=np.float32),
        y=y,
        splits=splits,
        task=task,
        task_transform=task_tf,
        beta0=beta0,
        p_true=p_true,
        metadata=meta,
    )


def build_features(dataset: Dataset, oracle_mode: OracleMode) -> tuple[np.ndarray, list[str]]:
    x_raw = dataset.x_raw
    tf = dataset.task_transform

    raw_names = [f"x_{i}" for i in range(x_raw.shape[1])]
    if oracle_mode == "raw_only":
        return x_raw, raw_names

    parts = [x_raw]
    names = list(raw_names)
    if oracle_mode in {"oracle_coords_only", "oracle_all"}:
        parts.append(tf.coords.astype(np.float32, copy=False))
        names.extend(list(tf.coord_names))
    if oracle_mode in {"oracle_s_only", "oracle_all"}:
        parts.append(tf.s.astype(np.float32, copy=False))
        names.extend(list(tf.s_names))

    x = np.concatenate(parts, axis=1)
    return x, names
