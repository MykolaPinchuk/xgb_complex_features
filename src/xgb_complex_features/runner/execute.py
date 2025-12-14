from __future__ import annotations

import json
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from time import perf_counter

import pandas as pd
from joblib import Parallel, delayed

from xgb_complex_features.config import load_yaml, resolve_output_paths
from xgb_complex_features.dgp.dataset import OracleMode, build_features, generate_dataset
from xgb_complex_features.diagnostics.dominance import compute_dominance
from xgb_complex_features.diagnostics.invariance import compute_all_invariance
from xgb_complex_features.models.xgb import compute_metrics, predict_proba_positive, train_xgb_classifier
from xgb_complex_features.runner.grid import DatasetSpec, iter_dataset_specs
from xgb_complex_features.utils import ensure_dir, make_rng, write_json

logger = logging.getLogger(__name__)


def _run_one_dataset(
    *,
    cfg: dict[str, Any],
    spec: DatasetSpec,
    root_seed: int,
) -> list[dict[str, Any]]:
    data_cfg = cfg.get("data", {}) or {}
    label_cfg = cfg.get("label", {}) or {}
    splits_cfg = cfg.get("splits", {}) or {}
    diagnostics_cfg = cfg.get("diagnostics", {}) or {}

    xgb_configs = cfg.get("xgb_configs", []) or []
    oracle_modes = cfg.get("oracle_modes", []) or []
    if not xgb_configs:
        raise ValueError("Config must include xgb_configs")
    if not oracle_modes:
        raise ValueError("Config must include oracle_modes")

    dataset_seed = int(root_seed) + int(spec.seed)
    ds = generate_dataset(
        n=spec.n,
        seed=dataset_seed,
        task_cfg=spec.task,
        regime_cfg=spec.regime,
        data_cfg=data_cfg,
        label_cfg=label_cfg,
        splits_cfg=splits_cfg,
    )

    rows: list[dict[str, Any]] = []
    for xgb_cfg in xgb_configs:
        xgb_config_id = str(xgb_cfg["id"])
        params = deepcopy(xgb_cfg.get("params", {}) or {})

        for oracle_mode in oracle_modes:
            oracle_mode = str(oracle_mode)
            if oracle_mode not in {"raw_only", "oracle_s_only", "oracle_coords_only", "oracle_all"}:
                raise ValueError(f"Unknown oracle_mode: {oracle_mode}")
            oracle_mode_t: OracleMode = oracle_mode  # type: ignore[assignment]

            x, _ = build_features(ds, oracle_mode=oracle_mode_t)
            tr, va, te = ds.splits.train, ds.splits.val, ds.splits.test
            x_train, y_train = x[tr], ds.y[tr]
            x_val, y_val = x[va], ds.y[va]
            x_test, y_test = x[te], ds.y[te]

            model_seed = int(
                make_rng(
                    root_seed,
                    spec.seed,
                    spec.task["id"],
                    spec.regime["id"],
                    spec.n,
                    oracle_mode,
                    xgb_config_id,
                    "model_seed",
                ).integers(0, 2**31 - 1)
            )
            fit = train_xgb_classifier(
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                params=params,
                seed=model_seed,
            )

            p_test = predict_proba_positive(fit.model, x_test)
            metrics = compute_metrics(y_test, p_test)

            rng_diag = make_rng(
                root_seed,
                spec.seed,
                spec.task["id"],
                spec.regime["id"],
                spec.n,
                oracle_mode,
                xgb_config_id,
                "diag",
            )
            invariance = compute_all_invariance(
                model=fit.model,
                oracle_mode=oracle_mode_t,
                task=ds.task,
                x_test_raw=ds.x_raw[te],
                rng=rng_diag,
                cfg=diagnostics_cfg,
            )

            dom_train = compute_dominance(x_raw=ds.x_raw, idx=tr, sum_groups=ds.task.diagnostics.sum_groups)
            dom_test = compute_dominance(x_raw=ds.x_raw, idx=te, sum_groups=ds.task.diagnostics.sum_groups)

            row: dict[str, Any] = {
                "task_id": ds.metadata["task_id"],
                "level": ds.metadata["task_level"],
                "task_kind": ds.metadata["task_kind"],
                "regime_id": ds.metadata["regime_id"],
                "regime_family": ds.metadata["regime_family"],
                "seed": int(spec.seed),
                "n": int(spec.n),
                "oracle_mode": oracle_mode,
                "model_variant": "raw" if oracle_mode == "raw_only" else "oracle",
                "xgb_config_id": xgb_config_id,
                "best_iteration": fit.best_iteration,
                "best_score": fit.best_score,
                **metrics,
                **invariance,
                "prevalence_train": ds.metadata["prevalence_train"],
                "prevalence_val": ds.metadata["prevalence_val"],
                "prevalence_test": ds.metadata["prevalence_test"],
                "sigma": ds.metadata.get("sigma"),
                "rho": ds.metadata.get("rho"),
                "mixture_json": json.dumps(ds.metadata.get("mixture"), sort_keys=True)
                if ds.metadata.get("mixture") is not None
                else None,
                "shift_type": ds.metadata.get("shift_type"),
                "shift_cols_json": json.dumps(ds.metadata.get("shift_cols"), sort_keys=True)
                if ds.metadata.get("shift_cols") is not None
                else None,
                "dominance_train_mean": dom_train["dominance_mean"],
                "dominance_train_median": dom_train["dominance_median"],
                "dominance_train_p90": dom_train["dominance_p90"],
                "dominance_test_mean": dom_test["dominance_mean"],
                "dominance_test_median": dom_test["dominance_median"],
                "dominance_test_p90": dom_test["dominance_p90"],
                "dominance_train_groups_json": dom_train["dominance_groups_json"],
                "dominance_test_groups_json": dom_test["dominance_groups_json"],
            }
            rows.append(row)

    return rows


def run_experiment(*, config_path: str) -> Path:
    start_wall = datetime.now()
    start_perf = perf_counter()

    cfg = load_yaml(config_path)
    exp = cfg.get("experiment", {}) or {}
    root_seed = int(exp.get("root_seed", 0))

    paths = resolve_output_paths(cfg, config_path=config_path)
    base_dir = paths.base_dir

    run_dir = base_dir
    if run_dir.exists() and not paths.overwrite:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base_dir / stamp
    ensure_dir(run_dir)

    # Persist resolved config alongside outputs.
    write_json(run_dir / "config_resolved.json", cfg)
    (run_dir / "config_source.yaml").write_text(Path(config_path).read_text(encoding="utf-8"), encoding="utf-8")

    runner_cfg = cfg.get("runner", {}) or {}
    n_jobs = int(runner_cfg.get("n_jobs", 1))
    if n_jobs == 0:
        n_jobs = 1

    specs = list(iter_dataset_specs(cfg))
    logger.info("Running %d dataset specs (n_jobs=%d) into %s", len(specs), n_jobs, run_dir)

    if n_jobs == 1:
        all_rows: list[dict[str, Any]] = []
        for i, spec in enumerate(specs, start=1):
            logger.info("Dataset %d/%d: task=%s regime=%s n=%d seed=%d", i, len(specs), spec.task["id"], spec.regime["id"], spec.n, spec.seed)
            all_rows.extend(_run_one_dataset(cfg=cfg, spec=spec, root_seed=root_seed))
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_run_one_dataset)(cfg=cfg, spec=spec, root_seed=root_seed) for spec in specs
        )
        all_rows = [row for part in results for row in part]

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("No results produced.")

    out_formats = set(paths.formats)
    if "parquet" in out_formats:
        df.to_parquet(run_dir / "results.parquet", index=False)
    if "csv" in out_formats:
        df.to_csv(run_dir / "results.csv", index=False)

    logger.info("Wrote %d rows", len(df))

    end_wall = datetime.now()
    duration = perf_counter() - start_perf
    run_meta = {
        "config_path": str(Path(config_path).resolve()),
        "run_dir": str(run_dir.resolve()),
        "start_time": start_wall.isoformat(),
        "end_time": end_wall.isoformat(),
        "duration_seconds": duration,
        "n_dataset_specs": len(specs),
        "n_results_rows": len(df),
        "n_tasks": len(cfg.get("tasks", [])),
        "n_regimes": len(cfg.get("regimes", [])),
        "n_oracle_modes": len(cfg.get("oracle_modes", []) or []),
        "n_jobs": n_jobs,
    }
    write_json(run_dir / "run_metadata.json", run_meta)
    return run_dir
