from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from xgboost import XGBClassifier


@dataclass(frozen=True)
class FitResult:
    model: XGBClassifier
    best_iteration: int | None
    best_score: float | None


def train_xgb_classifier(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any],
    seed: int,
) -> FitResult:
    params = dict(params)
    early_stopping_rounds = params.pop("early_stopping_rounds", 50)
    if early_stopping_rounds is not None:
        params["early_stopping_rounds"] = int(early_stopping_rounds)

    model = XGBClassifier(
        random_state=int(seed),
        **params,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )

    best_iteration = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)
    return FitResult(model=model, best_iteration=best_iteration, best_score=best_score)


def predict_proba_positive(model: XGBClassifier, x: np.ndarray) -> np.ndarray:
    p = model.predict_proba(x)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("predict_proba must return shape (n, 2)")
    return p[:, 1].astype(np.float64, copy=False)


def compute_metrics(y_true: np.ndarray, p_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    p_pred = np.asarray(p_pred).reshape(-1)
    out: dict[str, float] = {}
    out["prauc"] = float(average_precision_score(y_true, p_pred))
    try:
        out["rocauc"] = float(roc_auc_score(y_true, p_pred))
    except Exception:
        out["rocauc"] = float("nan")
    try:
        out["logloss"] = float(log_loss(y_true, p_pred, labels=[0, 1]))
    except Exception:
        out["logloss"] = float("nan")
    return out
