from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


@dataclass(frozen=True)
class LabelParams:
    target_prevalence: float
    sigma_eps: float
    a: float
    component_weight: float


def calibrate_beta0(
    s: np.ndarray,
    eps_noise: np.ndarray,
    *,
    target_prevalence: float,
    a: float,
    component_weight: float,
    max_iter: int = 80,
) -> float:
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    eps_noise = np.asarray(eps_noise, dtype=np.float64).reshape(-1)
    if s.shape != eps_noise.shape:
        raise ValueError("s and eps_noise must have the same shape")

    target = float(target_prevalence)
    if not (0.0 < target < 1.0):
        raise ValueError("target_prevalence must be in (0,1)")

    a = float(a)
    component_weight = float(component_weight)

    def mean_p(beta0: float) -> float:
        eta = beta0 + (a * component_weight) * s + eps_noise
        return float(sigmoid(eta).mean())

    lo, hi = -30.0, 30.0
    while mean_p(lo) > target:
        lo -= 10.0
        if lo < -300:
            break
    while mean_p(hi) < target:
        hi += 10.0
        if hi > 300:
            break

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        m = mean_p(mid)
        if m < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sample_labels(
    s: np.ndarray,
    eps_noise: np.ndarray,
    rng: np.random.Generator,
    *,
    beta0: float,
    a: float,
    component_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    eps_noise = np.asarray(eps_noise, dtype=np.float64).reshape(-1)
    if s.shape != eps_noise.shape:
        raise ValueError("s and eps_noise must have the same shape")

    eta = float(beta0) + (float(a) * float(component_weight)) * s + eps_noise
    p = sigmoid(eta)
    y = rng.binomial(1, p).astype(np.int8, copy=False)
    return y, p
