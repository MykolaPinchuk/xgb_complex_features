from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    from scipy.stats import beta as _beta
    from scipy.stats import norm as _norm

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False
    _beta = None
    _norm = None


MarginalKind = Literal["lognormal", "beta_scaled"]


@dataclass(frozen=True)
class LogNormalSpec:
    sigma: float
    mixture: dict | None = None


@dataclass(frozen=True)
class BetaScaledSpec:
    alpha: float = 2.0
    beta: float = 5.0
    scale: float = 1.0


def _sigma_per_row(n: int, sigma: float, mixture: dict | None, rng: np.random.Generator) -> np.ndarray:
    if mixture is None:
        return np.full(n, float(sigma), dtype=np.float64)

    p_low = float(mixture.get("p_low", 0.9))
    sigma_low = float(mixture.get("sigma_low", sigma))
    sigma_high = float(mixture.get("sigma_high", sigma))
    if not (0.0 < p_low < 1.0):
        raise ValueError("mixture.p_low must be in (0,1)")
    choose_low = rng.random(n) < p_low
    out = np.where(choose_low, sigma_low, sigma_high).astype(np.float64, copy=False)
    return out


def lognormal_from_latent(z: np.ndarray, sigma: float, mixture: dict | None, rng: np.random.Generator) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    s = _sigma_per_row(z.shape[0], sigma=float(sigma), mixture=mixture, rng=rng)
    x = np.exp(z * s[:, None])
    return x


def beta_scaled_from_latent(
    z: np.ndarray,
    alpha: float,
    beta: float,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    alpha = float(alpha)
    beta = float(beta)
    scale = float(scale)

    if _HAVE_SCIPY:
        u = _norm.cdf(z)  # type: ignore[union-attr]
        u = np.clip(u, 1e-12, 1.0 - 1e-12)
        x01 = _beta.ppf(u, alpha, beta)  # type: ignore[union-attr]
        x = x01 * scale
        return np.clip(x, 1e-12, None)

    # Fallback: independent beta; correlation is not preserved without SciPy.
    x = rng.beta(alpha, beta, size=z.shape) * scale
    return np.clip(x, 1e-12, None)


def make_positive_features(
    z: np.ndarray,
    marginal_kind: MarginalKind,
    *,
    sigma: float,
    mixture: dict | None,
    beta_spec: dict | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if marginal_kind == "lognormal":
        return lognormal_from_latent(z, sigma=sigma, mixture=mixture, rng=rng)
    if marginal_kind == "beta_scaled":
        spec = beta_spec or {}
        return beta_scaled_from_latent(
            z,
            alpha=float(spec.get("alpha", 2.0)),
            beta=float(spec.get("beta", 5.0)),
            scale=float(spec.get("scale", 1.0)),
            rng=rng,
        )

    raise ValueError(f"Unknown marginal_kind: {marginal_kind}")
