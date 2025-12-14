from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorrelationSpec:
    block_size: int
    n_blocked_features: int
    blocks: list[list[int]] | None = None


def _explicit_blocks(spec: CorrelationSpec, d: int) -> list[list[int]]:
    if spec.blocks is not None:
        blocks = [list(map(int, b)) for b in spec.blocks]
        for b in blocks:
            if not b:
                raise ValueError("Empty correlation block is not allowed.")
            if min(b) < 0 or max(b) >= d:
                raise ValueError(f"Block indices out of range for d={d}: {b}")
        return blocks

    if spec.n_blocked_features <= 0:
        return []
    if spec.block_size <= 1:
        raise ValueError("block_size must be >= 2 when n_blocked_features > 0")

    n = min(spec.n_blocked_features, d)
    blocks = []
    for start in range(0, n, spec.block_size):
        end = min(start + spec.block_size, n)
        blocks.append(list(range(start, end)))
    return blocks


def make_correlation_matrix(d: int, rho: float, spec: CorrelationSpec) -> np.ndarray:
    rho = float(rho)
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"rho must be in [0,1): {rho}")

    corr = np.eye(d, dtype=np.float64)
    for block in _explicit_blocks(spec, d):
        m = len(block)
        if m <= 1:
            continue
        if rho >= 1.0:
            raise ValueError("rho must be < 1")
        # Equicorrelation within block.
        for i in range(m):
            for j in range(i + 1, m):
                corr[block[i], block[j]] = rho
                corr[block[j], block[i]] = rho
    return corr


def sample_latent_normal(n: int, corr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    corr = np.asarray(corr, dtype=np.float64)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be a square matrix")
    d = corr.shape[0]

    # Cholesky is fastest; fall back to eigen if numerical issues.
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(corr)
        w = np.clip(w, 1e-12, None)
        L = V @ np.diag(np.sqrt(w))

    z = rng.standard_normal(size=(n, d), dtype=np.float64)
    return z @ L.T
