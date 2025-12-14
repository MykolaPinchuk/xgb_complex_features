from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class DatasetSpec:
    task: dict[str, Any]
    regime: dict[str, Any]
    seed: int
    n: int


def iter_dataset_specs(cfg: dict[str, Any]) -> Iterator[DatasetSpec]:
    tasks = cfg.get("tasks", [])
    regimes = cfg.get("regimes", [])
    n_values = cfg.get("n_values", [])
    seeds = cfg.get("seeds", [])

    if not tasks:
        raise ValueError("Config must include non-empty tasks")
    if not regimes:
        raise ValueError("Config must include non-empty regimes")
    if not n_values:
        raise ValueError("Config must include non-empty n_values")
    if not seeds:
        raise ValueError("Config must include non-empty seeds")

    for task in tasks:
        for regime in regimes:
            for n in n_values:
                for seed in seeds:
                    yield DatasetSpec(task=task, regime=regime, seed=int(seed), n=int(n))
