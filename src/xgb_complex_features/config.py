from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping at top level: {p}")
    return data


@dataclass(frozen=True)
class Paths:
    base_dir: Path
    formats: tuple[str, ...]
    overwrite: bool


def resolve_output_paths(cfg: dict[str, Any], config_path: str | Path) -> Paths:
    out = cfg.get("output", {})
    base_dir = Path(out.get("base_dir", "runs/exp"))
    if not base_dir.is_absolute():
        base_dir = (Path.cwd() / base_dir).resolve()
    formats = tuple(out.get("formats", ["parquet"]))
    overwrite = bool(out.get("overwrite", False))
    return Paths(base_dir=base_dir, formats=formats, overwrite=overwrite)
