from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def stable_int_hash(text: str) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def make_rng(*parts: Any) -> np.random.Generator:
    seeds = []
    for p in parts:
        if isinstance(p, (int, np.integer)):
            seeds.append(int(p))
        else:
            seeds.append(stable_int_hash(str(p)))
    ss = np.random.SeedSequence(seeds)
    return np.random.default_rng(ss)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
