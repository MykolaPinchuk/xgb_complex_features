from __future__ import annotations

from pathlib import Path

__all__ = ["__version__"]

__version__ = "0.1.0"

# Development convenience: allow `python -m xgb_complex_features` from repo root
# without requiring `pip install -e .` by adding the src package directory to
# this package's module search path.
_repo_root = Path(__file__).resolve().parent.parent
_src_pkg = _repo_root / "src" / "xgb_complex_features"
if _src_pkg.is_dir():
    __path__.append(str(_src_pkg))  # type: ignore[name-defined]
