# xgb_complex_features

Synthetic benchmark to measure how well XGBoost learns ratio/product structure from raw variables vs oracle features.

## Installation

Create a virtual environment (Python 3.10+) and install the pinned dependencies in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

This pulls in exact versions of numpy/pandas/xgboost/etc. declared in `pyproject.toml` / `requirements.txt`, which keeps the synthetic pipeline and diagnostics reproducible across machines.

## Quickstart

Run the smoke suite:

```bash
python -m xgb_complex_features smoke
```

This generates `runs/smoke/results.*`, aggregates into `runs/smoke/aggregate/`, and writes `runs/smoke/report.md`.

## Terminology

- **Raw variables**: the simulated positive columns `x_0..x_{d-1}`.
- **Constructed true features**: deterministic functions of raw variables used by the DGP (e.g., `u = log(x1/(x2+eps))`).
- **Signal `s`**: the final scalar fed into the label link function (e.g., `s = u1*u2` or a non-monotone transform of `u`).

Oracle modes in configs:

- `raw_only`: train XGBoost on raw variables only.
- `oracle_coords_only`: raw variables + constructed true features (`u`, `u1`, `u2`, gate indicator, etc.), but **not** `s`.
- `oracle_s_only`: raw variables + the ground-truth signal `s` (this is *not* the label; it is a deterministic feature used to generate the label).
- `oracle_all`: raw + constructed true features + `s`.

Run a configurable experiment:

```bash
python -m xgb_complex_features run --config configs/exp_default.yaml
python -m xgb_complex_features aggregate --input runs/exp_default --output out/exp_default
python -m xgb_complex_features report --input out/exp_default --output out/exp_default/report.md
```

For an “all levels” quick run (covers Levels 1–7 at n=10k):

```bash
python -m xgb_complex_features run --config configs/exp_all_levels_10k.yaml
python -m xgb_complex_features aggregate --input runs/exp_all_levels_10k --output runs/exp_all_levels_10k/aggregate
python -m xgb_complex_features report --input runs/exp_all_levels_10k/aggregate --output runs/exp_all_levels_10k/report.md
```

## Notes on label prevalence

Label prevalence is calibrated before splitting to hit ~5% positives overall (via `label.target_prevalence`). Because some regimes introduce test-time shifts, you should expect ~5% prevalence on train/val and roughly 6% on the held-out test split in the default configs. Adjust `target_prevalence` in configs if you need different baselines.

## Makefile helpers

After installation you can also use the provided shortcuts:

```bash
make smoke         # run+aggregate+report for configs/smoke.yaml
make exp_default   # run+aggregate+report for configs/exp_default.yaml
```

Both targets will emit artifacts under `runs/<name>/` and refresh the corresponding Markdown reports automatically.
