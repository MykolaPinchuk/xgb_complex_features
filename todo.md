# TODO

## Completed (this iteration)

1. **Result slicing/visualization**
   - Added heatmap + ranked tables from `summary_by_level*` and `summary_by_level_regime_family*`.
   - Added per-level plots correlating Δ with dominance/invariance inside the Markdown report.

2. **Capacity & seed sweeps**
   - Added multiple XGB configs to test capacity sensitivity (smoke + exp_default).
   - Added variance reporting (std/IQR) for Δ and expanded smoke to 2 seeds.

3. **Diagnostics expansion**
   - Added per-level invariance/iso-variance/dominance medians to the aggregated tables.
   - Added relative Δ% tables (oracle vs raw) in the report.

4. **Configuration & reproducibility polish**
   - Pinned dependency versions in `pyproject.toml` / `requirements.txt` and documented the install workflow.
   - Added a `workflow` CLI command and Makefile targets (run → aggregate → report).
   - Documented expected prevalence behavior under shift regimes.

## Future ideas

- Add more regimes (e.g., `beta_scaled`) and/or an `n_values` sweep in `exp_default` to study sample complexity curves.
- Consider additional shortcut diagnostics (e.g., SHAP, feature-importance stability) if needed.
