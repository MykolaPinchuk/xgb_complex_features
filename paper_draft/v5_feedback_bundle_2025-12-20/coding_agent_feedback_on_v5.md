# Coding agent feedback on v5

## Feedback to writer agent (v5 → next revision)

- Keep the v5 framing changes: log-coordinate task definition, “levels are motif labels”, and “preserve shift is an identifiability device”. The practitioner boxes are a real improvement.
- Add one quantitative “why shift” artifact near the start of Results: a small table/figure showing gaps by regime family (static vs `shift_preserve`). v5 explains this well, but the main flow should show the numbers.
- Show `oracle_coords_only` explicitly (not only oracle signal). Practitioners will read “oracle signal” as unrealistic. “Oracle coords” is the closer analogue to feature engineering.
- Remove or relocate the internal “Requested runs for the next iteration” list from the paper body. It reads like internal project management. Move it to an internal memo or rewrite it as a reader-facing “Planned experiments” paragraph.
- Tighten paper↔implementation alignment in two spots:
  - Ensure the coordinate equations match the code (ratio stabilizer placement; whether ε is per-coordinate vs global).
  - Level 7: either mark as provisional everywhere (including headline tables) or drop it from headline summaries until the implementation is fully aligned.
- Add one simple schematic figure (pipeline) to make the framework “click” for non-academic readers.

## Small results to improve the paper (low-effort, high-impact)

Use existing results already in `paper_draft/2025-12-14_110315/tables/`:

- Regime-family breakdown (static vs `shift_preserve`) as a main-text table/figure: `tables/exp_default__summary_by_regime_family.csv`.
- “Feature engineering realism” view: raw vs `oracle_coords_only` vs `oracle_s_only` by level (or by interaction family): `tables/exp_default__summary_by_level.csv`.
- Level × regime-family heatmap (median ΔPRAUC): `tables/exp_default__summary_by_level_regime_family.csv`.
- A short vignette table: top 5 largest Δ runs under `shift_preserve` (from `tables/exp_default__deltas.csv`) to concretize the failure mode.

Minor new runs that strengthen the main claim without exploding scope:

- `raw_only` on `log(x+ε)` baseline (restricted to preserve shift + a few tasks, 1 XGB config, small seed count). This directly tests whether failures are mostly dynamic-range / scale issues.
- Seed expansion on a reduced grid (e.g., Levels 3–5, preserve shift only, 1 config). This yields uncertainty bands for the key claim.

