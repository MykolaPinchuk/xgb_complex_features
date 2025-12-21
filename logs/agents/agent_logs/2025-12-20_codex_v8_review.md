# Agent log: v8 draft review

- Timestamp: 2025-12-20T18:30:05-08:00
- Agent: Codex CLI
- Model: GPT-5.2
- Artifact reviewed: `paper_draft/2025-12-20_171247_panel_stability_update/xgb_invariant_features_updated_v8.pdf`

## Summary verdict

v8 is a meaningful improvement over v7_fixed: it correctly introduces “benchmark panel stability” framing, adds win-rate columns to headline tables, and includes an appendix table of invariance diagnostics.

## Issues to fix (low effort, high priority)

1) **Table 1 win-rate intervals appear truncated**
   - In extracted text, win-rate entries render as `76.9% (68`, `97.2% (85`, etc., i.e., the Wilson interval appears cut off.
   - Table 2 win-rate intervals render fully, so this looks like a Table 1 column-width/typesetting issue.

## Minor paper↔code alignment nits (optional but recommended)

- Stabilizer description still says “computed on the training split”; in code, per-coordinate stabilizers are fit **before splitting** on the full base dataset in `generate_dataset()` → `fit_task()`.
- Level 7 appendix still uses “training median” language for the gate threshold τ; in code it is a quantile threshold computed during task fitting (also before splitting).

## Optional writing nit

- Abstract still claims raw “often fits training data”; unless training metrics are shown, consider softening.

