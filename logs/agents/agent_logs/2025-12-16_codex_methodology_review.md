# Agent log: methodology review

- Timestamp: 2025-12-16T19:29:52-08:00
- Agent: Codex CLI
- Model: GPT-5.2

## Scope

- Reviewed the paper handoff notes in `paper_draft/2025-12-14_110315/writedown.md`.
- Verified how the DGP, shifts, and diagnostics are implemented in:
  - `src/xgb_complex_features/dgp/dataset.py`
  - `src/xgb_complex_features/dgp/tasks.py`
  - `src/xgb_complex_features/dgp/label.py`
  - `src/xgb_complex_features/diagnostics/invariance.py`

## Key methodological strengths

- Clear estimand: Δ PRAUC = PRAUC(oracle) − PRAUC(raw_only) under controlled DGPs.
- “Shift_preserve” is a strong stress test for whether the model learned intended invariances vs magnitude shortcuts.
- Extra diagnostics (invariance perturbations, iso-coordinate variance, sum-dominance) help interpret failures beyond raw accuracy.

## Issues / threats to validity (highest impact)

- **Level-7 gated task implementation mismatch**: current gating uses ratio and product coordinates built on the same raw columns via `fit_task(...)`, so the intended “ratio vs product on different variables” story does not hold. This also breaks the interpretation of the “preserve” shift for this task because the preserve transform composes incompatible scalings when coords overlap.
- **“Preserve” invariance is not exact for ratios** because the ratio coordinate uses an additive epsilon in the denominator; scaling numerator/denominator by `c` changes the effective epsilon term. This is likely small with `epsilon_rel=1e-3`, but it undermines any claim of exact invariance unless addressed.
- **Task difficulty is not normalized across tasks/levels**: signal scale/distribution can vary by task/regime, so “harder level ⇒ larger gap” claims can be confounded by SNR differences unless standardized or bounded with an appropriate upper baseline.

## Recommended next steps (to strengthen scientific claims)

1. Fix the gated task to use disjoint columns for ratio/product (matching the conceptual definition), or exclude Level 7 from “preserve shift” claims until fixed.
2. Make the “preserve” shift exactly preserve the ground-truth coordinate when that is the point (e.g., set epsilon=0 for ratio tasks, or scale epsilon consistently and document the choice).
3. Add a small set of credibility baselines/ablations:
   - raw-only with known signal columns only (separates representation vs feature selection),
   - raw-only with `log(x)` preprocessing,
   - linear model on oracle coords (sanity upper bound).
4. Add uncertainty and identifiability checks:
   - more seeds on a smaller subset (Levels 4–6),
   - a simple n-sweep for sample complexity,
   - an independent sigma/rho sweep (currently confounded in `exp_default`).
