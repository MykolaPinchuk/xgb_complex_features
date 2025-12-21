# Agent log: v7_fixed review (targeted fixes check)

- Timestamp: 2025-12-20T16:46:31-08:00
- Agent: Codex CLI
- Model: GPT-5.2
- Artifact reviewed: `paper_draft/2025-12-14_110315/xgb_invariant_features_updated_v7_fixed.pdf`

## Verdict

The two requested fixes are largely in place and the draft is reasonable to release as an initial working paper.

## Fix 1: coordinate definitions vs implementation

What v7_fixed got right:

- It no longer claims a single global ϵ reused across coordinates.
- Eq. (3) now matches the implemented ratio form: numerator sum is unregularized; denominator has `+ ϵ`.
- Eq. (4) matches the implemented product form: `log(product + ϵ)`.

Remaining mismatch (small but real):

- The draft says stabilizers are computed “on the training split”. In the code, task fitting (including stabilizers) happens **before splitting** and uses the full base dataset (`generate_dataset()` calls `fit_task()` before `_split_indices()`).
  - Easiest paper-only fix: say “computed on the dataset used to define the task” (or “computed prior to splitting”) rather than “training split”.

## Fix 2: Level 7 gating definition

What v7_fixed got right:

- It now describes gating as using a **raw-feature gate** and Appendix A explicitly sets `x_g = x_4`, matching `tasks.py` (`g = x[:, 4]` / `x_base[:, 4]`).

Remaining mismatch (same as above):

- Appendix A describes τ as “training median”; in code the threshold is a quantile computed during `fit_task()` before splitting. Consider changing to “a quantile threshold computed during task fitting”.

## Optional tiny polish (no new results)

- Appendix A “Level 1 ratio” example uses `u_R({0,1},{2,3})`, while the default Level-1 ratio task in `configs/exp_default.yaml` / `tasks.py` uses a 1-vs-1 construction. If Appendix A is meant to be implementation-faithful, update those index sets.
- The Abstract still says “raw mode often fits training data”; if train metrics are not shown, soften/remove.

