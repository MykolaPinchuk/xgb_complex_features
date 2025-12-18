# Working-Paper Notes: Can XGBoost Learn Ratio/Product Structure From Raw Variables?

- Generated: 2025-12-14T11:12:30-08:00
- This directory is intended as handoff material for a paper-writing agent.
- Companion artifacts (this directory is intentionally scoped to **exp_default** only):
  - `paper_draft/2025-12-14_110315/tables/exp_default__*.csv` (machine-readable tables)
  - `paper_draft/2025-12-14_110315/plots.pdf` (all exp_default plots in a single PDF)
- **Do not add other experiment outputs here.** If new runs are needed (e.g., exp_all_levels, smoke), keep them under `runs/` and reference them explicitly in future drafts rather than copying artifacts into this `paper_draft` folder.

## 1) Problem statement (what we’re testing)

We want to measure how well gradient-boosted decision trees (XGBoost classifier) can learn **ratio/product structure** and their compositions when trained only on **raw positive variables** `x_0..x_{d-1}`, versus an oracle variant that is given the **constructed true feature(s)** that the DGP uses.

Core question: **Does “raw-only XGB” rediscover the constructed feature(s)** under varied nuisance distributions and test-time shifts, or does it rely on shortcuts that break under distribution change?

Key quantity throughout: **Δ PRAUC = PRAUC(oracle) − PRAUC(raw_only)** on a held-out test split.

## 2) What’s already implemented and measured

### Pipeline capabilities (already in code)

- Synthetic DGP for positive raw features with correlation, tail-heaviness, mixtures, and test-time shifts.
- Task suite (Levels 1–7, used as motif labels) covering log-ratios/log-products, sum-based coordinates, contrasts, coordinate interactions, non-monotone transforms, and gating.
- Oracle feature modes:
  - `raw_only`
  - `oracle_coords_only` (adds constructed coordinates like `u`, `u1`, `u2`, gating indicator; excludes final signal)
  - `oracle_s_only` (adds the final signal `s` used to generate labels)
- Evaluation:
  - Primary metric: PRAUC on test
  - Diagnostics: ratio/product invariance perturbations and iso-coordinate variance
  - Dominance stats for sum-based tasks (`max(x_i)/sum(x_i)`) to flag shortcut regimes

### Note on level framing (proposed wording for the paper)

The current code assigns each task an integer `level`. This is a convenient taxonomy label. It is not a formal ordering by function class or “compositional depth”. In particular, the tasks labeled Level 4 and Level 5 are all coordinate interaction tasks of the form $s=u_1u_2$. The key scientific distinction is not the polynomial degree. It is the invariance type and aggregation geometry that generate different shortcut opportunities under the regime panel.

If designing a complexity framework from scratch, it is cleaner to separate a small set of signal-form levels from orthogonal tags:

1. **Single-coordinate monotone**: $s=u$ (ratio-like or product-like coordinate).
2. **Additive contrasts**: $s=\sum_k \pm u_k$ (still “linear in $\log x$” for lognormal inputs, but higher arity and cancellation).
3. **Sum-aggregation coordinates**: coordinates use $\log(x_1+x_2)$ or $\log((x_1+x_2)(x_3+x_4))$, which introduces max-dominance effects.
4. **Coordinate interactions**: $s=\sum_k u_k v_k$ (quadratic or bilinear in $\log x$). This includes ratio×ratio, product×product, and ratio×product. These are subtypes with different invariances, not strictly different complexity levels.
5. **Non-monotone shapes**: $s=\phi(u)$ or $s=\phi(u_1u_2)$ with non-monotone $\phi$ (multi-threshold decision geometry).
6. **Conditional or gated structure**: $s=\mathbb{1}\{g>t\}s_1+(1-\mathbb{1}\{g>t\})s_2$ (piecewise mixture of simpler forms).

Suggested wording: treat “levels” as a taxonomy of motifs. Present Level 4 and Level 5 together as an “interaction family” with ratio, product, and hybrid variants. State explicitly that the level index is not a strict complexity ranking. This keeps the narrative consistent with the empirical differences across regimes, which are driven by distributional geometry and invariance violations rather than by degree alone.

### Existing experiment runs (already completed)

All tables copied into this directory are from **`exp_default`** (12 tasks × 6 regimes × 3 seeds × 3 XGB configs at n=30k). Broader panels (e.g., `exp_all_levels_10k`) remain under `runs/…` but are intentionally omitted from `paper_draft/2025-12-14_110315/` to avoid mixing result sets.

## 3) Main results (high-level)

### 3.1 Interaction tasks show large oracle gaps

In the main run (`exp_default`), interaction tasks of the form $s=u_1u_2$ show a large drop for raw-only relative to oracle. In the current config labeling, the largest median gap appears at level=4 (ratio×ratio and product×product):

- Level 4 has the **largest relative drop** (raw vs oracle) among labeled levels:
  - Example (median across all regimes/seeds for one config): raw-only median PRAUC ≈ **0.113**, oracle median PRAUC ≈ **0.245–0.249** (relative drop ≈ **~53–55%** depending on oracle mode and config).

This is consistent with the hypothesis that learning multiplicative interactions of log-coordinates is nontrivial for raw-only trees under nuisance variation, while the oracle feature makes it easy.

(See: `paper_draft/2025-12-14_110315/tables/exp_default__summary_by_level.csv`.)

### 3.2 Invariance-preserving shifts can catastrophically break raw-only models

The “shift_preserve” regime is designed to preserve the true ratio/product coordinates while changing the raw magnitudes. Under this stress test, we observe extreme failure cases for raw-only:

- Largest Δ example in `exp_default`:
  - Task: `l4_product_x_product` (Level 4), regime: `shift_preserve_c5`
  - raw-only PRAUC ≈ **0.0787**, oracle PRAUC ≈ **0.914**, Δ ≈ **0.835**

This suggests raw-only models are exploiting magnitude/covariate shortcuts that do not respect the underlying invariances.

(See: `paper_draft/2025-12-14_110315/tables/exp_default__deltas.csv` for row-level detail.)

### 3.3 Oracle features measurably improve invariance diagnostics

Across delta rows, oracle feature variants reduce invariance error and iso-coordinate variance:

- `exp_default` diagnostic medians (aggregated over runs):
  - invariance error drops from ~**0.011** (raw) to ~**0.0014–0.0034** (oracle)
  - iso-coordinate variance drops from ~**2.8e-4–3.0e-4** (raw) to ~**7.6e-6–3.9e-5** (oracle)

(See: `paper_draft/2025-12-14_110315/tables/exp_default__summary_by_level.csv`.)

### 3.4 Not all tasks benefit from oracle (negative Δ exists)

There are conditions where oracle feature modes are slightly worse than raw-only (negative Δ). Example from `exp_default` bottom-Δ list:

- `l2_ratio_of_sums` under `shift_naive_c5` shows Δ down to about **−0.07** in some configs/seeds.

This matters for a paper narrative: it prevents overclaiming and suggests interactions between model capacity, early stopping, and feature scaling.

(See: `paper_draft/2025-12-14_110315/tables/exp_default__deltas.csv`.)

## 4) What plots we already have (for a paper draft)

The plots (exp_default only) reside in `paper_draft/2025-12-14_110315/figures/` and are bundled in `paper_draft/2025-12-14_110315/plots.pdf`. For the draft paper, the most important figures to include early are:

- Δ distributions by task (boxplot) for `exp_default` (Fig 1)
- Level × regime heatmap for `exp_default` (Fig 5)
- Invariance-vs-Δ scatter for `exp_default` (Fig 3)

## 5) Interpretation / framing ideas for the paper agent

Useful framing (supported by the existing runs):

- Trees can fit complicated decision rules, but **do not necessarily learn the intended invariances** from raw variables alone under distribution shift.
- Providing explicit constructed features (coordinates or signal) can act as an **inductive bias injection** that restores invariance and dramatically improves generalization on interaction-heavy tasks.
- The “shift_preserve” regime is a strong lens: it isolates whether the model learns the invariant coordinate vs magnitude shortcuts.

Suggested paper narrative arc:

1. Define the benchmark + task families (monotone coordinates, sum-based coordinates, contrasts, interactions, non-monotone, gated).
2. Show that on benign regimes, raw-only may look “okay”, but under invariance-preserving shift it can collapse.
3. Show that oracle features (especially `s`) close the gap, and diagnostics corroborate invariance.
4. Discuss how this relates to feature engineering/representation learning and robustness.

## 6) Known limitations of current evidence (important for a draft)

Even for an arXiv working paper, we should explicitly acknowledge:

- No `n`-sweep yet in `exp_default` (single n=30k); sample complexity curves are missing.
- Limited model family: XGBoost only; no linear baseline, no MLP baseline, no monotone constraints, no calibrated/log-feature preprocessing baseline.
- Some regime factors are confounded in `exp_default` (sigma and rho are not independently swept).
- `exp_all_levels_10k` has a single seed and a single XGB config (good breadth, weaker uncertainty quantification).

## 7) “Next results” that most improve paper credibility (if we choose to add them)

High-value additions that directly strengthen claims:

1. **n-sweep** for at least Levels 3–6 (e.g., n ∈ {5k, 10k, 30k, 100k}) to establish sample complexity.
2. **Independent sigma/rho sweep** (fix sigma and vary rho; fix rho and vary sigma) for the main run.
3. **More seeds** (≥10) on a smaller subset (e.g., Levels 4–6) to tighten uncertainty.
4. **Baselines/ablations**:
   - raw-only on `log(x)` vs raw-only on `x` (to test coordinate choice sensitivity)
   - logistic regression / linear model on oracle coordinates (sanity upper bound)
   - a simple MLP baseline (to compare tree vs neural representation)

## 8) Where the raw data lives (for paper agent + future analysis)

- Full run rows:
  - `runs/exp_default/results.parquet`
  - `runs/exp_all_levels_10k/results.parquet`
  - `runs/smoke/results.parquet`
- Aggregated summaries and deltas:
  - `runs/exp_default/aggregate/`
  - `runs/exp_all_levels_10k/aggregate/`
  - `runs/smoke/aggregate/`
