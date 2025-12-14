# PRD: XGB Ability to Learn Ratio/Product Structure From Raw Variables

## 1. Summary

Build a Python project to benchmark how well XGBoost (binary classifier) can learn **multivariate nonlinear structure**—specifically **ratios** and **products** (including comparisons, interactions, nonmonotonic bands, and piecewise gating)—when trained on **raw variables only**, versus an **oracle** variant that is given the true constructed feature(s).

Key output: for each synthetic task and nuisance distribution regime, quantify the **generalization gap** between raw-only and oracle models, plus **invariance/structure diagnostics** to detect shortcut learning driven by distributional assumptions.

---

## 2. Goals

1. Provide a configurable synthetic DGP framework focused on **ratios and products** (no trig/cyclical).
2. Define feature groups ordered by increasing complexity; each level includes both ratio and product variants.
3. Run experiments across a **distribution panel** (tail-heaviness, correlation, mixture extremes, scale-shift) to avoid distribution-driven conclusions.
4. Compare:

   * **Raw-only XGB** (sees only raw variables)
   * **Oracle XGB** (raw vars + true constructed feature(s) used in the DGP)
5. Produce:

   * Metrics (PRAUC primary) and Δ (oracle − raw)
   * Invariance/structure scores
   * Aggregated summaries across regimes and seeds
   * Plots + machine-readable result tables

---

## 3. Non-goals

* Real datasets or domain-specific feature engineering.
* Feature selection benchmarking.
* Categorical / target encoding experiments (unless added later).
* Trigonometric/cyclic DGP components.
* Hyperparameter optimization beyond a small fixed set + optional capacity sweeps.

---

## 4. Primary questions answered

1. Does raw-only XGB recover ratio/product structure under varied input distributions?
2. Which structures are most brittle (large Δ, poor invariance) as nuisance conditions change?
3. Do “harder” function classes (comparisons, interactions, nonmonotonic bands, gating) increase sample/capacity needs?

---

## 5. Definitions

### 5.1 DGP template (common)

* Generate latent: (z \sim \mathcal N(0,\Sigma)), with block correlation.
* Map to raw **positive** variables (default):
  (x_i = \exp(\sigma z_i))
* Construct signal components (s_k = f_k(x_{S_k}))
* Linear predictor:
  (\eta = \beta_0 + \sum_k w_k s_k + \epsilon,\ \epsilon\sim\mathcal N(0,\sigma_\epsilon^2))
* Label: (y \sim \text{Bernoulli}(\sigma(\eta)))
* Choose (\beta_0) by binary-search to hit target prevalence (e.g., 5%).

### 5.2 Oracle vs raw-only

* Raw-only features: all generated raw columns (x_1,\dots,x_d)
* Oracle features: raw-only + the true constructed coordinate(s) (u) and/or (s) used in DGP components

### 5.3 Primary metric

* PRAUC (Average Precision) on Test split

### 5.4 Key report quantity

* (\Delta = \text{PRAUC}*{oracle} - \text{PRAUC}*{raw})

---

## 6. Feature groups (ordered by complexity)

Notation: (\epsilon) stabilizes denominators and logs (configurable). Use log-coordinates by default.

### Level 1: Single monotone coordinate

* Ratio: (u=\log\frac{x_1}{x_2+\epsilon}), (s=u)
* Product: (u=\log(x_1x_2+\epsilon)), (s=u)
* Label uses (p=\sigma(a s))

### Level 2: Multi-term normalization / aggregation (still monotone)

* Ratio-of-sums: (u=\log\frac{x_1+x_2}{x_3+x_4+\epsilon}), (s=u)
* Product-of-sums: (u=\log((x_1+x_2)(x_3+x_4)+\epsilon)), (s=u)

### Level 3: Comparison surfaces (bilinear)

* Ratio diff: (s=\log\frac{x_1}{x_2+\epsilon}-\log\frac{x_3}{x_4+\epsilon})
* Product diff: (s=\log(x_1x_2+\epsilon)-\log(x_3x_4+\epsilon))

### Level 4: Interaction of two coordinates

* Ratio×Ratio: (u_1=\log\frac{x_1}{x_2+\epsilon},\ u_2=\log\frac{x_3}{x_4+\epsilon},\ s=u_1u_2)
* Product×Product: (u_1=\log(x_1x_2+\epsilon),\ u_2=\log(x_3x_4+\epsilon),\ s=u_1u_2)

### Level 5: Hybrid ratio/product interaction

* Ratio×Product: (u_1=\log\frac{x_1}{x_2+\epsilon},\ u_2=\log(x_3x_4+\epsilon),\ s=u_1u_2)
* (Optional) Hybrid comparison: (s=u_1-u_2)

### Level 6: Smooth nonmonotonicity (no trig)

Choose a coordinate (u) from any level above and apply:

* Peak / U-shape: (s=-((u-\mu)/\delta)^2)
* Smooth band-pass: (s=\exp(-(u-\mu)^2/(2\delta^2)))

Apply for both ratio-based (u) and product-based (u).

### Level 7: Piecewise gating (conditional dependence)

Gate variable (g=x_5) (or latent-derived). With threshold (t):

* If (g>t): use a ratio coordinate (optionally Level 6 nonmonotone)
* Else: use a product coordinate (optionally Level 6 nonmonotone)

---

## 7. Distribution panel (nuisance regimes)

Every task must be evaluated across a panel to avoid distribution-driven conclusions:

1. Tail/heaviness: lognormal with (\sigma \in {0.3, 0.7, 1.2})
2. Correlation: within-block (\rho \in {0, 0.5, 0.9})
3. Mixture extremes: 90% (\sigma=0.3), 10% (\sigma=1.2)
4. Scale shift: at test-time multiply a configured subset of raw features by (c \in {2,5,10}) (train unchanged)
5. Optional bounded positive regime: Beta-scaled features to reduce “max dominates sum” shortcuts

### Required diagnostic

For any sum (S=\sum_{i\in A} x_i), compute and report:

* dominance ratio: (\max(x_i)/S) distribution (mean, median, p90)

---

## 8. Experimental design

### 8.1 Splits

* Deterministic split into Train / Val / Test (e.g., 60/20/20) with a seed.
* Optional: keep a separate “diagnostics sample” grid for invariance testing (not used for training).

### 8.2 Repeats

* Multiple seeds per condition (configurable).

### 8.3 Sample sizes

* Support single n or a sweep (e.g., [10k, 30k, 100k]) to estimate sample complexity.

### 8.4 Model configurations

* Provide:

  * A fixed “baseline” XGB config
  * An optional “capacity sweep” grid (depth/leaves/min_child_weight) for sensitivity

Must use the same training protocol for raw vs oracle, including early stopping on Val.

---

## 9. Structure / invariance diagnostics

### 9.1 Ratio invariance

For held-out points involving a ratio coordinate ((x_1,x_2)):

* Scale invariance transform: ((x_1,x_2)\rightarrow(c x_1, c x_2))
* Score: mean absolute prediction change across random (c) values

### 9.2 Product invariance

For held-out points involving product coordinate ((x_1,x_2)):

* Compensating transform: ((x_1,x_2)\rightarrow(c x_1, x_2/c))
* Score: mean absolute prediction change

### 9.3 Iso-coordinate variance

Generate multiple points with the same ratio (or product) but different magnitudes; report variance of predicted probabilities.

Diagnostics should be computed for raw-only and oracle models.

---

## 10. Outputs

### 10.1 Machine-readable results

Write one row per run (task × regime × seed × n × model_variant):

* identifiers: task_id, level, regime_id, seed, n, model_variant (raw/oracle), xgb_config_id
* metrics: prauc, rocauc (optional), logloss (optional)
* deltas: Δ prauc (stored at aggregation stage)
* diagnostics: ratio_scale_invariance, product_comp_invariance, iso_var_ratio, iso_var_product
* nuisance stats: dominance ratios, correlation summary, shift applied, etc.

Store as Parquet (preferred) and CSV.

### 10.2 Aggregated summaries

* For each task and level: median and 10–90% interval of Δ across regimes and seeds
* For each regime family: same summaries
* Optionally: Δ vs n curves

### 10.3 Plots

* Δ distribution per task (box/violin)
* Δ vs n lines (if n sweep enabled)
* Invariance scores vs Δ scatter
* Dominance metric vs Δ scatter

### 10.4 Reports

* Auto-generated Markdown report with tables and embedded plots
* Optional HTML export

---

## 11. Project structure (implementation requirements)

### 11.1 Suggested modules

* `configs/` (YAML experiment configs)
* `src/dgp/`

  * `latent.py` (Σ construction, block correlations)
  * `marginals.py` (lognormal, mixture, beta regimes)
  * `tasks.py` (Level 1–7 task definitions, returns oracle features)
  * `label.py` (prevalence calibration, noise)
* `src/models/`

  * `xgb.py` (train/eval wrapper, early stopping, seed handling)
* `src/diagnostics/`

  * `invariance.py` (ratio/product invariance tests)
  * `dominance.py` (sum dominance stats)
* `src/runner/`

  * `grid.py` (cartesian product of tasks×regimes×seeds×n×configs)
  * `execute.py` (parallel execution, caching, logging)
* `src/reporting/`

  * `aggregate.py` (compute Δ, quantiles)
  * `plots.py` (matplotlib)
  * `report_md.py` (Markdown report builder)
* `cli.py` or `__main__.py` (CLI entrypoints)

### 11.2 Reproducibility requirements

* Single root seed -> deterministic seeds for:

  * data generation
  * train/val/test split
  * XGB training
  * invariance sampling
* Store full run config (serialized) alongside results.

### 11.3 Performance requirements

* Support parallelism across runs (joblib / multiprocessing).
* Cache datasets per (task, regime, seed, n) so raw/oracle share the same base data.

---

## 12. Configuration schema (YAML)

Must support:

* `tasks`: list of task IDs (Level 1–7 variants), component count, parameters ((\mu,\delta,a,\epsilon), gating threshold)
* `regimes`: list of nuisance regimes ((\sigma,\rho), mixture flags, shift settings, beta regime)
* `n_values`: list
* `seeds`: list or count
* `xgb_configs`: baseline + optional sweep
* `splits`: fractions
* `metrics`: prauc primary
* `output`: paths, formats

---

## 13. Acceptance criteria

1. Can run a “smoke suite” (small n, 1 seed, 2 tasks, 2 regimes) end-to-end producing:

   * results parquet/csv
   * basic plots
   * markdown report
2. Raw vs oracle Δ is computed correctly and consistent across reruns with same seed.
3. Invariance diagnostics respond correctly on known cases:

   * oracle model should show much lower invariance error than raw in at least one nontrivial task (e.g., Level 4 or Level 6).
4. Regime panel is actually applied and logged (σ, ρ, mixture, shift).
5. Dominance diagnostic is emitted for sum-based tasks and included in results.

---

## 14. Risks / common failure modes to explicitly handle

* **“Complex” ratio-of-sums looks easy** because it reduces to near-linear boundaries or sum-dominance shortcuts. Mitigate via regime panel + dominance diagnostics + shift tests.
* **Denominator near-zero instability**. Mitigate via configurable (\epsilon) and explicit monitoring of denom quantiles.
* **High variance from model randomness**. Mitigate via multiple seeds and fixed training protocol.
* **Overfitting from too-high capacity** masking structure learning. Mitigate via early stopping + capacity sweep reporting.

---

## 15. CLI workflows (must implement)

* `python -m project run --config configs/exp.yaml`
* `python -m project aggregate --input runs/... --output out/...`
* `python -m project report --input out/... --output report.md`
* `python -m project smoke` (hardcoded tiny config)

---

If you want this PRD to be maximally agent-friendly, say “produce `configs/exp_default.yaml` plus a `smoke.yaml` and implement exactly the module layout above”; otherwise the agent may choose a different layout but must meet the acceptance criteria and output requirements.

