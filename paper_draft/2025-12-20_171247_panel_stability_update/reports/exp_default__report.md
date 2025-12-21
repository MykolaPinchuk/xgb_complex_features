# XGB Complex Features Report

- Generated: 2025-12-20T17:05:15
- Input: `runs/exp_default/aggregate`

## Overview

- Run rows: 1944
- Delta rows: 1296
- Tasks (12): l1_product, l1_ratio, l2_product_of_sums, l2_ratio_of_sums, l3_product_diff, l3_ratio_diff, l4_product_x_product, l4_ratio_x_ratio, l5_ratio_x_product, l6_nonmonotone_product, l6_nonmonotone_ratio, l7_gated_ratio_vs_product
- Regimes (6): ln_sigma0.3_rho0.0, ln_sigma0.7_rho0.5, ln_sigma1.2_rho0.9, mixture_extremes_rho0.5, shift_naive_c5, shift_preserve_c5
- Oracle modes (3): oracle_coords_only, oracle_s_only, raw_only
- XGB configs (3): baseline, depth4_light, depth7_high_capacity
- n values (1): 30000
- seeds (3): 0, 1, 2
- Execution time: 9211.13s (start 2025-12-13T12:49:43.415135 → end 2025-12-13T15:23:14.546007)

## Experiment setup

This benchmark trains gradient-boosted trees (XGBoost) on synthetic ratio/product tasks to test whether a model trained on **raw variables only** can rediscover the **constructed true feature(s)** used by the data-generating process (DGP).

DGP sketch (per row):

1) Generate positive raw variables `x` (lognormal by default, with configurable correlation).
2) Compute constructed true feature(s) `u` from `x` (e.g., log-ratios, log-products, sum-normalizations).
3) Combine them into a final signal `s` (e.g., `s=u`, `s=u1-u2`, `s=u1*u2`, or a non-monotone transform).
4) Sample label `y ~ Bernoulli(sigmoid(beta0 + a*s + noise))`.

Important: **we never provide `y` to the model as a feature**. The oracle modes only add deterministic functions of `x` that the DGP itself uses.

Feature views (oracle modes):

- `raw_only`: only the generated raw columns `x_i`.
- `oracle_coords_only`: raw columns plus the constructed true feature coordinates (`u`, `u1`, `u2`, gate indicator, etc.), but **not** the final signal `s`.
- `oracle_s_only`: raw columns plus the ground-truth signal `s` (a deterministic feature used to generate `y`).

Runs cover **12 task(s)** (l1_product, l1_ratio, l2_product_of_sums, l2_ratio_of_sums, l3_product_diff, l3_ratio_diff, l4_product_x_product, l4_ratio_x_ratio, l5_ratio_x_product, l6_nonmonotone_product, l6_nonmonotone_ratio, l7_gated_ratio_vs_product), **4 regime family(ies)** (mixture_extremes, shift_naive, shift_preserve, tail_corr), and dataset size(s) **n = 30000** with seeds 0, 1, 2.

The latent prevalence is calibrated before any train/val/test split, then deterministic splits (default 60/20/20) feed the model. Regime metadata captures tail-heaviness, correlations, and scale-shift stress tests.

## What we measure

- **Primary metric**: PRAUC (Average Precision) on the test split. We report the generalization gap Δ = PRAUC_oracle − PRAUC_raw for each oracle mode.
- **Invariance diagnostics**: perturbations that should leave the true ratio/product invariant (scaling numerator/denominator together or compensating product transforms). We report the mean absolute prediction change, so smaller is better.
- **Iso-coordinate variance**: prediction variance when keeping ratio/product constant while varying magnitudes.
- **Dominance stats**: `max(x_i)/sum(x_i)` for each sum used in ratio/product-of-sums tasks to detect shortcut opportunities.
- **Variance summaries**: Δ PRAUC standard deviation and IQR across regimes/seeds/configs help flag brittle levels even when medians look healthy.

## Regimes

| regime_id | regime_family | sigma | rho | shift_type | mixture_json |
| --- | --- | --- | --- | --- | --- |
| ln_sigma0.3_rho0.0 | tail_corr | 0.3 | 0 | none | None |
| ln_sigma0.7_rho0.5 | tail_corr | 0.7 | 0.5 | none | None |
| ln_sigma1.2_rho0.9 | tail_corr | 1.2 | 0.9 | none | None |
| mixture_extremes_rho0.5 | mixture_extremes |  | 0.5 | none | {"p_low": 0.9, "sigma_high": 1.2, "sigma_low": 0.3} |
| shift_naive_c5 | shift_naive | 0.7 | 0.5 | naive | None |
| shift_preserve_c5 | shift_preserve | 0.7 | 0.5 | preserve | None |
## Summary by task

| task_id | level | oracle_mode | xgb_config_id | delta_prauc_median | delta_prauc_p10 | delta_prauc_p90 | delta_prauc_std | delta_prauc_iqr | prauc_raw_median | prauc_oracle_median | ratio_scale_invariance_raw_median | ratio_scale_invariance_oracle_median | product_comp_invariance_raw_median | product_comp_invariance_oracle_median | iso_var_ratio_raw_median | iso_var_ratio_oracle_median | iso_var_product_raw_median | iso_var_product_oracle_median | dominance_train_mean_median | dominance_train_p90_median | dominance_test_mean_median | dominance_test_p90_median | n_runs | delta_prauc_pos_rate_frac_pos | delta_prauc_pos_rate_ci_lo | delta_prauc_pos_rate_ci_hi | delta_prauc_pos_rate_n_eff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| l1_product | 1 | oracle_coords_only | baseline | 0.0009581 | -0.006308 | 0.2928 | 0.1184 | 0.008365 | 0.4424 | 0.453 |  |  | 0.01346 | 0.004001 |  |  | 0.0005206 | 6.213e-05 |  |  |  |  | 18 | 0.6111 | 0.3862 | 0.7969 | 18 |
| l1_product | 1 | oracle_coords_only | depth4_light | 0.005332 | -0.009615 | 0.3006 | 0.1195 | 0.01607 | 0.4417 | 0.4642 |  |  | 0.01311 | 0.002678 |  |  | 0.0005505 | 2.657e-05 |  |  |  |  | 18 | 0.6667 | 0.4375 | 0.8372 | 18 |
| l1_product | 1 | oracle_coords_only | depth7_high_capacity | 0.007818 | -0.01109 | 0.3108 | 0.1219 | 0.01728 | 0.4356 | 0.4567 |  |  | 0.01316 | 0.004436 |  |  | 0.0004781 | 6.423e-05 |  |  |  |  | 18 | 0.6667 | 0.4375 | 0.8372 | 18 |
| l1_product | 1 | oracle_s_only | baseline | 0.004634 | -0.01035 | 0.3035 | 0.1203 | 0.01333 | 0.4424 | 0.4546 |  |  | 0.01346 | 0.004033 |  |  | 0.0005206 | 4.963e-05 |  |  |  |  | 18 | 0.6111 | 0.3862 | 0.7969 | 18 |
| l1_product | 1 | oracle_s_only | depth4_light | 0.002574 | -0.01611 | 0.3038 | 0.1214 | 0.02555 | 0.4417 | 0.4589 |  |  | 0.01311 | 0.002656 |  |  | 0.0005505 | 1.897e-05 |  |  |  |  | 18 | 0.5556 | 0.3372 | 0.7544 | 18 |
| l1_product | 1 | oracle_s_only | depth7_high_capacity | 0.005279 | -0.01583 | 0.3044 | 0.1188 | 0.02201 | 0.4356 | 0.4467 |  |  | 0.01316 | 0.005186 |  |  | 0.0004781 | 7.046e-05 |  |  |  |  | 18 | 0.6667 | 0.4375 | 0.8372 | 18 |
| l1_ratio | 1 | oracle_coords_only | baseline | 0.01178 | -0.01816 | 0.07381 | 0.03372 | 0.02422 | 0.1482 | 0.2011 | 0.01092 | 0.001937 |  |  | 0.0001941 | 1.097e-05 |  |  |  |  |  |  | 18 | 0.8333 | 0.6078 | 0.9416 | 18 |
| l1_ratio | 1 | oracle_coords_only | depth4_light | 0.007427 | -0.005837 | 0.06276 | 0.03025 | 0.0277 | 0.1621 | 0.209 | 0.01104 | 0.001812 |  |  | 0.0002159 | 7.874e-06 |  |  |  |  |  |  | 18 | 0.7222 | 0.4913 | 0.875 | 18 |
| l1_ratio | 1 | oracle_coords_only | depth7_high_capacity | 0.01681 | -0.0103 | 0.08818 | 0.0389 | 0.03214 | 0.1362 | 0.2019 | 0.01052 | 0.002412 |  |  | 0.000212 | 1.31e-05 |  |  |  |  |  |  | 18 | 0.7222 | 0.4913 | 0.875 | 18 |
| l1_ratio | 1 | oracle_s_only | baseline | 0.01235 | -0.00718 | 0.0869 | 0.03603 | 0.02986 | 0.1482 | 0.2125 | 0.01092 | 0.002065 |  |  | 0.0001941 | 1.061e-05 |  |  |  |  |  |  | 18 | 0.7778 | 0.5479 | 0.91 | 18 |
| l1_ratio | 1 | oracle_s_only | depth4_light | 0.008537 | -0.007973 | 0.06088 | 0.02889 | 0.02884 | 0.1621 | 0.2104 | 0.01104 | 0.001428 |  |  | 0.0002159 | 4.739e-06 |  |  |  |  |  |  | 18 | 0.7222 | 0.4913 | 0.875 | 18 |
| l1_ratio | 1 | oracle_s_only | depth7_high_capacity | 0.01741 | -0.02138 | 0.1046 | 0.04514 | 0.03354 | 0.1362 | 0.2085 | 0.01052 | 0.003049 |  |  | 0.000212 | 1.947e-05 |  |  |  |  |  |  | 18 | 0.6111 | 0.3862 | 0.7969 | 18 |
| l2_product_of_sums | 2 | oracle_coords_only | baseline | 0.01086 | -0.003143 | 0.1475 | 0.07435 | 0.01873 | 0.3942 | 0.4068 |  |  | 0.01382 | 0.004697 |  |  | 0.0004218 | 4.93e-05 | 0.6163 | 0.7338 | 0.6164 | 0.7336 | 18 | 0.7778 | 0.5479 | 0.91 | 18 |
| l2_product_of_sums | 2 | oracle_coords_only | depth4_light | 0.009617 | -0.005637 | 0.1401 | 0.0761 | 0.02995 | 0.4 | 0.4147 |  |  | 0.0116 | 0.003402 |  |  | 0.0005313 | 3.026e-05 | 0.6163 | 0.7338 | 0.6164 | 0.7336 | 18 | 0.8333 | 0.6078 | 0.9416 | 18 |
| l2_product_of_sums | 2 | oracle_coords_only | depth7_high_capacity | 0.01141 | 0.0005628 | 0.1538 | 0.0739 | 0.02511 | 0.3938 | 0.4112 |  |  | 0.01385 | 0.00494 |  |  | 0.0005715 | 6.34e-05 | 0.6163 | 0.7338 | 0.6164 | 0.7336 | 18 | 0.8889 | 0.672 | 0.969 | 18 |
| l2_product_of_sums | 2 | oracle_s_only | baseline | 0.01206 | -0.002553 | 0.1518 | 0.08062 | 0.01788 | 0.3942 | 0.4169 |  |  | 0.01382 | 0.004652 |  |  | 0.0004218 | 6.944e-05 | 0.6163 | 0.7338 | 0.6164 | 0.7336 | 18 | 0.8889 | 0.672 | 0.969 | 18 |
| l2_product_of_sums | 2 | oracle_s_only | depth4_light | 0.007374 | -0.006107 | 0.1434 | 0.07837 | 0.03848 | 0.4 | 0.4064 |  |  | 0.0116 | 0.002308 |  |  | 0.0005313 | 1.605e-05 | 0.6163 | 0.7338 | 0.6164 | 0.7336 | 18 | 0.6667 | 0.4375 | 0.8372 | 18 |
| l2_product_of_sums | 2 | oracle_s_only | depth7_high_capacity | 0.0132 | -0.002305 | 0.1478 | 0.07616 | 0.02697 | 0.3938 | 0.4141 |  |  | 0.01385 | 0.005216 |  |  | 0.0005715 | 7.042e-05 | 0.6163 | 0.7338 | 0.6164 | 0.7336 | 18 | 0.8333 | 0.6078 | 0.9416 | 18 |
| l2_ratio_of_sums | 2 | oracle_coords_only | baseline | 0.01752 | -0.003811 | 0.05485 | 0.03213 | 0.03804 | 0.09078 | 0.1294 | 0.009471 | 0.001065 |  |  | 0.000137 | 5.516e-06 |  |  | 0.6158 | 0.7335 | 0.6164 | 0.7336 | 18 | 0.7778 | 0.5479 | 0.91 | 18 |
| l2_ratio_of_sums | 2 | oracle_coords_only | depth4_light | 0.01502 | -0.002822 | 0.06205 | 0.031 | 0.03219 | 0.09404 | 0.1334 | 0.009495 | 0.0009381 |  |  | 0.0001432 | 3.294e-06 |  |  | 0.6158 | 0.7335 | 0.6164 | 0.7336 | 18 | 0.7222 | 0.4913 | 0.875 | 18 |
| l2_ratio_of_sums | 2 | oracle_coords_only | depth7_high_capacity | 0.02375 | 0.002533 | 0.05432 | 0.02239 | 0.0365 | 0.08606 | 0.1208 | 0.009386 | 0.00262 |  |  | 0.0001646 | 1.49e-05 |  |  | 0.6158 | 0.7335 | 0.6164 | 0.7336 | 18 | 1 | 0.8241 | 1 | 18 |
| l2_ratio_of_sums | 2 | oracle_s_only | baseline | 0.02232 | 0.0003128 | 0.05394 | 0.02719 | 0.033 | 0.09078 | 0.1291 | 0.009471 | 0.001931 |  |  | 0.000137 | 8.804e-06 |  |  | 0.6158 | 0.7335 | 0.6164 | 0.7336 | 18 | 0.8889 | 0.672 | 0.969 | 18 |
| l2_ratio_of_sums | 2 | oracle_s_only | depth4_light | 0.01764 | -0.006193 | 0.05743 | 0.03227 | 0.03672 | 0.09404 | 0.1233 | 0.009495 | 0.0005024 |  |  | 0.0001432 | 9.63e-07 |  |  | 0.6158 | 0.7335 | 0.6164 | 0.7336 | 18 | 0.7222 | 0.4913 | 0.875 | 18 |
| l2_ratio_of_sums | 2 | oracle_s_only | depth7_high_capacity | 0.02019 | -0.006517 | 0.05559 | 0.02439 | 0.03708 | 0.08606 | 0.1186 | 0.009386 | 0.002136 |  |  | 0.0001646 | 1.235e-05 |  |  | 0.6158 | 0.7335 | 0.6164 | 0.7336 | 18 | 0.8333 | 0.6078 | 0.9416 | 18 |
| l3_product_diff | 3 | oracle_coords_only | baseline | 0.02523 | -0.007629 | 0.1503 | 0.06965 | 0.03848 | 0.1967 | 0.2993 |  |  | 0.0163 | 0.005959 |  |  | 0.0005374 | 7.999e-05 |  |  |  |  | 18 | 0.7222 | 0.4913 | 0.875 | 18 |
| l3_product_diff | 3 | oracle_coords_only | depth4_light | 0.01738 | -0.008885 | 0.1596 | 0.06807 | 0.03032 | 0.1991 | 0.3099 |  |  | 0.01605 | 0.005901 |  |  | 0.0004686 | 6.229e-05 |  |  |  |  | 18 | 0.7222 | 0.4913 | 0.875 | 18 |
| l3_product_diff | 3 | oracle_coords_only | depth7_high_capacity | 0.01356 | -0.004001 | 0.1698 | 0.06662 | 0.04301 | 0.1888 | 0.2916 |  |  | 0.01382 | 0.006937 |  |  | 0.000405 | 0.0001024 |  |  |  |  | 18 | 0.6667 | 0.4375 | 0.8372 | 18 |
| l3_product_diff | 3 | oracle_s_only | baseline | 0.03999 | 0.006356 | 0.1538 | 0.06823 | 0.04893 | 0.1967 | 0.3054 |  |  | 0.0163 | 0.002368 |  |  | 0.0005374 | 1.905e-05 |  |  |  |  | 18 | 0.9444 | 0.7424 | 0.9901 | 18 |
| l3_product_diff | 3 | oracle_s_only | depth4_light | 0.03095 | 0.005412 | 0.1681 | 0.06801 | 0.0497 | 0.1991 | 0.3144 |  |  | 0.01605 | 0.002313 |  |  | 0.0004686 | 1.772e-05 |  |  |  |  | 18 | 0.9444 | 0.7424 | 0.9901 | 18 |
| l3_product_diff | 3 | oracle_s_only | depth7_high_capacity | 0.04502 | 0.002005 | 0.1845 | 0.07705 | 0.05368 | 0.1888 | 0.3095 |  |  | 0.01382 | 0.005108 |  |  | 0.000405 | 5.327e-05 |  |  |  |  | 18 | 0.8889 | 0.672 | 0.969 | 18 |
| l3_ratio_diff | 3 | oracle_coords_only | baseline | 0.02849 | 0.003822 | 0.1495 | 0.05631 | 0.03562 | 0.2267 | 0.3213 | 0.01467 | 0.002792 |  |  | 0.0006124 | 2.545e-05 |  |  |  |  |  |  | 18 | 0.9444 | 0.7424 | 0.9901 | 18 |
| l3_ratio_diff | 3 | oracle_coords_only | depth4_light | 0.02484 | 0.005319 | 0.1492 | 0.05543 | 0.03439 | 0.2277 | 0.3288 | 0.01443 | 0.002168 |  |  | 0.0006706 | 2.146e-05 |  |  |  |  |  |  | 18 | 0.9444 | 0.7424 | 0.9901 | 18 |
| l3_ratio_diff | 3 | oracle_coords_only | depth7_high_capacity | 0.03203 | 0.00695 | 0.1492 | 0.05816 | 0.03245 | 0.2158 | 0.3145 | 0.01299 | 0.00365 |  |  | 0.0004011 | 3.969e-05 |  |  |  |  |  |  | 18 | 1 | 0.8241 | 1 | 18 |
| l3_ratio_diff | 3 | oracle_s_only | baseline | 0.04977 | 0.009403 | 0.151 | 0.05311 | 0.02752 | 0.2267 | 0.3413 | 0.01467 | 0.002937 |  |  | 0.0006124 | 1.854e-05 |  |  |  |  |  |  | 18 | 0.9444 | 0.7424 | 0.9901 | 18 |
| l3_ratio_diff | 3 | oracle_s_only | depth4_light | 0.03832 | 0.01177 | 0.1507 | 0.0546 | 0.03068 | 0.2277 | 0.3376 | 0.01443 | 0.001439 |  |  | 0.0006706 | 7.64e-06 |  |  |  |  |  |  | 18 | 1 | 0.8241 | 1 | 18 |
| l3_ratio_diff | 3 | oracle_s_only | depth7_high_capacity | 0.05142 | 0.01724 | 0.1641 | 0.06157 | 0.03897 | 0.2158 | 0.3325 | 0.01299 | 0.003893 |  |  | 0.0004011 | 3.43e-05 |  |  |  |  |  |  | 18 | 0.9444 | 0.7424 | 0.9901 | 18 |
| l4_product_x_product | 4 | oracle_coords_only | baseline | 0.01213 | -0.0004722 | 0.3355 | 0.2173 | 0.04418 | 0.8037 | 0.8993 |  |  | 0.01285 | 0.003389 |  |  | 0.001617 | 5.455e-05 |  |  |  |  | 18 | 0.8333 | 0.6078 | 0.9416 | 18 |
| l4_product_x_product | 4 | oracle_coords_only | depth4_light | 0.01505 | 1.578e-05 | 0.3885 | 0.2314 | 0.03959 | 0.8027 | 0.9033 |  |  | 0.01374 | 0.002279 |  |  | 0.001357 | 4.68e-05 |  |  |  |  | 18 | 0.8889 | 0.672 | 0.969 | 18 |
| l4_product_x_product | 4 | oracle_coords_only | depth7_high_capacity | 0.01355 | -0.0006594 | 0.3613 | 0.2088 | 0.0387 | 0.8004 | 0.8901 |  |  | 0.01332 | 0.003173 |  |  | 0.001156 | 7.566e-05 |  |  |  |  | 18 | 0.8333 | 0.6078 | 0.9416 | 18 |
| l4_product_x_product | 4 | oracle_s_only | baseline | 0.01821 | 0.00466 | 0.3479 | 0.2189 | 0.05939 | 0.8037 | 0.9028 |  |  | 0.01285 | 0.002605 |  |  | 0.001617 | 3.784e-05 |  |  |  |  | 18 | 0.9444 | 0.7424 | 0.9901 | 18 |
| l4_product_x_product | 4 | oracle_s_only | depth4_light | 0.01983 | 0.005001 | 0.395 | 0.2305 | 0.04939 | 0.8027 | 0.9065 |  |  | 0.01374 | 0.002051 |  |  | 0.001357 | 1.337e-05 |  |  |  |  | 18 | 1 | 0.8241 | 1 | 18 |
| l4_product_x_product | 4 | oracle_s_only | depth7_high_capacity | 0.0192 | 0.006235 | 0.3719 | 0.21 | 0.04958 | 0.8004 | 0.9046 |  |  | 0.01332 | 0.002544 |  |  | 0.001156 | 4.76e-05 |  |  |  |  | 18 | 1 | 0.8241 | 1 | 18 |
| l4_ratio_x_ratio | 4 | oracle_coords_only | baseline | 0.1057 | 0.009127 | 0.162 | 0.07908 | 0.1028 | 0.06764 | 0.1983 | 0.002637 | 0.001133 |  |  | 2.746e-05 | 9.404e-06 |  |  |  |  |  |  | 18 | 0.9444 | 0.7424 | 0.9901 | 18 |
| l4_ratio_x_ratio | 4 | oracle_coords_only | depth4_light | 0.1138 | 0.006618 | 0.1751 | 0.07428 | 0.08212 | 0.06225 | 0.2006 | 0.00361 | 0.001343 |  |  | 3.009e-05 | 1.368e-05 |  |  |  |  |  |  | 18 | 1 | 0.8241 | 1 | 18 |
| l4_ratio_x_ratio | 4 | oracle_coords_only | depth7_high_capacity | 0.1003 | 0.005906 | 0.1506 | 0.07332 | 0.09732 | 0.06711 | 0.1805 | 0.003621 | 0.001976 |  |  | 3.565e-05 | 1.942e-05 |  |  |  |  |  |  | 18 | 0.9444 | 0.7424 | 0.9901 | 18 |
| l4_ratio_x_ratio | 4 | oracle_s_only | baseline | 0.1125 | 0.02499 | 0.1648 | 0.07361 | 0.09871 | 0.06764 | 0.2024 | 0.002637 | 0.0005362 |  |  | 2.746e-05 | 2.861e-06 |  |  |  |  |  |  | 18 | 1 | 0.8241 | 1 | 18 |
| l4_ratio_x_ratio | 4 | oracle_s_only | depth4_light | 0.1233 | 0.02245 | 0.1723 | 0.06537 | 0.09661 | 0.06225 | 0.2123 | 0.00361 | 0.0001669 |  |  | 3.009e-05 | 8.141e-07 |  |  |  |  |  |  | 18 | 1 | 0.8241 | 1 | 18 |
| l4_ratio_x_ratio | 4 | oracle_s_only | depth7_high_capacity | 0.1187 | 0.009942 | 0.1666 | 0.06797 | 0.102 | 0.06711 | 0.201 | 0.003621 | 0.0009674 |  |  | 3.565e-05 | 6.873e-06 |  |  |  |  |  |  | 18 | 1 | 0.8241 | 1 | 18 |
| l5_ratio_x_product | 5 | oracle_coords_only | baseline | 0.04961 | 0.004222 | 0.3912 | 0.1339 | 0.02889 | 0.3761 | 0.4527 | 0.01253 | 0.004921 | 0.006157 | 0.00223 | 0.0005711 | 0.0001647 | 0.0001778 | 2.812e-05 |  |  |  |  | 18 | 0.9444 | 0.7424 | 0.9901 | 18 |
| l5_ratio_x_product | 5 | oracle_coords_only | depth4_light | 0.04593 | 0.006748 | 0.3943 | 0.1358 | 0.02472 | 0.3813 | 0.4583 | 0.0144 | 0.004262 | 0.007358 | 0.001788 | 0.0007716 | 0.0001169 | 0.0002992 | 1.401e-05 |  |  |  |  | 18 | 1 | 0.8241 | 1 | 18 |

### Relative Δ% (oracle vs raw) by task

| task_id | level | oracle_mode | xgb_config_id | prauc_oracle_median | prauc_raw_median | relative_drop_pct |
| --- | --- | --- | --- | --- | --- | --- |
| l4_ratio_x_ratio | 4 | oracle_s_only | depth4_light | 0.2123 | 0.06225 | 70.68 |
| l4_ratio_x_ratio | 4 | oracle_coords_only | depth4_light | 0.2006 | 0.06225 | 68.97 |
| l4_ratio_x_ratio | 4 | oracle_s_only | depth7_high_capacity | 0.201 | 0.06711 | 66.6 |
| l4_ratio_x_ratio | 4 | oracle_s_only | baseline | 0.2024 | 0.06764 | 66.58 |
| l4_ratio_x_ratio | 4 | oracle_coords_only | baseline | 0.1983 | 0.06764 | 65.89 |
| l4_ratio_x_ratio | 4 | oracle_coords_only | depth7_high_capacity | 0.1805 | 0.06711 | 62.82 |
| l3_product_diff | 3 | oracle_s_only | depth7_high_capacity | 0.3095 | 0.1888 | 38.99 |
| l3_product_diff | 3 | oracle_s_only | depth4_light | 0.3144 | 0.1991 | 36.66 |
| l3_product_diff | 3 | oracle_coords_only | depth4_light | 0.3099 | 0.1991 | 35.74 |
| l3_product_diff | 3 | oracle_s_only | baseline | 0.3054 | 0.1967 | 35.6 |
| l3_product_diff | 3 | oracle_coords_only | depth7_high_capacity | 0.2916 | 0.1888 | 35.25 |
| l3_ratio_diff | 3 | oracle_s_only | depth7_high_capacity | 0.3325 | 0.2158 | 35.08 |
| l1_ratio | 1 | oracle_s_only | depth7_high_capacity | 0.2085 | 0.1362 | 34.65 |
| l3_product_diff | 3 | oracle_coords_only | baseline | 0.2993 | 0.1967 | 34.29 |
| l3_ratio_diff | 3 | oracle_s_only | baseline | 0.3413 | 0.2267 | 33.59 |
| l3_ratio_diff | 3 | oracle_s_only | depth4_light | 0.3376 | 0.2277 | 32.55 |
| l1_ratio | 1 | oracle_coords_only | depth7_high_capacity | 0.2019 | 0.1362 | 32.52 |
| l3_ratio_diff | 3 | oracle_coords_only | depth7_high_capacity | 0.3145 | 0.2158 | 31.36 |
| l3_ratio_diff | 3 | oracle_coords_only | depth4_light | 0.3288 | 0.2277 | 30.75 |
| l1_ratio | 1 | oracle_s_only | baseline | 0.2125 | 0.1482 | 30.27 |
| l2_ratio_of_sums | 2 | oracle_coords_only | baseline | 0.1294 | 0.09078 | 29.82 |
| l2_ratio_of_sums | 2 | oracle_s_only | baseline | 0.1291 | 0.09078 | 29.66 |
| l2_ratio_of_sums | 2 | oracle_coords_only | depth4_light | 0.1334 | 0.09404 | 29.52 |
| l3_ratio_diff | 3 | oracle_coords_only | baseline | 0.3213 | 0.2267 | 29.44 |
| l2_ratio_of_sums | 2 | oracle_coords_only | depth7_high_capacity | 0.1208 | 0.08606 | 28.76 |
| l2_ratio_of_sums | 2 | oracle_s_only | depth7_high_capacity | 0.1186 | 0.08606 | 27.47 |
| l1_ratio | 1 | oracle_coords_only | baseline | 0.2011 | 0.1482 | 26.31 |
| l6_nonmonotone_ratio | 6 | oracle_s_only | depth4_light | 0.06778 | 0.051 | 24.76 |
| l2_ratio_of_sums | 2 | oracle_s_only | depth4_light | 0.1233 | 0.09404 | 23.74 |
| l1_ratio | 1 | oracle_s_only | depth4_light | 0.2104 | 0.1621 | 22.96 |
| l6_nonmonotone_ratio | 6 | oracle_s_only | depth7_high_capacity | 0.06633 | 0.05133 | 22.61 |
| l1_ratio | 1 | oracle_coords_only | depth4_light | 0.209 | 0.1621 | 22.44 |
| l5_ratio_x_product | 5 | oracle_s_only | depth4_light | 0.486 | 0.3813 | 21.55 |
| l5_ratio_x_product | 5 | oracle_s_only | baseline | 0.4768 | 0.3761 | 21.11 |
| l5_ratio_x_product | 5 | oracle_s_only | depth7_high_capacity | 0.4731 | 0.3781 | 20.09 |
| l6_nonmonotone_ratio | 6 | oracle_coords_only | depth4_light | 0.06355 | 0.051 | 19.74 |
| l6_nonmonotone_ratio | 6 | oracle_s_only | baseline | 0.06443 | 0.05212 | 19.1 |
| l5_ratio_x_product | 5 | oracle_coords_only | baseline | 0.4527 | 0.3761 | 16.91 |
| l5_ratio_x_product | 5 | oracle_coords_only | depth4_light | 0.4583 | 0.3813 | 16.8 |
| l5_ratio_x_product | 5 | oracle_coords_only | depth7_high_capacity | 0.4439 | 0.3781 | 14.83 |
| l6_nonmonotone_product | 6 | oracle_s_only | depth7_high_capacity | 0.06624 | 0.05655 | 14.62 |
| l6_nonmonotone_product | 6 | oracle_s_only | depth4_light | 0.06894 | 0.05915 | 14.21 |
| l6_nonmonotone_ratio | 6 | oracle_coords_only | depth7_high_capacity | 0.05972 | 0.05133 | 14.05 |
| l6_nonmonotone_ratio | 6 | oracle_coords_only | baseline | 0.05959 | 0.05212 | 12.54 |
| l6_nonmonotone_product | 6 | oracle_coords_only | depth7_high_capacity | 0.06462 | 0.05655 | 12.48 |
| l4_product_x_product | 4 | oracle_s_only | depth7_high_capacity | 0.9046 | 0.8004 | 11.51 |
| l4_product_x_product | 4 | oracle_s_only | depth4_light | 0.9065 | 0.8027 | 11.46 |
| l4_product_x_product | 4 | oracle_coords_only | depth4_light | 0.9033 | 0.8027 | 11.14 |
| l4_product_x_product | 4 | oracle_s_only | baseline | 0.9028 | 0.8037 | 10.98 |
| l6_nonmonotone_product | 6 | oracle_coords_only | depth4_light | 0.0664 | 0.05915 | 10.92 |

## Summary by level

| level | oracle_mode | xgb_config_id | delta_prauc_median | delta_prauc_p10 | delta_prauc_p90 | delta_prauc_std | delta_prauc_iqr | prauc_raw_median | prauc_oracle_median | ratio_scale_invariance_raw_median | ratio_scale_invariance_oracle_median | product_comp_invariance_raw_median | product_comp_invariance_oracle_median | iso_var_ratio_raw_median | iso_var_ratio_oracle_median | iso_var_product_raw_median | iso_var_product_oracle_median | dominance_train_mean_median | dominance_train_p90_median | dominance_test_mean_median | dominance_test_p90_median | n_runs | delta_prauc_pos_rate_frac_pos | delta_prauc_pos_rate_ci_lo | delta_prauc_pos_rate_ci_hi | delta_prauc_pos_rate_n_eff | delta_prauc_median_loo_task_min | delta_prauc_median_loo_task_max | delta_prauc_median_loo_task_n_loo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | oracle_coords_only | baseline | 0.006043 | -0.01161 | 0.09116 | 0.08838 | 0.02515 | 0.2046 | 0.244 | 0.01092 | 0.001937 | 0.01346 | 0.004001 | 0.0001941 | 1.097e-05 | 0.0005206 | 6.213e-05 |  |  |  |  | 36 | 0.7222 | 0.5601 | 0.8415 | 36 | 0.0009581 | 0.01178 | 2 |
| 1 | oracle_coords_only | depth4_light | 0.007029 | -0.008794 | 0.08009 | 0.08936 | 0.02861 | 0.2074 | 0.2369 | 0.01104 | 0.001812 | 0.01311 | 0.002678 | 0.0002159 | 7.874e-06 | 0.0005505 | 2.657e-05 |  |  |  |  | 36 | 0.6944 | 0.5314 | 0.82 | 36 | 0.005332 | 0.007427 | 2 |
| 1 | oracle_coords_only | depth7_high_capacity | 0.0091 | -0.01173 | 0.1073 | 0.09151 | 0.03363 | 0.1907 | 0.2293 | 0.01052 | 0.002412 | 0.01316 | 0.004436 | 0.000212 | 1.31e-05 | 0.0004781 | 6.423e-05 |  |  |  |  | 36 | 0.6944 | 0.5314 | 0.82 | 36 | 0.007818 | 0.01681 | 2 |
| 1 | oracle_s_only | baseline | 0.007966 | -0.009763 | 0.09888 | 0.08976 | 0.02919 | 0.2046 | 0.2387 | 0.01092 | 0.002065 | 0.01346 | 0.004033 | 0.0001941 | 1.061e-05 | 0.0005206 | 4.963e-05 |  |  |  |  | 36 | 0.6944 | 0.5314 | 0.82 | 36 | 0.004634 | 0.01235 | 2 |
| 1 | oracle_s_only | depth4_light | 0.006183 | -0.01301 | 0.08162 | 0.08992 | 0.02896 | 0.2074 | 0.2443 | 0.01104 | 0.001428 | 0.01311 | 0.002656 | 0.0002159 | 4.739e-06 | 0.0005505 | 1.897e-05 |  |  |  |  | 36 | 0.6389 | 0.4758 | 0.7752 | 36 | 0.002574 | 0.008537 | 2 |
| 1 | oracle_s_only | depth7_high_capacity | 0.009707 | -0.0191 | 0.1163 | 0.09065 | 0.03061 | 0.1907 | 0.226 | 0.01052 | 0.003049 | 0.01316 | 0.005186 | 0.000212 | 1.947e-05 | 0.0004781 | 7.046e-05 |  |  |  |  | 36 | 0.6389 | 0.4758 | 0.7752 | 36 | 0.005279 | 0.01741 | 2 |
| 2 | oracle_coords_only | baseline | 0.01133 | -0.003332 | 0.07183 | 0.05819 | 0.03392 | 0.1317 | 0.1614 | 0.009471 | 0.001065 | 0.01382 | 0.004697 | 0.000137 | 5.516e-06 | 0.0004218 | 4.93e-05 | 0.6162 | 0.7338 | 0.6164 | 0.7336 | 36 | 0.7778 | 0.6192 | 0.8828 | 36 | 0.01086 | 0.01752 | 2 |
| 2 | oracle_coords_only | depth4_light | 0.01228 | -0.005926 | 0.07028 | 0.05919 | 0.03045 | 0.1352 | 0.1517 | 0.009495 | 0.0009381 | 0.0116 | 0.003402 | 0.0001432 | 3.294e-06 | 0.0005313 | 3.026e-05 | 0.6162 | 0.7338 | 0.6164 | 0.7336 | 36 | 0.7778 | 0.6192 | 0.8828 | 36 | 0.009617 | 0.01502 | 2 |
| 2 | oracle_coords_only | depth7_high_capacity | 0.01295 | 0.001539 | 0.06976 | 0.05517 | 0.03578 | 0.1221 | 0.1531 | 0.009386 | 0.00262 | 0.01385 | 0.00494 | 0.0001646 | 1.49e-05 | 0.0005715 | 6.34e-05 | 0.6162 | 0.7338 | 0.6164 | 0.7336 | 36 | 0.9444 | 0.8186 | 0.9846 | 36 | 0.01141 | 0.02375 | 2 |
| 2 | oracle_s_only | baseline | 0.01414 | -7.911e-05 | 0.06689 | 0.06123 | 0.0315 | 0.1317 | 0.1569 | 0.009471 | 0.001931 | 0.01382 | 0.004652 | 0.000137 | 8.804e-06 | 0.0004218 | 6.944e-05 | 0.6162 | 0.7338 | 0.6164 | 0.7336 | 36 | 0.8889 | 0.7469 | 0.9559 | 36 | 0.01206 | 0.02232 | 2 |
| 2 | oracle_s_only | depth4_light | 0.01222 | -0.007071 | 0.06689 | 0.06095 | 0.03711 | 0.1352 | 0.1623 | 0.009495 | 0.0005024 | 0.0116 | 0.002308 | 0.0001432 | 9.63e-07 | 0.0005313 | 1.605e-05 | 0.6162 | 0.7338 | 0.6164 | 0.7336 | 36 | 0.6944 | 0.5314 | 0.82 | 36 | 0.007374 | 0.01764 | 2 |
| 2 | oracle_s_only | depth7_high_capacity | 0.01591 | -0.004081 | 0.07034 | 0.05761 | 0.03686 | 0.1221 | 0.1511 | 0.009386 | 0.002136 | 0.01385 | 0.005216 | 0.0001646 | 1.235e-05 | 0.0005715 | 7.042e-05 | 0.6162 | 0.7338 | 0.6164 | 0.7336 | 36 | 0.8333 | 0.6811 | 0.9213 | 36 | 0.0132 | 0.02019 | 2 |
| 3 | oracle_coords_only | baseline | 0.02801 | -0.00652 | 0.1527 | 0.06333 | 0.03222 | 0.2106 | 0.3186 | 0.01467 | 0.002792 | 0.0163 | 0.005959 | 0.0006124 | 2.545e-05 | 0.0005374 | 7.999e-05 |  |  |  |  | 36 | 0.8333 | 0.6811 | 0.9213 | 36 | 0.02523 | 0.02849 | 2 |
| 3 | oracle_coords_only | depth4_light | 0.02138 | -0.002333 | 0.1497 | 0.06216 | 0.0297 | 0.2181 | 0.3247 | 0.01443 | 0.002168 | 0.01605 | 0.005901 | 0.0006706 | 2.146e-05 | 0.0004686 | 6.229e-05 |  |  |  |  | 36 | 0.8333 | 0.6811 | 0.9213 | 36 | 0.01738 | 0.02484 | 2 |
| 3 | oracle_coords_only | depth7_high_capacity | 0.02722 | -0.001175 | 0.1551 | 0.06289 | 0.04295 | 0.1979 | 0.3035 | 0.01299 | 0.00365 | 0.01382 | 0.006937 | 0.0004011 | 3.969e-05 | 0.000405 | 0.0001024 |  |  |  |  | 36 | 0.8333 | 0.6811 | 0.9213 | 36 | 0.01356 | 0.03203 | 2 |
| 3 | oracle_s_only | baseline | 0.04674 | 0.006236 | 0.1533 | 0.06114 | 0.04114 | 0.2106 | 0.3298 | 0.01467 | 0.002937 | 0.0163 | 0.002368 | 0.0006124 | 1.854e-05 | 0.0005374 | 1.905e-05 |  |  |  |  | 36 | 0.9444 | 0.8186 | 0.9846 | 36 | 0.03999 | 0.04977 | 2 |
| 3 | oracle_s_only | depth4_light | 0.03577 | 0.006117 | 0.1603 | 0.06167 | 0.04423 | 0.2181 | 0.3342 | 0.01443 | 0.001439 | 0.01605 | 0.002313 | 0.0006706 | 7.64e-06 | 0.0004686 | 1.772e-05 |  |  |  |  | 36 | 0.9722 | 0.8583 | 0.9951 | 36 | 0.03095 | 0.03832 | 2 |
| 3 | oracle_s_only | depth7_high_capacity | 0.04897 | 0.006683 | 0.1785 | 0.06979 | 0.04806 | 0.1979 | 0.3222 | 0.01299 | 0.003893 | 0.01382 | 0.005108 | 0.0004011 | 3.43e-05 | 0.000405 | 5.327e-05 |  |  |  |  | 36 | 0.9167 | 0.7817 | 0.9713 | 36 | 0.04502 | 0.05142 | 2 |
| 4 | oracle_coords_only | baseline | 0.03538 | 0.0001933 | 0.1897 | 0.1635 | 0.1167 | 0.1157 | 0.2356 | 0.002637 | 0.001133 | 0.01285 | 0.003389 | 2.746e-05 | 9.404e-06 | 0.001617 | 5.455e-05 |  |  |  |  | 36 | 0.8889 | 0.7469 | 0.9559 | 36 | 0.01213 | 0.1057 | 2 |
| 4 | oracle_coords_only | depth4_light | 0.04546 | 0.002847 | 0.2306 | 0.172 | 0.114 | 0.1129 | 0.2372 | 0.00361 | 0.001343 | 0.01374 | 0.002279 | 3.009e-05 | 1.368e-05 | 0.001357 | 4.68e-05 |  |  |  |  | 36 | 0.9444 | 0.8186 | 0.9846 | 36 | 0.01505 | 0.1138 | 2 |
| 4 | oracle_coords_only | depth7_high_capacity | 0.03235 | 0.0002426 | 0.2091 | 0.1566 | 0.111 | 0.1138 | 0.2341 | 0.003621 | 0.001976 | 0.01332 | 0.003173 | 3.565e-05 | 1.942e-05 | 0.001156 | 7.566e-05 |  |  |  |  | 36 | 0.8889 | 0.7469 | 0.9559 | 36 | 0.01355 | 0.1003 | 2 |
| 4 | oracle_s_only | baseline | 0.05155 | 0.008654 | 0.1958 | 0.1633 | 0.1163 | 0.1157 | 0.2457 | 0.002637 | 0.0005362 | 0.01285 | 0.002605 | 2.746e-05 | 2.861e-06 | 0.001617 | 3.784e-05 |  |  |  |  | 36 | 0.9722 | 0.8583 | 0.9951 | 36 | 0.01821 | 0.1125 | 2 |
| 4 | oracle_s_only | depth4_light | 0.05038 | 0.01166 | 0.2194 | 0.1695 | 0.1212 | 0.1129 | 0.249 | 0.00361 | 0.0001669 | 0.01374 | 0.002051 | 3.009e-05 | 8.141e-07 | 0.001357 | 1.337e-05 |  |  |  |  | 36 | 1 | 0.9036 | 1 | 36 | 0.01983 | 0.1233 | 2 |
| 4 | oracle_s_only | depth7_high_capacity | 0.04046 | 0.009691 | 0.2133 | 0.1562 | 0.12 | 0.1138 | 0.2453 | 0.003621 | 0.0009674 | 0.01332 | 0.002544 | 3.565e-05 | 6.873e-06 | 0.001156 | 4.76e-05 |  |  |  |  | 36 | 1 | 0.9036 | 1 | 36 | 0.0192 | 0.1187 | 2 |
| 5 | oracle_coords_only | baseline | 0.04961 | 0.004222 | 0.3912 | 0.1339 | 0.02889 | 0.3761 | 0.4527 | 0.01253 | 0.004921 | 0.006157 | 0.00223 | 0.0005711 | 0.0001647 | 0.0001778 | 2.812e-05 |  |  |  |  | 18 | 0.9444 | 0.7424 | 0.9901 | 18 |  |  | 1 |
| 5 | oracle_coords_only | depth4_light | 0.04593 | 0.006748 | 0.3943 | 0.1358 | 0.02472 | 0.3813 | 0.4583 | 0.0144 | 0.004262 | 0.007358 | 0.001788 | 0.0007716 | 0.0001169 | 0.0002992 | 1.401e-05 |  |  |  |  | 18 | 1 | 0.8241 | 1 | 18 |  |  | 1 |
| 5 | oracle_coords_only | depth7_high_capacity | 0.0443 | 0.003988 | 0.3573 | 0.1249 | 0.028 | 0.3781 | 0.4439 | 0.01137 | 0.005475 | 0.005633 | 0.002911 | 0.0006228 | 0.0001228 | 0.0001379 | 3.982e-05 |  |  |  |  | 18 | 0.9444 | 0.7424 | 0.9901 | 18 |  |  | 1 |
| 5 | oracle_s_only | baseline | 0.06212 | 0.01778 | 0.4154 | 0.1398 | 0.03853 | 0.3761 | 0.4768 | 0.01253 | 0.002334 | 0.006157 | 0.0007193 | 0.0005711 | 2.592e-05 | 0.0001778 | 4.719e-06 |  |  |  |  | 18 | 1 | 0.8241 | 1 | 18 |  |  | 1 |
| 5 | oracle_s_only | depth4_light | 0.0676 | 0.0195 | 0.4396 | 0.1462 | 0.04283 | 0.3813 | 0.486 | 0.0144 | 0.001665 | 0.007358 | 0.0006113 | 0.0007716 | 1.38e-05 | 0.0002992 | 3.278e-06 |  |  |  |  | 18 | 1 | 0.8241 | 1 | 18 |  |  | 1 |
| 5 | oracle_s_only | depth7_high_capacity | 0.06275 | 0.01755 | 0.4198 | 0.1396 | 0.01664 | 0.3781 | 0.4731 | 0.01137 | 0.002989 | 0.005633 | 0.001593 | 0.0006228 | 3.051e-05 | 0.0001379 | 1.141e-05 |  |  |  |  | 18 | 1 | 0.8241 | 1 | 18 |  |  | 1 |
| 6 | oracle_coords_only | baseline | 0.006231 | -0.0009186 | 0.02039 | 0.009971 | 0.01389 | 0.05393 | 0.06075 | 0.0002098 | 8.181e-05 | 0.003098 | 0.0003725 | 3.929e-07 | 3.022e-07 | 1.35e-05 | 2.425e-06 |  |  |  |  | 36 | 0.8889 | 0.7469 | 0.9559 | 36 | 0.003454 | 0.00724 | 2 |
| 6 | oracle_coords_only | depth4_light | 0.00695 | -0.000623 | 0.02177 | 0.01064 | 0.01635 | 0.05373 | 0.06603 | 4.283e-05 | 2.512e-05 | 0.004956 | 0.0006579 | 1.488e-07 | 9.845e-08 | 4.335e-05 | 1.528e-06 |  |  |  |  | 36 | 0.8333 | 0.6811 | 0.9213 | 36 | 0.002615 | 0.01426 | 2 |
| 6 | oracle_coords_only | depth7_high_capacity | 0.007863 | 4.636e-05 | 0.01659 | 0.008253 | 0.01054 | 0.05329 | 0.06274 | 4.439e-05 | 0.0001713 | 0.001612 | 0.001149 | 1.858e-07 | 1.9e-07 | 7.82e-06 | 4.662e-06 |  |  |  |  | 36 | 0.9167 | 0.7817 | 0.9713 | 36 | 0.003227 | 0.01012 | 2 |
| 6 | oracle_s_only | baseline | 0.007924 | 7.535e-05 | 0.02224 | 0.008742 | 0.01308 | 0.05393 | 0.06527 | 0.0002098 | 1.029e-05 | 0.003098 | 0.0005848 | 3.929e-07 | 1.06e-07 | 1.35e-05 | 3.26e-06 |  |  |  |  | 36 | 0.8889 | 0.7469 | 0.9559 | 36 | 0.004596 | 0.01184 | 2 |
| 6 | oracle_s_only | depth4_light | 0.007982 | -0.0006412 | 0.02236 | 0.01212 | 0.01801 | 0.05373 | 0.06855 | 4.283e-05 | 1.383e-05 | 0.004956 | 0.0004741 | 1.488e-07 | 1.167e-08 | 4.335e-05 | 7.487e-07 |  |  |  |  | 36 | 0.8333 | 0.6811 | 0.9213 | 36 | 0.001817 | 0.01689 | 2 |
| 6 | oracle_s_only | depth7_high_capacity | 0.009917 | -0.0003027 | 0.01905 | 0.008667 | 0.01129 | 0.05329 | 0.06624 | 4.439e-05 | 8.092e-05 | 0.001612 | 0.0005287 | 1.858e-07 | 3.97e-07 | 7.82e-06 | 2.313e-06 |  |  |  |  | 36 | 0.8611 | 0.7134 | 0.9392 | 36 | 0.005462 | 0.01499 | 2 |
| 7 | oracle_coords_only | baseline | 0.007698 | -0.005826 | 0.02425 | 0.0124 | 0.01778 | 0.3334 | 0.3323 | 0.0287 | 0.03318 | 0.01873 | 0.01483 | 0.003211 | 0.003341 | 0.00111 | 0.0006827 |  |  |  |  | 18 | 0.7778 | 0.5479 | 0.91 | 18 |  |  | 1 |
| 7 | oracle_coords_only | depth4_light | 0.009797 | -0.00553 | 0.02879 | 0.0131 | 0.01772 | 0.3438 | 0.3499 | 0.03217 | 0.03248 | 0.02095 | 0.01352 | 0.003552 | 0.003345 | 0.001099 | 0.0006858 |  |  |  |  | 18 | 0.7778 | 0.5479 | 0.91 | 18 |  |  | 1 |
| 7 | oracle_coords_only | depth7_high_capacity | 0.0107 | -0.001591 | 0.02016 | 0.009193 | 0.007573 | 0.3241 | 0.3298 | 0.02898 | 0.0329 | 0.01873 | 0.01615 | 0.003123 | 0.003979 | 0.0008717 | 0.0006795 |  |  |  |  | 18 | 0.8333 | 0.6078 | 0.9416 | 18 |  |  | 1 |
| 7 | oracle_s_only | baseline | 0.01638 | -0.000935 | 0.04108 | 0.01905 | 0.02563 | 0.3334 | 0.3462 | 0.0287 | 0.03125 | 0.01873 | 0.016 | 0.003211 | 0.003003 | 0.00111 | 0.001118 |  |  |  |  | 18 | 0.8889 | 0.672 | 0.969 | 18 |  |  | 1 |
| 7 | oracle_s_only | depth4_light | 0.01807 | -0.01227 | 0.04464 | 0.02459 | 0.03489 | 0.3438 | 0.3427 | 0.03217 | 0.03325 | 0.02095 | 0.0174 | 0.003552 | 0.003624 | 0.001099 | 0.001498 |  |  |  |  | 18 | 0.7222 | 0.4913 | 0.875 | 18 |  |  | 1 |
| 7 | oracle_s_only | depth7_high_capacity | 0.02215 | 0.0003775 | 0.03813 | 0.01577 | 0.02252 | 0.3241 | 0.3381 | 0.02898 | 0.02683 | 0.01873 | 0.01559 | 0.003123 | 0.002423 | 0.0008717 | 0.0009049 |  |  |  |  | 18 | 0.8889 | 0.672 | 0.969 | 18 |  |  | 1 |
### Relative Δ% (oracle vs raw) by level

| level | oracle_mode | xgb_config_id | prauc_oracle_median | prauc_raw_median | relative_drop_pct |
| --- | --- | --- | --- | --- | --- |
| 4 | oracle_s_only | depth4_light | 0.249 | 0.1129 | 54.64 |
| 4 | oracle_s_only | depth7_high_capacity | 0.2453 | 0.1138 | 53.59 |
| 4 | oracle_s_only | baseline | 0.2457 | 0.1157 | 52.92 |
| 4 | oracle_coords_only | depth4_light | 0.2372 | 0.1129 | 52.38 |
| 4 | oracle_coords_only | depth7_high_capacity | 0.2341 | 0.1138 | 51.37 |
| 4 | oracle_coords_only | baseline | 0.2356 | 0.1157 | 50.9 |
| 3 | oracle_s_only | depth7_high_capacity | 0.3222 | 0.1979 | 38.59 |
| 3 | oracle_s_only | baseline | 0.3298 | 0.2106 | 36.16 |
| 3 | oracle_coords_only | depth7_high_capacity | 0.3035 | 0.1979 | 34.8 |
| 3 | oracle_s_only | depth4_light | 0.3342 | 0.2181 | 34.74 |
| 3 | oracle_coords_only | baseline | 0.3186 | 0.2106 | 33.92 |
| 3 | oracle_coords_only | depth4_light | 0.3247 | 0.2181 | 32.84 |
| 6 | oracle_s_only | depth4_light | 0.06855 | 0.05373 | 21.62 |
| 5 | oracle_s_only | depth4_light | 0.486 | 0.3813 | 21.55 |
| 5 | oracle_s_only | baseline | 0.4768 | 0.3761 | 21.11 |
| 2 | oracle_coords_only | depth7_high_capacity | 0.1531 | 0.1221 | 20.22 |
| 5 | oracle_s_only | depth7_high_capacity | 0.4731 | 0.3781 | 20.09 |
| 6 | oracle_s_only | depth7_high_capacity | 0.06624 | 0.05329 | 19.55 |
| 2 | oracle_s_only | depth7_high_capacity | 0.1511 | 0.1221 | 19.16 |
| 6 | oracle_coords_only | depth4_light | 0.06603 | 0.05373 | 18.62 |
| 2 | oracle_coords_only | baseline | 0.1614 | 0.1317 | 18.41 |
| 6 | oracle_s_only | baseline | 0.06527 | 0.05393 | 17.37 |
| 5 | oracle_coords_only | baseline | 0.4527 | 0.3761 | 16.91 |
| 1 | oracle_coords_only | depth7_high_capacity | 0.2293 | 0.1907 | 16.85 |
| 5 | oracle_coords_only | depth4_light | 0.4583 | 0.3813 | 16.8 |
| 2 | oracle_s_only | depth4_light | 0.1623 | 0.1352 | 16.67 |
| 1 | oracle_coords_only | baseline | 0.244 | 0.2046 | 16.15 |
| 2 | oracle_s_only | baseline | 0.1569 | 0.1317 | 16.07 |
| 1 | oracle_s_only | depth7_high_capacity | 0.226 | 0.1907 | 15.62 |
| 1 | oracle_s_only | depth4_light | 0.2443 | 0.2074 | 15.1 |
| 6 | oracle_coords_only | depth7_high_capacity | 0.06274 | 0.05329 | 15.06 |
| 5 | oracle_coords_only | depth7_high_capacity | 0.4439 | 0.3781 | 14.83 |
| 1 | oracle_s_only | baseline | 0.2387 | 0.2046 | 14.3 |
| 1 | oracle_coords_only | depth4_light | 0.2369 | 0.2074 | 12.43 |
| 6 | oracle_coords_only | baseline | 0.06075 | 0.05393 | 11.23 |
| 2 | oracle_coords_only | depth4_light | 0.1517 | 0.1352 | 10.87 |
| 7 | oracle_s_only | depth7_high_capacity | 0.3381 | 0.3241 | 4.158 |
| 7 | oracle_s_only | baseline | 0.3462 | 0.3334 | 3.691 |
| 7 | oracle_coords_only | depth4_light | 0.3499 | 0.3438 | 1.768 |
| 7 | oracle_coords_only | depth7_high_capacity | 0.3298 | 0.3241 | 1.748 |
| 7 | oracle_s_only | depth4_light | 0.3427 | 0.3438 | -0.3118 |
| 7 | oracle_coords_only | baseline | 0.3323 | 0.3334 | -0.3357 |
## Highest Δ levels (ranked)

| level | oracle_mode | xgb_config_id | delta_prauc_median | delta_prauc_p90 | delta_prauc_p10 | delta_prauc_std | delta_prauc_iqr | prauc_raw_median | prauc_oracle_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | oracle_s_only | depth4_light | 0.0676 | 0.4396 | 0.0195 | 0.1462 | 0.04283 | 0.3813 | 0.486 |
| 5 | oracle_s_only | depth7_high_capacity | 0.06275 | 0.4198 | 0.01755 | 0.1396 | 0.01664 | 0.3781 | 0.4731 |
| 5 | oracle_s_only | baseline | 0.06212 | 0.4154 | 0.01778 | 0.1398 | 0.03853 | 0.3761 | 0.4768 |
| 4 | oracle_s_only | baseline | 0.05155 | 0.1958 | 0.008654 | 0.1633 | 0.1163 | 0.1157 | 0.2457 |
| 4 | oracle_s_only | depth4_light | 0.05038 | 0.2194 | 0.01166 | 0.1695 | 0.1212 | 0.1129 | 0.249 |
| 5 | oracle_coords_only | baseline | 0.04961 | 0.3912 | 0.004222 | 0.1339 | 0.02889 | 0.3761 | 0.4527 |
| 3 | oracle_s_only | depth7_high_capacity | 0.04897 | 0.1785 | 0.006683 | 0.06979 | 0.04806 | 0.1979 | 0.3222 |
| 3 | oracle_s_only | baseline | 0.04674 | 0.1533 | 0.006236 | 0.06114 | 0.04114 | 0.2106 | 0.3298 |
| 5 | oracle_coords_only | depth4_light | 0.04593 | 0.3943 | 0.006748 | 0.1358 | 0.02472 | 0.3813 | 0.4583 |
| 4 | oracle_coords_only | depth4_light | 0.04546 | 0.2306 | 0.002847 | 0.172 | 0.114 | 0.1129 | 0.2372 |
| 5 | oracle_coords_only | depth7_high_capacity | 0.0443 | 0.3573 | 0.003988 | 0.1249 | 0.028 | 0.3781 | 0.4439 |
| 4 | oracle_s_only | depth7_high_capacity | 0.04046 | 0.2133 | 0.009691 | 0.1562 | 0.12 | 0.1138 | 0.2453 |
| 3 | oracle_s_only | depth4_light | 0.03577 | 0.1603 | 0.006117 | 0.06167 | 0.04423 | 0.2181 | 0.3342 |
| 4 | oracle_coords_only | baseline | 0.03538 | 0.1897 | 0.0001933 | 0.1635 | 0.1167 | 0.1157 | 0.2356 |
| 4 | oracle_coords_only | depth7_high_capacity | 0.03235 | 0.2091 | 0.0002426 | 0.1566 | 0.111 | 0.1138 | 0.2341 |
| 3 | oracle_coords_only | baseline | 0.02801 | 0.1527 | -0.00652 | 0.06333 | 0.03222 | 0.2106 | 0.3186 |
| 3 | oracle_coords_only | depth7_high_capacity | 0.02722 | 0.1551 | -0.001175 | 0.06289 | 0.04295 | 0.1979 | 0.3035 |
| 7 | oracle_s_only | depth7_high_capacity | 0.02215 | 0.03813 | 0.0003775 | 0.01577 | 0.02252 | 0.3241 | 0.3381 |
| 3 | oracle_coords_only | depth4_light | 0.02138 | 0.1497 | -0.002333 | 0.06216 | 0.0297 | 0.2181 | 0.3247 |
| 7 | oracle_s_only | depth4_light | 0.01807 | 0.04464 | -0.01227 | 0.02459 | 0.03489 | 0.3438 | 0.3427 |
| 7 | oracle_s_only | baseline | 0.01638 | 0.04108 | -0.000935 | 0.01905 | 0.02563 | 0.3334 | 0.3462 |
| 2 | oracle_s_only | depth7_high_capacity | 0.01591 | 0.07034 | -0.004081 | 0.05761 | 0.03686 | 0.1221 | 0.1511 |
| 2 | oracle_s_only | baseline | 0.01414 | 0.06689 | -7.911e-05 | 0.06123 | 0.0315 | 0.1317 | 0.1569 |
| 2 | oracle_coords_only | depth7_high_capacity | 0.01295 | 0.06976 | 0.001539 | 0.05517 | 0.03578 | 0.1221 | 0.1531 |
| 2 | oracle_coords_only | depth4_light | 0.01228 | 0.07028 | -0.005926 | 0.05919 | 0.03045 | 0.1352 | 0.1517 |

## Summary by regime family

| regime_family | oracle_mode | xgb_config_id | delta_prauc_median | delta_prauc_p10 | delta_prauc_p90 | delta_prauc_std | delta_prauc_iqr | prauc_raw_median | prauc_oracle_median | ratio_scale_invariance_raw_median | ratio_scale_invariance_oracle_median | product_comp_invariance_raw_median | product_comp_invariance_oracle_median | iso_var_ratio_raw_median | iso_var_ratio_oracle_median | iso_var_product_raw_median | iso_var_product_oracle_median | dominance_train_mean_median | dominance_train_p90_median | dominance_test_mean_median | dominance_test_p90_median | n_runs | delta_prauc_pos_rate_frac_pos | delta_prauc_pos_rate_ci_lo | delta_prauc_pos_rate_ci_hi | delta_prauc_pos_rate_n_eff | delta_prauc_median_loo_task_min | delta_prauc_median_loo_task_max | delta_prauc_median_loo_task_n_loo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mixture_extremes | oracle_coords_only | baseline | 0.01599 | 0.002924 | 0.04606 | 0.02934 | 0.02943 | 0.2779 | 0.3025 | 0.006994 | 0.001755 | 0.01177 | 0.003651 | 0.0001327 | 1.077e-05 | 0.000409 | 5.242e-05 | 0.5731 | 0.6474 | 0.5739 | 0.6485 | 36 | 0.9722 | 0.8583 | 0.9951 | 36 | 0.01302 | 0.02048 | 12 |
| mixture_extremes | oracle_coords_only | depth4_light | 0.01751 | 0.001661 | 0.05389 | 0.03239 | 0.02484 | 0.2818 | 0.3091 | 0.008477 | 0.0009895 | 0.01284 | 0.003552 | 0.0001628 | 1.272e-05 | 0.0003729 | 3.862e-05 | 0.5731 | 0.6474 | 0.5739 | 0.6485 | 36 | 0.9444 | 0.8186 | 0.9846 | 36 | 0.01647 | 0.01859 | 12 |
| mixture_extremes | oracle_coords_only | depth7_high_capacity | 0.02672 | 0.005428 | 0.05132 | 0.03063 | 0.03435 | 0.2621 | 0.3019 | 0.005658 | 0.002228 | 0.01183 | 0.00371 | 0.0001183 | 1.363e-05 | 0.0003778 | 4.8e-05 | 0.5731 | 0.6474 | 0.5739 | 0.6485 | 36 | 0.9722 | 0.8583 | 0.9951 | 36 | 0.02466 | 0.0305 | 12 |
| mixture_extremes | oracle_s_only | baseline | 0.02511 | 0.003861 | 0.06997 | 0.03444 | 0.04256 | 0.2779 | 0.3209 | 0.006994 | 0.001732 | 0.01177 | 0.002488 | 0.0001327 | 8.773e-06 | 0.000409 | 2.083e-05 | 0.5731 | 0.6474 | 0.5739 | 0.6485 | 36 | 0.9722 | 0.8583 | 0.9951 | 36 | 0.0221 | 0.03252 | 12 |
| mixture_extremes | oracle_s_only | depth4_light | 0.02728 | 0.0005993 | 0.07662 | 0.03797 | 0.04325 | 0.2818 | 0.3289 | 0.008477 | 0.0009407 | 0.01284 | 0.002656 | 0.0001628 | 3.567e-06 | 0.0003729 | 3.46e-05 | 0.5731 | 0.6474 | 0.5739 | 0.6485 | 36 | 0.9167 | 0.7817 | 0.9713 | 36 | 0.02102 | 0.03439 | 12 |
| mixture_extremes | oracle_s_only | depth7_high_capacity | 0.03238 | 0.005366 | 0.07282 | 0.03721 | 0.04512 | 0.2621 | 0.3107 | 0.005658 | 0.001899 | 0.01183 | 0.003755 | 0.0001183 | 1.497e-05 | 0.0003778 | 2.871e-05 | 0.5731 | 0.6474 | 0.5739 | 0.6485 | 36 | 0.9722 | 0.8583 | 0.9951 | 36 | 0.01816 | 0.03626 | 12 |
| shift_naive | oracle_coords_only | baseline | 0.01551 | -0.01009 | 0.09478 | 0.06533 | 0.05853 | 0.2952 | 0.3403 | 0.01092 | 0.003749 | 0.01669 | 0.005581 | 0.0002816 | 3.944e-05 | 0.0006343 | 9.634e-05 | 0.6302 | 0.7603 | 0.7226 | 0.9 | 36 | 0.7222 | 0.5601 | 0.8415 | 36 | 0.01433 | 0.01755 | 12 |
| shift_naive | oracle_coords_only | depth4_light | 0.01613 | -0.007894 | 0.08802 | 0.0616 | 0.04818 | 0.3443 | 0.3645 | 0.01077 | 0.001946 | 0.0171 | 0.003725 | 0.0002874 | 1.402e-05 | 0.0007909 | 4.775e-05 | 0.6302 | 0.7603 | 0.7226 | 0.9 | 36 | 0.8056 | 0.6497 | 0.9025 | 36 | 0.01155 | 0.02025 | 12 |
| shift_naive | oracle_coords_only | depth7_high_capacity | 0.01732 | -0.001171 | 0.1019 | 0.05901 | 0.04329 | 0.3058 | 0.338 | 0.01098 | 0.005722 | 0.01734 | 0.006012 | 0.0003409 | 7.441e-05 | 0.0005982 | 0.0001091 | 0.6302 | 0.7603 | 0.7226 | 0.9 | 36 | 0.8611 | 0.7134 | 0.9392 | 36 | 0.01385 | 0.01815 | 12 |
| shift_naive | oracle_s_only | baseline | 0.02254 | -0.007971 | 0.09204 | 0.06302 | 0.058 | 0.2952 | 0.3621 | 0.01092 | 0.002086 | 0.01669 | 0.004577 | 0.0002816 | 1.922e-05 | 0.0006343 | 4.928e-05 | 0.6302 | 0.7603 | 0.7226 | 0.9 | 36 | 0.8333 | 0.6811 | 0.9213 | 36 | 0.02152 | 0.02901 | 12 |
| shift_naive | oracle_s_only | depth4_light | 0.02312 | -0.01224 | 0.09448 | 0.05822 | 0.06286 | 0.3443 | 0.3523 | 0.01077 | 0.0009393 | 0.0171 | 0.002253 | 0.0002874 | 4.527e-06 | 0.0007909 | 1.535e-05 | 0.6302 | 0.7603 | 0.7226 | 0.9 | 36 | 0.7222 | 0.5601 | 0.8415 | 36 | 0.02068 | 0.03024 | 12 |
| shift_naive | oracle_s_only | depth7_high_capacity | 0.02703 | -0.008056 | 0.1127 | 0.05597 | 0.05378 | 0.3058 | 0.3572 | 0.01098 | 0.004132 | 0.01734 | 0.005624 | 0.0003409 | 4.283e-05 | 0.0005982 | 9.075e-05 | 0.6302 | 0.7603 | 0.7226 | 0.9 | 36 | 0.8056 | 0.6497 | 0.9025 | 36 | 0.02322 | 0.02807 | 12 |
| shift_preserve | oracle_coords_only | baseline | 0.1464 | 0.01737 | 0.3922 | 0.1726 | 0.1854 | 0.1256 | 0.3517 | 0.01127 | 0.001998 | 0.0175 | 0.003509 | 0.0002511 | 1.492e-05 | 0.0007314 | 3.893e-05 | 0.6295 | 0.7596 | 0.63 | 0.7595 | 36 | 0.9722 | 0.8583 | 0.9951 | 36 | 0.1282 | 0.1571 | 12 |
| shift_preserve | oracle_coords_only | depth4_light | 0.1497 | 0.01732 | 0.3976 | 0.1821 | 0.2366 | 0.1246 | 0.3677 | 0.01092 | 0.00147 | 0.01489 | 0.001299 | 0.0002732 | 1.061e-05 | 0.0005424 | 8.08e-06 | 0.6295 | 0.7596 | 0.63 | 0.7595 | 36 | 0.9722 | 0.8583 | 0.9951 | 36 | 0.1434 | 0.1525 | 12 |
| shift_preserve | oracle_coords_only | depth7_high_capacity | 0.145 | 0.01274 | 0.3628 | 0.1663 | 0.2053 | 0.1164 | 0.3144 | 0.009721 | 0.003232 | 0.01601 | 0.003251 | 0.0002137 | 3.579e-05 | 0.0008433 | 2.807e-05 | 0.6295 | 0.7596 | 0.63 | 0.7595 | 36 | 1 | 0.9036 | 1 | 36 | 0.1376 | 0.1507 | 12 |
| shift_preserve | oracle_s_only | baseline | 0.1512 | 0.01888 | 0.4181 | 0.1778 | 0.2096 | 0.1256 | 0.3637 | 0.01127 | 0.001324 | 0.0175 | 0.002226 | 0.0002511 | 6.821e-06 | 0.0007314 | 1.422e-05 | 0.6295 | 0.7596 | 0.63 | 0.7595 | 36 | 1 | 0.9036 | 1 | 36 | 0.1322 | 0.1592 | 12 |
| shift_preserve | oracle_s_only | depth4_light | 0.1603 | 0.02187 | 0.4415 | 0.1872 | 0.2424 | 0.1246 | 0.3735 | 0.01092 | 0.0004161 | 0.01489 | 0.001359 | 0.0002732 | 2.357e-06 | 0.0005424 | 1.502e-05 | 0.6295 | 0.7596 | 0.63 | 0.7595 | 36 | 0.9722 | 0.8583 | 0.9951 | 36 | 0.1437 | 0.172 | 12 |
| shift_preserve | oracle_s_only | depth7_high_capacity | 0.1591 | 0.01677 | 0.4207 | 0.172 | 0.2217 | 0.1164 | 0.3619 | 0.009721 | 0.002347 | 0.01601 | 0.004951 | 0.0002137 | 1.591e-05 | 0.0008433 | 4.89e-05 | 0.6295 | 0.7596 | 0.63 | 0.7595 | 36 | 1 | 0.9036 | 1 | 36 | 0.1429 | 0.1658 | 12 |
| tail_corr | oracle_coords_only | baseline | 0.006558 | -0.005747 | 0.03733 | 0.02375 | 0.01898 | 0.1366 | 0.1672 | 0.01053 | 0.002315 | 0.01128 | 0.003626 | 0.0001603 | 1.465e-05 | 0.0004002 | 6.57e-05 | 0.6024 | 0.7076 | 0.6026 | 0.7074 | 108 | 0.7685 | 0.6806 | 0.838 | 108 | 0.00597 | 0.007714 | 12 |
| tail_corr | oracle_coords_only | depth4_light | 0.006291 | -0.005727 | 0.03607 | 0.02295 | 0.01863 | 0.1414 | 0.1693 | 0.01112 | 0.001881 | 0.01085 | 0.00317 | 0.0001767 | 1.127e-05 | 0.0004201 | 4.828e-05 | 0.6024 | 0.7076 | 0.6026 | 0.7074 | 108 | 0.75 | 0.6607 | 0.8221 | 108 | 0.005532 | 0.007245 | 12 |
| tail_corr | oracle_coords_only | depth7_high_capacity | 0.007131 | -0.003708 | 0.03618 | 0.02083 | 0.01707 | 0.1342 | 0.1661 | 0.01077 | 0.002942 | 0.009342 | 0.003843 | 0.0002223 | 2.541e-05 | 0.0003614 | 6.867e-05 | 0.6024 | 0.7076 | 0.6026 | 0.7074 | 108 | 0.7778 | 0.6906 | 0.8459 | 108 | 0.00661 | 0.007818 | 12 |
| tail_corr | oracle_s_only | baseline | 0.01109 | -0.003908 | 0.05684 | 0.02773 | 0.02482 | 0.1366 | 0.1689 | 0.01053 | 0.002311 | 0.01128 | 0.003095 | 0.0001603 | 1.686e-05 | 0.0004002 | 2.605e-05 | 0.6024 | 0.7076 | 0.6026 | 0.7074 | 108 | 0.8426 | 0.7623 | 0.8993 | 108 | 0.008937 | 0.01201 | 12 |
| tail_corr | oracle_s_only | depth4_light | 0.01276 | -0.004433 | 0.05164 | 0.02757 | 0.02263 | 0.1414 | 0.1722 | 0.01112 | 0.001104 | 0.01085 | 0.001916 | 0.0001767 | 4.951e-06 | 0.0004201 | 1.265e-05 | 0.6024 | 0.7076 | 0.6026 | 0.7074 | 108 | 0.7963 | 0.7108 | 0.8615 | 108 | 0.009725 | 0.01427 | 12 |
| tail_corr | oracle_s_only | depth7_high_capacity | 0.01185 | -0.002305 | 0.05202 | 0.02562 | 0.02241 | 0.1342 | 0.1599 | 0.01077 | 0.002111 | 0.009342 | 0.003079 | 0.0002223 | 1.417e-05 | 0.0003614 | 4.374e-05 | 0.6024 | 0.7076 | 0.6026 | 0.7074 | 108 | 0.8056 | 0.721 | 0.8692 | 108 | 0.01138 | 0.01381 | 12 |
### Relative Δ% (oracle vs raw) by regime family

| regime_family | oracle_mode | xgb_config_id | prauc_oracle_median | prauc_raw_median | relative_drop_pct |
| --- | --- | --- | --- | --- | --- |
| shift_preserve | oracle_s_only | depth7_high_capacity | 0.3619 | 0.1164 | 67.84 |
| shift_preserve | oracle_s_only | depth4_light | 0.3735 | 0.1246 | 66.63 |
| shift_preserve | oracle_coords_only | depth4_light | 0.3677 | 0.1246 | 66.11 |
| shift_preserve | oracle_s_only | baseline | 0.3637 | 0.1256 | 65.47 |
| shift_preserve | oracle_coords_only | baseline | 0.3517 | 0.1256 | 64.3 |
| shift_preserve | oracle_coords_only | depth7_high_capacity | 0.3144 | 0.1164 | 62.97 |
| tail_corr | oracle_coords_only | depth7_high_capacity | 0.1661 | 0.1342 | 19.2 |
| tail_corr | oracle_s_only | baseline | 0.1689 | 0.1366 | 19.09 |
| shift_naive | oracle_s_only | baseline | 0.3621 | 0.2952 | 18.47 |
| tail_corr | oracle_coords_only | baseline | 0.1672 | 0.1366 | 18.3 |
| tail_corr | oracle_s_only | depth4_light | 0.1722 | 0.1414 | 17.91 |
| tail_corr | oracle_coords_only | depth4_light | 0.1693 | 0.1414 | 16.47 |
| tail_corr | oracle_s_only | depth7_high_capacity | 0.1599 | 0.1342 | 16.07 |
| mixture_extremes | oracle_s_only | depth7_high_capacity | 0.3107 | 0.2621 | 15.64 |
| shift_naive | oracle_s_only | depth7_high_capacity | 0.3572 | 0.3058 | 14.4 |
| mixture_extremes | oracle_s_only | depth4_light | 0.3289 | 0.2818 | 14.32 |
| mixture_extremes | oracle_s_only | baseline | 0.3209 | 0.2779 | 13.39 |
| shift_naive | oracle_coords_only | baseline | 0.3403 | 0.2952 | 13.24 |
| mixture_extremes | oracle_coords_only | depth7_high_capacity | 0.3019 | 0.2621 | 13.2 |
| shift_naive | oracle_coords_only | depth7_high_capacity | 0.338 | 0.3058 | 9.532 |
| mixture_extremes | oracle_coords_only | depth4_light | 0.3091 | 0.2818 | 8.835 |
| mixture_extremes | oracle_coords_only | baseline | 0.3025 | 0.2779 | 8.125 |
| shift_naive | oracle_coords_only | depth4_light | 0.3645 | 0.3443 | 5.538 |
| shift_naive | oracle_s_only | depth4_light | 0.3523 | 0.3443 | 2.281 |
## Summary by level × regime family

| level | regime_family | oracle_mode | xgb_config_id | delta_prauc_median | delta_prauc_p10 | delta_prauc_p90 | delta_prauc_std | delta_prauc_iqr | prauc_raw_median | prauc_oracle_median | ratio_scale_invariance_raw_median | ratio_scale_invariance_oracle_median | product_comp_invariance_raw_median | product_comp_invariance_oracle_median | iso_var_ratio_raw_median | iso_var_ratio_oracle_median | iso_var_product_raw_median | iso_var_product_oracle_median | dominance_train_mean_median | dominance_train_p90_median | dominance_test_mean_median | dominance_test_p90_median | n_runs | delta_prauc_pos_rate_frac_pos | delta_prauc_pos_rate_ci_lo | delta_prauc_pos_rate_ci_hi | delta_prauc_pos_rate_n_eff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | mixture_extremes | oracle_coords_only | baseline | 0.0114 | 0.003071 | 0.02198 | 0.007886 | 0.01148 | 0.3178 | 0.3292 | 0.004991 | 0.001976 | 0.009173 | 0.003651 | 4.546e-05 | 9.498e-06 | 0.0001585 | 2.599e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 1 | mixture_extremes | oracle_coords_only | depth4_light | 0.005711 | -0.0002896 | 0.01246 | 0.005884 | 0.006079 | 0.331 | 0.3318 | 0.009922 | 0.0009895 | 0.01284 | 0.003552 | 0.0002048 | 3.078e-06 | 0.0003415 | 2.599e-05 |  |  |  |  | 6 | 0.8333 | 0.4365 | 0.9699 | 6 |
| 1 | mixture_extremes | oracle_coords_only | depth7_high_capacity | 0.01911 | 0.01054 | 0.03117 | 0.009254 | 0.01488 | 0.3118 | 0.3305 | 0.007825 | 0.001321 | 0.01113 | 0.002637 | 0.0001183 | 8.05e-06 | 0.0002879 | 1.625e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 1 | mixture_extremes | oracle_s_only | baseline | 0.01175 | 0.0005752 | 0.02339 | 0.009664 | 0.01419 | 0.3178 | 0.3271 | 0.004991 | 0.002056 | 0.009173 | 0.002488 | 4.546e-05 | 7.835e-06 | 0.0001585 | 1.453e-05 |  |  |  |  | 6 | 0.8333 | 0.4365 | 0.9699 | 6 |
| 1 | mixture_extremes | oracle_s_only | depth4_light | 0.004937 | -0.008411 | 0.01126 | 0.008488 | 0.01208 | 0.331 | 0.3302 | 0.009922 | 0.001786 | 0.01284 | 0.003367 | 0.0002048 | 6.75e-06 | 0.0003415 | 3.46e-05 |  |  |  |  | 6 | 0.6667 | 0.3 | 0.9032 | 6 |
| 1 | mixture_extremes | oracle_s_only | depth7_high_capacity | 0.01113 | 0.007983 | 0.03238 | 0.01087 | 0.01756 | 0.3118 | 0.3229 | 0.007825 | 0.003148 | 0.01113 | 0.003755 | 0.0001183 | 2.551e-05 | 0.0002879 | 2.994e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 1 | shift_naive | oracle_coords_only | baseline | -0.002539 | -0.01789 | 0.03606 | 0.02831 | 0.01828 | 0.3476 | 0.3514 | 0.0132 | 0.003749 | 0.02855 | 0.0132 | 0.0002816 | 3.278e-05 | 0.002033 | 0.0005325 |  |  |  |  | 6 | 0.5 | 0.1876 | 0.8124 | 6 |
| 1 | shift_naive | oracle_coords_only | depth4_light | 0.005211 | -0.02683 | 0.05052 | 0.03418 | 0.03927 | 0.3581 | 0.3554 | 0.01129 | 0.001521 | 0.02947 | 0.01075 | 0.0002874 | 6.259e-06 | 0.003241 | 0.0003695 |  |  |  |  | 6 | 0.5 | 0.1876 | 0.8124 | 6 |
| 1 | shift_naive | oracle_coords_only | depth7_high_capacity | 0.009745 | -0.03178 | 0.0666 | 0.04773 | 0.02697 | 0.3314 | 0.3472 | 0.0161 | 0.006783 | 0.01617 | 0.006809 | 0.0003733 | 7.441e-05 | 0.000562 | 0.0001251 |  |  |  |  | 6 | 0.6667 | 0.3 | 0.9032 | 6 |
| 1 | shift_naive | oracle_s_only | baseline | -0.004561 | -0.01998 | 0.04024 | 0.03209 | 0.01429 | 0.3476 | 0.3467 | 0.0132 | 0.002086 | 0.02855 | 0.009592 | 0.0002816 | 1.922e-05 | 0.002033 | 0.0002236 |  |  |  |  | 6 | 0.3333 | 0.09677 | 0.7 | 6 |
| 1 | shift_naive | oracle_s_only | depth4_light | -0.004769 | -0.01683 | 0.04091 | 0.02685 | 0.03029 | 0.3581 | 0.3663 | 0.01129 | 0.001578 | 0.02947 | 0.005739 | 0.0002874 | 4.527e-06 | 0.003241 | 0.0001497 |  |  |  |  | 6 | 0.3333 | 0.09677 | 0.7 | 6 |
| 1 | shift_naive | oracle_s_only | depth7_high_capacity | 0.0004355 | -0.02348 | 0.06812 | 0.04605 | 0.04231 | 0.3314 | 0.3296 | 0.0161 | 0.004515 | 0.01617 | 0.00922 | 0.0003733 | 4.283e-05 | 0.000562 | 0.0002265 |  |  |  |  | 6 | 0.5 | 0.1876 | 0.8124 | 6 |
| 1 | shift_preserve | oracle_coords_only | baseline | 0.1899 | 0.07758 | 0.3319 | 0.1182 | 0.2162 | 0.1303 | 0.3215 | 0.01127 | 0.001644 | 0.03065 | 0.004555 | 0.0002442 | 9.287e-06 | 0.002063 | 5.128e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 1 | shift_preserve | oracle_coords_only | depth4_light | 0.19 | 0.04839 | 0.3384 | 0.1315 | 0.2513 | 0.1561 | 0.3302 | 0.01092 | 0.001414 | 0.02481 | 0.0006077 | 0.0002732 | 1.061e-05 | 0.001528 | 7.214e-06 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 1 | shift_preserve | oracle_coords_only | depth7_high_capacity | 0.2125 | 0.07558 | 0.3298 | 0.1186 | 0.2211 | 0.1277 | 0.3243 | 0.009041 | 0.001089 | 0.02889 | 0.007665 | 0.0002184 | 7.942e-06 | 0.002115 | 0.000131 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 1 | shift_preserve | oracle_s_only | baseline | 0.203 | 0.08896 | 0.3289 | 0.1143 | 0.2087 | 0.1303 | 0.3277 | 0.01127 | 0.0007563 | 0.03065 | 0.004736 | 0.0002442 | 2.18e-06 | 0.002063 | 7.777e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 1 | shift_preserve | oracle_s_only | depth4_light | 0.1944 | 0.05516 | 0.3372 | 0.1287 | 0.2459 | 0.1561 | 0.3357 | 0.01092 | 0.0009085 | 0.02481 | 0.002309 | 0.0002732 | 2.384e-06 | 0.001528 | 1.671e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 1 | shift_preserve | oracle_s_only | depth7_high_capacity | 0.2123 | 0.09533 | 0.3233 | 0.1066 | 0.2011 | 0.1277 | 0.3298 | 0.009041 | 0.001854 | 0.02889 | 0.005172 | 0.0002184 | 1.285e-05 | 0.002115 | 6.357e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 1 | tail_corr | oracle_coords_only | baseline | 0.002269 | -0.01076 | 0.01684 | 0.01206 | 0.008221 | 0.1903 | 0.2018 | 0.01093 | 0.001945 | 0.01228 | 0.003612 | 0.0001632 | 1.106e-05 | 0.0004129 | 6.563e-05 |  |  |  |  | 18 | 0.6111 | 0.3862 | 0.7969 | 18 |
| 1 | tail_corr | oracle_coords_only | depth4_light | 0.002107 | -0.008359 | 0.02055 | 0.01136 | 0.01098 | 0.1889 | 0.2 | 0.0112 | 0.002048 | 0.01056 | 0.002186 | 0.0001858 | 8.272e-06 | 0.0004798 | 2.716e-05 |  |  |  |  | 18 | 0.6111 | 0.3862 | 0.7969 | 18 |
| 1 | tail_corr | oracle_coords_only | depth7_high_capacity | -0.0003336 | -0.01334 | 0.01433 | 0.01342 | 0.01515 | 0.1814 | 0.1836 | 0.01101 | 0.002292 | 0.009227 | 0.004372 | 0.0002389 | 1.298e-05 | 0.0003858 | 6.64e-05 |  |  |  |  | 18 | 0.5 | 0.2903 | 0.7097 | 18 |
| 1 | tail_corr | oracle_s_only | baseline | 0.004958 | -0.01035 | 0.0179 | 0.01278 | 0.01373 | 0.1903 | 0.2027 | 0.01093 | 0.002529 | 0.01228 | 0.003995 | 0.0001632 | 1.601e-05 | 0.0004129 | 4.819e-05 |  |  |  |  | 18 | 0.6667 | 0.4375 | 0.8372 | 18 |
| 1 | tail_corr | oracle_s_only | depth4_light | 0.003672 | -0.0127 | 0.01714 | 0.01305 | 0.02039 | 0.1889 | 0.2037 | 0.0112 | 0.001457 | 0.01056 | 0.002646 | 0.0001858 | 5.101e-06 | 0.0004798 | 1.923e-05 |  |  |  |  | 18 | 0.6111 | 0.3862 | 0.7969 | 18 |
| 1 | tail_corr | oracle_s_only | depth7_high_capacity | -0.0007994 | -0.01868 | 0.02325 | 0.01541 | 0.01836 | 0.1814 | 0.1898 | 0.01101 | 0.002226 | 0.009227 | 0.004644 | 0.0002389 | 1.269e-05 | 0.0003858 | 7.401e-05 |  |  |  |  | 18 | 0.4444 | 0.2456 | 0.6628 | 18 |
| 2 | mixture_extremes | oracle_coords_only | baseline | 0.01798 | 0.006812 | 0.03789 | 0.01338 | 0.01995 | 0.3053 | 0.3177 | 0.007996 | 0.0004963 | 0.01266 | 0.005249 | 0.0001206 | 2.172e-06 | 0.000409 | 5.723e-05 | 0.5731 | 0.6474 | 0.5739 | 0.6485 | 6 | 1 | 0.6097 | 1 | 6 |
| 2 | mixture_extremes | oracle_coords_only | depth4_light | 0.02373 | 0.01331 | 0.03383 | 0.009491 | 0.0163 | 0.3081 | 0.332 | 0.008477 | 0.0003016 | 0.01447 | 0.004144 | 0.0001269 | 1.211e-06 | 0.0005304 | 4.432e-05 | 0.5731 | 0.6474 | 0.5739 | 0.6485 | 6 | 1 | 0.6097 | 1 | 6 |
| 2 | mixture_extremes | oracle_coords_only | depth7_high_capacity | 0.02654 | 0.007835 | 0.04639 | 0.01782 | 0.03362 | 0.2985 | 0.3251 | 0.004124 | 0.002552 | 0.01395 | 0.00547 | 6.18e-05 | 1.098e-05 | 0.0003958 | 7.012e-05 | 0.5731 | 0.6474 | 0.5739 | 0.6485 | 6 | 1 | 0.6097 | 1 | 6 |
| 2 | mixture_extremes | oracle_s_only | baseline | 0.02298 | 0.01018 | 0.03339 | 0.009621 | 0.01804 | 0.3053 | 0.3232 | 0.007996 | 0.002251 | 0.01266 | 0.004799 | 0.0001206 | 8.902e-06 | 0.000409 | 4.861e-05 | 0.5731 | 0.6474 | 0.5739 | 0.6485 | 6 | 1 | 0.6097 | 1 | 6 |
| 2 | mixture_extremes | oracle_s_only | depth4_light | 0.02728 | 0.01476 | 0.04051 | 0.0111 | 0.01472 | 0.3081 | 0.3324 | 0.008477 | 0.0007849 | 0.01447 | 0.004263 | 0.0001269 | 1.343e-06 | 0.0005304 | 3.987e-05 | 0.5731 | 0.6474 | 0.5739 | 0.6485 | 6 | 1 | 0.6097 | 1 | 6 |
| 2 | mixture_extremes | oracle_s_only | depth7_high_capacity | 0.01813 | 0.008598 | 0.04437 | 0.01535 | 0.02318 | 0.2985 | 0.3117 | 0.004124 | 0.001356 | 0.01395 | 0.005123 | 6.18e-05 | 6.372e-06 | 0.0003958 | 5.152e-05 | 0.5731 | 0.6474 | 0.5739 | 0.6485 | 6 | 1 | 0.6097 | 1 | 6 |
| 2 | shift_naive | oracle_coords_only | baseline | 0.00545 | -0.03635 | 0.05561 | 0.0426 | 0.04129 | 0.3683 | 0.3714 | 0.006793 | 0.001002 | 0.01573 | 0.005581 | 9.348e-05 | 3.864e-06 | 0.0006343 | 7.112e-05 | 0.6302 | 0.7603 | 0.7226 | 0.9 | 6 | 0.5 | 0.1876 | 0.8124 | 6 |
| 2 | shift_naive | oracle_coords_only | depth4_light | 0.006491 | -0.02865 | 0.05962 | 0.03967 | 0.04377 | 0.3782 | 0.3825 | 0.004929 | 0.0009391 | 0.01596 | 0.005128 | 4.318e-05 | 1.59e-06 | 0.0007909 | 5.37e-05 | 0.6302 | 0.7603 | 0.7226 | 0.9 | 6 | 0.6667 | 0.3 | 0.9032 | 6 |
| 2 | shift_naive | oracle_coords_only | depth7_high_capacity | 0.01517 | 0.005805 | 0.05896 | 0.02363 | 0.03395 | 0.3612 | 0.3716 | 0.01098 | 0.004815 | 0.02147 | 0.00464 | 0.0002021 | 3.054e-05 | 0.0007009 | 5.41e-05 | 0.6302 | 0.7603 | 0.7226 | 0.9 | 6 | 1 | 0.6097 | 1 | 6 |
| 2 | shift_naive | oracle_s_only | baseline | 0.0135 | -0.02434 | 0.06689 | 0.04044 | 0.04942 | 0.3683 | 0.3818 | 0.006793 | 0.001503 | 0.01573 | 0.005543 | 9.348e-05 | 5.805e-06 | 0.0006343 | 0.0001043 | 0.6302 | 0.7603 | 0.7226 | 0.9 | 6 | 0.8333 | 0.4365 | 0.9699 | 6 |
| 2 | shift_naive | oracle_s_only | depth4_light | -0.003153 | -0.03895 | 0.05993 | 0.04606 | 0.03885 | 0.3782 | 0.3693 | 0.004929 | 0.0003388 | 0.01596 | 0.003824 | 4.318e-05 | 6.596e-07 | 0.0007909 | 5.211e-05 | 0.6302 | 0.7603 | 0.7226 | 0.9 | 6 | 0.3333 | 0.09677 | 0.7 | 6 |
| 2 | shift_naive | oracle_s_only | depth7_high_capacity | 0.02035 | -0.002028 | 0.06056 | 0.02658 | 0.0418 | 0.3612 | 0.3803 | 0.01098 | 0.00282 | 0.02147 | 0.007445 | 0.0002021 | 1.627e-05 | 0.0007009 | 0.0001384 | 0.6302 | 0.7603 | 0.7226 | 0.9 | 6 | 0.8333 | 0.4365 | 0.9699 | 6 |
| 2 | shift_preserve | oracle_coords_only | baseline | 0.09575 | 0.05578 | 0.2356 | 0.0776 | 0.139 | 0.1203 | 0.2643 | 0.01077 | 0.001138 | 0.0175 | 0.004678 | 0.0002511 | 4.786e-06 | 0.000754 | 4.159e-05 | 0.6295 | 0.7596 | 0.63 | 0.7595 | 6 | 1 | 0.6097 | 1 | 6 |
| 2 | shift_preserve | oracle_coords_only | depth4_light | 0.09185 | 0.0595 | 0.2431 | 0.08165 | 0.1253 | 0.1335 | 0.2673 | 0.01043 | 0.001301 | 0.01127 | 0.002908 | 0.0002609 | 5.388e-06 | 0.0005424 | 2.126e-05 | 0.6295 | 0.7596 | 0.63 | 0.7595 | 6 | 1 | 0.6097 | 1 | 6 |
| 2 | shift_preserve | oracle_coords_only | depth7_high_capacity | 0.09636 | 0.04573 | 0.2372 | 0.08213 | 0.1502 | 0.1204 | 0.2597 | 0.009721 | 0.00304 | 0.02252 | 0.005199 | 0.0002094 | 2.145e-05 | 0.000912 | 6.051e-05 | 0.6295 | 0.7596 | 0.63 | 0.7595 | 6 | 1 | 0.6097 | 1 | 6 |
| 2 | shift_preserve | oracle_s_only | baseline | 0.08291 | 0.0511 | 0.2577 | 0.09261 | 0.1581 | 0.1203 | 0.254 | 0.01077 | 0.001023 | 0.0175 | 0.005861 | 0.0002511 | 7.206e-06 | 0.000754 | 9.597e-05 | 0.6295 | 0.7596 | 0.63 | 0.7595 | 6 | 1 | 0.6097 | 1 | 6 |
| 2 | shift_preserve | oracle_s_only | depth4_light | 0.08613 | 0.04523 | 0.2449 | 0.08899 | 0.1338 | 0.1335 | 0.2671 | 0.01043 | 0.0002136 | 0.01127 | 0.0005988 | 0.0002609 | 5.938e-07 | 0.0005424 | 2.185e-06 | 0.6295 | 0.7596 | 0.63 | 0.7595 | 6 | 1 | 0.6097 | 1 | 6 |
| 2 | shift_preserve | oracle_s_only | depth7_high_capacity | 0.09124 | 0.04852 | 0.2466 | 0.08577 | 0.1473 | 0.1204 | 0.2545 | 0.009721 | 0.003576 | 0.02252 | 0.007475 | 0.0002094 | 4.892e-05 | 0.000912 | 0.0001 | 0.6295 | 0.7596 | 0.63 | 0.7595 | 6 | 1 | 0.6097 | 1 | 6 |
| 2 | tail_corr | oracle_coords_only | baseline | 0.00484 | -0.004475 | 0.02323 | 0.01183 | 0.01233 | 0.1046 | 0.1228 | 0.01044 | 0.002422 | 0.01178 | 0.003778 | 0.0001559 | 1.226e-05 | 0.0003869 | 3.416e-05 | 0.6024 | 0.7076 | 0.6026 | 0.7074 | 18 | 0.7222 | 0.4913 | 0.875 | 18 |
| 2 | tail_corr | oracle_coords_only | depth4_light | 0.004794 | -0.006793 | 0.02675 | 0.012 | 0.01261 | 0.1106 | 0.1292 | 0.01081 | 0.0009372 | 0.01109 | 0.003148 | 0.0001595 | 3.138e-06 | 0.0003759 | 2.74e-05 | 0.6024 | 0.7076 | 0.6026 | 0.7074 | 18 | 0.6667 | 0.4375 | 0.8372 | 18 |
| 2 | tail_corr | oracle_coords_only | depth7_high_capacity | 0.008507 | -0.0003368 | 0.03002 | 0.01109 | 0.0134 | 0.1002 | 0.1208 | 0.01036 | 0.001729 | 0.009744 | 0.004367 | 0.0001823 | 8.086e-06 | 0.0004109 | 6.629e-05 | 0.6024 | 0.7076 | 0.6026 | 0.7074 | 18 | 0.8889 | 0.672 | 0.969 | 18 |
| 2 | tail_corr | oracle_s_only | baseline | 0.003795 | -0.004613 | 0.02472 | 0.01283 | 0.01227 | 0.1046 | 0.1291 | 0.01044 | 0.002129 | 0.01178 | 0.004263 | 0.0001559 | 8.835e-06 | 0.0003869 | 4.672e-05 | 0.6024 | 0.7076 | 0.6026 | 0.7074 | 18 | 0.8333 | 0.6078 | 0.9416 | 18 |
| 2 | tail_corr | oracle_s_only | depth4_light | 0.004748 | -0.009865 | 0.02522 | 0.01469 | 0.01492 | 0.1106 | 0.124 | 0.01081 | 0.001029 | 0.01109 | 0.002227 | 0.0001595 | 1.784e-06 | 0.0003759 | 1.62e-05 | 0.6024 | 0.7076 | 0.6026 | 0.7074 | 18 | 0.6111 | 0.3862 | 0.7969 | 18 |
| 2 | tail_corr | oracle_s_only | depth7_high_capacity | 0.003968 | -0.006556 | 0.02601 | 0.01273 | 0.01938 | 0.1002 | 0.1131 | 0.01036 | 0.001929 | 0.009744 | 0.004671 | 0.0001823 | 9.607e-06 | 0.0004109 | 6.282e-05 | 0.6024 | 0.7076 | 0.6026 | 0.7074 | 18 | 0.7222 | 0.4913 | 0.875 | 18 |
| 3 | mixture_extremes | oracle_coords_only | baseline | 0.0338 | 0.02175 | 0.04154 | 0.00969 | 0.00879 | 0.3006 | 0.324 | 0.0132 | 0.002667 | 0.01494 | 0.007276 | 0.0003467 | 2.715e-05 | 0.0004636 | 7.687e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 3 | mixture_extremes | oracle_coords_only | depth4_light | 0.0239 | 0.01839 | 0.03054 | 0.005832 | 0.004777 | 0.3095 | 0.3279 | 0.01539 | 0.002196 | 0.01504 | 0.006081 | 0.0006677 | 2.39e-05 | 0.0007196 | 0.0001131 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 3 | mixture_extremes | oracle_coords_only | depth7_high_capacity | 0.04562 | 0.04077 | 0.05132 | 0.004927 | 0.004981 | 0.2854 | 0.3266 | 0.01238 | 0.002645 | 0.01217 | 0.00758 | 0.0003314 | 2.399e-05 | 0.0004115 | 0.0001712 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 3 | mixture_extremes | oracle_s_only | baseline | 0.05263 | 0.03777 | 0.06312 | 0.01106 | 0.01663 | 0.3006 | 0.353 | 0.0132 | 0.001611 | 0.01494 | 0.004139 | 0.0003467 | 1.426e-05 | 0.0004636 | 2.5e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 3 | mixture_extremes | oracle_s_only | depth4_light | 0.04168 | 0.03646 | 0.05432 | 0.008046 | 0.00931 | 0.3095 | 0.3581 | 0.01539 | 0.001675 | 0.01504 | 0.002244 | 0.0006677 | 9.597e-06 | 0.0007196 | 2.009e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 3 | mixture_extremes | oracle_s_only | depth7_high_capacity | 0.06076 | 0.05094 | 0.07282 | 0.01019 | 0.01258 | 0.2854 | 0.3458 | 0.01238 | 0.004016 | 0.01217 | 0.004746 | 0.0003314 | 4.394e-05 | 0.0004115 | 2.871e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 3 | shift_naive | oracle_coords_only | baseline | 0.01943 | 0.002417 | 0.06977 | 0.03473 | 0.01955 | 0.2263 | 0.2723 | 0.04985 | 0.01366 | 0.01432 | 0.004261 | 0.003815 | 0.0003529 | 0.0005528 | 7.118e-05 |  |  |  |  | 6 | 0.8333 | 0.4365 | 0.9699 | 6 |
| 3 | shift_naive | oracle_coords_only | depth4_light | 0.01146 | -0.003927 | 0.03228 | 0.01643 | 0.01614 | 0.2555 | 0.2692 | 0.04699 | 0.006858 | 0.01427 | 0.002312 | 0.003379 | 8.439e-05 | 0.0004489 | 1.85e-05 |  |  |  |  | 6 | 0.8333 | 0.4365 | 0.9699 | 6 |
| 3 | shift_naive | oracle_coords_only | depth7_high_capacity | 0.01756 | 0.00501 | 0.03203 | 0.01147 | 0.01559 | 0.2437 | 0.2487 | 0.04343 | 0.01731 | 0.01367 | 0.006012 | 0.003077 | 0.0004227 | 0.0004935 | 0.0001035 |  |  |  |  | 6 | 0.8333 | 0.4365 | 0.9699 | 6 |
| 3 | shift_naive | oracle_s_only | baseline | 0.05688 | 0.01508 | 0.06528 | 0.02256 | 0.03041 | 0.2263 | 0.2647 | 0.04985 | 0.0007849 | 0.01432 | 0.001246 | 0.003815 | 1.874e-06 | 0.0005528 | 6.251e-06 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 3 | shift_naive | oracle_s_only | depth4_light | 0.03267 | 0.009411 | 0.05219 | 0.01991 | 0.01828 | 0.2555 | 0.2649 | 0.04699 | 0.0001061 | 0.01427 | 0.002381 | 0.003379 | 1.669e-07 | 0.0004489 | 1.535e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 3 | shift_naive | oracle_s_only | depth7_high_capacity | 0.04963 | 0.003399 | 0.07266 | 0.02966 | 0.04029 | 0.2437 | 0.2728 | 0.04343 | 0.01358 | 0.01367 | 0.005244 | 0.003077 | 0.0002695 | 0.0004935 | 6.879e-05 |  |  |  |  | 6 | 0.8333 | 0.4365 | 0.9699 | 6 |
| 3 | shift_preserve | oracle_coords_only | baseline | 0.1782 | 0.1365 | 0.2162 | 0.0354 | 0.05171 | 0.1779 | 0.3517 | 0.01709 | 0.00166 | 0.008704 | 0.003461 | 0.0006223 | 1.235e-05 | 0.0001158 | 3.893e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 3 | shift_preserve | oracle_coords_only | depth4_light | 0.1719 | 0.146 | 0.2075 | 0.02867 | 0.04721 | 0.1849 | 0.3677 | 0.01975 | 0.00147 | 0.01243 | 0.001785 | 0.0007456 | 8.296e-06 | 0.0002183 | 1.588e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 3 | shift_preserve | oracle_coords_only | depth7_high_capacity | 0.1766 | 0.1497 | 0.2122 | 0.02847 | 0.0447 | 0.1511 | 0.3144 | 0.01524 | 0.003644 | 0.009518 | 0.006706 | 0.0005636 | 3.579e-05 | 0.0001415 | 8.415e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 3 | shift_preserve | oracle_s_only | baseline | 0.1809 | 0.1399 | 0.23 | 0.04197 | 0.05326 | 0.1779 | 0.365 | 0.01709 | 0.003084 | 0.008704 | 0.001512 | 0.0006223 | 1.795e-05 | 0.0001158 | 1.072e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 3 | shift_preserve | oracle_s_only | depth4_light | 0.1845 | 0.1486 | 0.2212 | 0.03298 | 0.04647 | 0.1849 | 0.3735 | 0.01975 | 0.00101 | 0.01243 | 0.0006997 | 0.0007456 | 3.73e-06 | 0.0002183 | 1.526e-06 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 3 | shift_preserve | oracle_s_only | depth7_high_capacity | 0.2097 | 0.1592 | 0.2557 | 0.04216 | 0.06392 | 0.1511 | 0.3661 | 0.01524 | 0.002988 | 0.009518 | 0.005326 | 0.0005636 | 1.611e-05 | 0.0001415 | 5.102e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 3 | tail_corr | oracle_coords_only | baseline | 0.01769 | -0.007976 | 0.04672 | 0.02118 | 0.03356 | 0.1967 | 0.2107 | 0.01452 | 0.003495 | 0.01704 | 0.007567 | 0.0006026 | 3.49e-05 | 0.0005403 | 8.83e-05 |  |  |  |  | 18 | 0.7222 | 0.4913 | 0.875 | 18 |
| 3 | tail_corr | oracle_coords_only | depth4_light | 0.01423 | -0.004323 | 0.03886 | 0.01935 | 0.02858 | 0.2013 | 0.2129 | 0.01399 | 0.003032 | 0.01704 | 0.00638 | 0.000551 | 1.903e-05 | 0.0005131 | 0.0001083 |  |  |  |  | 18 | 0.7222 | 0.4913 | 0.875 | 18 |
| 3 | tail_corr | oracle_coords_only | depth7_high_capacity | 0.006372 | -0.004001 | 0.03161 | 0.01856 | 0.02571 | 0.1942 | 0.1985 | 0.01226 | 0.004703 | 0.01552 | 0.007233 | 0.0003959 | 5.198e-05 | 0.0004684 | 0.0001299 |  |  |  |  | 18 | 0.7222 | 0.4913 | 0.875 | 18 |
| 3 | tail_corr | oracle_s_only | baseline | 0.02768 | 0.0003034 | 0.06464 | 0.02381 | 0.03662 | 0.1967 | 0.2291 | 0.01452 | 0.002968 | 0.01704 | 0.00352 | 0.0006026 | 2.418e-05 | 0.0005403 | 2.215e-05 |  |  |  |  | 18 | 0.8889 | 0.672 | 0.969 | 18 |
| 3 | tail_corr | oracle_s_only | depth4_light | 0.02272 | 0.004643 | 0.05919 | 0.02069 | 0.03384 | 0.2013 | 0.2301 | 0.01399 | 0.00189 | 0.01704 | 0.003963 | 0.000551 | 7.94e-06 | 0.0005131 | 3.333e-05 |  |  |  |  | 18 | 0.9444 | 0.7424 | 0.9901 | 18 |
| 3 | tail_corr | oracle_s_only | depth7_high_capacity | 0.02502 | 0.00201 | 0.05417 | 0.02048 | 0.03441 | 0.1942 | 0.2204 | 0.01226 | 0.004195 | 0.01552 | 0.005023 | 0.0003959 | 3.74e-05 | 0.0004684 | 5.553e-05 |  |  |  |  | 18 | 0.8889 | 0.672 | 0.969 | 18 |
| 4 | mixture_extremes | oracle_coords_only | baseline | 0.05744 | 0.009592 | 0.1164 | 0.05091 | 0.09669 | 0.4525 | 0.5145 | 0.001733 | 0.0005314 | 0.01177 | 0.002961 | 1.519e-05 | 7.704e-06 | 0.0007401 | 5.242e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 4 | mixture_extremes | oracle_coords_only | depth4_light | 0.0681 | 0.01435 | 0.1258 | 0.05414 | 0.1071 | 0.4476 | 0.5164 | 0.0018 | 0.0005552 | 0.0131 | 0.002598 | 2.817e-05 | 1.337e-05 | 0.0007956 | 2.901e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 4 | mixture_extremes | oracle_coords_only | depth7_high_capacity | 0.06149 | 0.01355 | 0.123 | 0.05235 | 0.1023 | 0.4444 | 0.5142 | 0.001042 | 0.0006945 | 0.009921 | 0.003181 | 1.213e-05 | 7.606e-06 | 0.0005763 | 8.709e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 4 | mixture_extremes | oracle_s_only | baseline | 0.06901 | 0.01307 | 0.1306 | 0.05607 | 0.1032 | 0.4525 | 0.5212 | 0.001733 | 0.0002336 | 0.01177 | 0.002115 | 1.519e-05 | 1.612e-06 | 0.0007401 | 2.425e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 4 | mixture_extremes | oracle_s_only | depth4_light | 0.07507 | 0.01463 | 0.1376 | 0.05915 | 0.1167 | 0.4476 | 0.5234 | 0.0018 | 5.95e-05 | 0.0131 | 0.002819 | 2.817e-05 | 2.197e-07 | 0.0007956 | 4.513e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 4 | mixture_extremes | oracle_s_only | depth7_high_capacity | 0.07286 | 0.01014 | 0.1397 | 0.06289 | 0.1209 | 0.4444 | 0.5229 | 0.001042 | 0.0009853 | 0.009921 | 0.00415 | 1.213e-05 | 4.889e-06 | 0.0005763 | 6.889e-05 |  |  |  |  | 6 | 1 | 0.6097 | 1 | 6 |
| 4 | shift_naive | oracle_coords_only | baseline | 0.08726 | 0.02632 | 0.236 | 0.1077 | 0.06986 | 0.5002 | 0.7362 | 0.008218 | 0.002954 | 0.02425 | 0.005172 | 0.0001392 | 2.056e-05 | 0.002483 | 9.634e-05 |  |  |  |  | 6 | 0.8333 | 0.4365 | 0.9699 | 6 |
| 4 | shift_naive | oracle_coords_only | depth4_light | 0.09601 | 0.02395 | 0.2297 | 0.09912 | 0.08089 | 0.5331 | 0.7627 | 0.007486 | 0.001931 | 0.02663 | 0.003725 | 9.126e-05 | 1.402e-05 | 0.003379 | 0.0001166 |  |  |  |  | 6 | 0.8333 | 0.4365 | 0.9699 | 6 |

### Heatmap: Δ PRAUC median (level × regime)

![](plots/delta_heatmap.png)

## Highest Δ level × regime combos

| level | regime_family | oracle_mode | xgb_config_id | delta_prauc_median | delta_prauc_p90 | delta_prauc_p10 | delta_prauc_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | shift_preserve | oracle_s_only | depth4_light | 0.4462 | 0.449 | 0.4387 | 0.005426 |
| 5 | shift_preserve | oracle_s_only | baseline | 0.4248 | 0.4336 | 0.414 | 0.01001 |
| 5 | shift_preserve | oracle_s_only | depth7_high_capacity | 0.423 | 0.4262 | 0.4194 | 0.003467 |
| 5 | shift_preserve | oracle_coords_only | depth4_light | 0.4058 | 0.4068 | 0.3926 | 0.008104 |
| 5 | shift_preserve | oracle_coords_only | baseline | 0.3947 | 0.3954 | 0.3907 | 0.002554 |
| 5 | shift_preserve | oracle_coords_only | depth7_high_capacity | 0.3766 | 0.3784 | 0.3545 | 0.01358 |
| 4 | shift_preserve | oracle_s_only | depth4_light | 0.2317 | 0.7373 | 0.172 | 0.2621 |
| 4 | shift_preserve | oracle_coords_only | depth4_light | 0.2306 | 0.7347 | 0.1625 | 0.2624 |
| 4 | shift_preserve | oracle_s_only | depth7_high_capacity | 0.2133 | 0.679 | 0.1543 | 0.2373 |
| 1 | shift_preserve | oracle_coords_only | depth7_high_capacity | 0.2125 | 0.3298 | 0.07558 | 0.1186 |
| 1 | shift_preserve | oracle_s_only | depth7_high_capacity | 0.2123 | 0.3233 | 0.09533 | 0.1066 |
| 3 | shift_preserve | oracle_s_only | depth7_high_capacity | 0.2097 | 0.2557 | 0.1592 | 0.04216 |
| 4 | shift_preserve | oracle_coords_only | depth7_high_capacity | 0.2091 | 0.6686 | 0.1395 | 0.2368 |
| 1 | shift_preserve | oracle_s_only | baseline | 0.203 | 0.3289 | 0.08896 | 0.1143 |
| 4 | shift_preserve | oracle_s_only | baseline | 0.1958 | 0.7101 | 0.1581 | 0.2546 |
| 1 | shift_preserve | oracle_s_only | depth4_light | 0.1944 | 0.3372 | 0.05516 | 0.1287 |
| 1 | shift_preserve | oracle_coords_only | depth4_light | 0.19 | 0.3384 | 0.04839 | 0.1315 |
| 1 | shift_preserve | oracle_coords_only | baseline | 0.1899 | 0.3319 | 0.07758 | 0.1182 |
| 4 | shift_preserve | oracle_coords_only | baseline | 0.1897 | 0.6985 | 0.1525 | 0.2517 |
| 3 | shift_preserve | oracle_s_only | depth4_light | 0.1845 | 0.2212 | 0.1486 | 0.03298 |
| 3 | shift_preserve | oracle_s_only | baseline | 0.1809 | 0.23 | 0.1399 | 0.04197 |
| 3 | shift_preserve | oracle_coords_only | baseline | 0.1782 | 0.2162 | 0.1365 | 0.0354 |
| 3 | shift_preserve | oracle_coords_only | depth7_high_capacity | 0.1766 | 0.2122 | 0.1497 | 0.02847 |
| 3 | shift_preserve | oracle_coords_only | depth4_light | 0.1719 | 0.2075 | 0.146 | 0.02867 |
| 4 | shift_naive | oracle_s_only | depth7_high_capacity | 0.1003 | 0.2009 | 0.04964 | 0.07775 |
| 4 | shift_naive | oracle_s_only | depth4_light | 0.0987 | 0.2029 | 0.05267 | 0.07417 |
| 2 | shift_preserve | oracle_coords_only | depth7_high_capacity | 0.09636 | 0.2372 | 0.04573 | 0.08213 |
| 4 | shift_naive | oracle_coords_only | depth4_light | 0.09601 | 0.2297 | 0.02395 | 0.09912 |
| 2 | shift_preserve | oracle_coords_only | baseline | 0.09575 | 0.2356 | 0.05578 | 0.0776 |
| 4 | shift_naive | oracle_s_only | baseline | 0.09383 | 0.2307 | 0.05282 | 0.09627 |
| 2 | shift_preserve | oracle_coords_only | depth4_light | 0.09185 | 0.2431 | 0.0595 | 0.08165 |
| 2 | shift_preserve | oracle_s_only | depth7_high_capacity | 0.09124 | 0.2466 | 0.04852 | 0.08577 |
| 4 | shift_naive | oracle_coords_only | baseline | 0.08726 | 0.236 | 0.02632 | 0.1077 |
| 2 | shift_preserve | oracle_s_only | depth4_light | 0.08613 | 0.2449 | 0.04523 | 0.08899 |
| 4 | shift_naive | oracle_coords_only | depth7_high_capacity | 0.08441 | 0.222 | 0.023 | 0.09829 |
| 2 | shift_preserve | oracle_s_only | baseline | 0.08291 | 0.2577 | 0.0511 | 0.09261 |
| 5 | shift_naive | oracle_s_only | depth4_light | 0.07807 | 0.1028 | 0.04102 | 0.03174 |
| 4 | mixture_extremes | oracle_s_only | depth4_light | 0.07507 | 0.1376 | 0.01463 | 0.05915 |
| 4 | mixture_extremes | oracle_s_only | depth7_high_capacity | 0.07286 | 0.1397 | 0.01014 | 0.06289 |
| 4 | mixture_extremes | oracle_s_only | baseline | 0.06901 | 0.1306 | 0.01307 | 0.05607 |

## Diagnostics (median)

| oracle_mode | xgb_config_id | invariance_raw | invariance_oracle | invariance_improvement | iso_raw | iso_oracle | iso_improvement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_coords_only | baseline | 0.01129 | 0.002968 | 0.006892 | 0.0002962 | 3.235e-05 | 0.0001854 |
| oracle_coords_only | depth4_light | 0.01114 | 0.002092 | 0.007514 | 0.0002763 | 1.611e-05 | 0.000185 |
| oracle_coords_only | depth7_high_capacity | 0.01068 | 0.00341 | 0.005372 | 0.0002864 | 3.906e-05 | 0.0001645 |
| oracle_s_only | baseline | 0.01129 | 0.002241 | 0.007732 | 0.0002962 | 1.727e-05 | 0.0002031 |
| oracle_s_only | depth4_light | 0.01114 | 0.001423 | 0.008405 | 0.0002763 | 7.604e-06 | 0.0002246 |
| oracle_s_only | depth7_high_capacity | 0.01068 | 0.003059 | 0.006136 | 0.0002864 | 2.817e-05 | 0.0001784 |
## Largest Δ runs

| task_id | level | regime_id | seed | n | oracle_mode | xgb_config_id | prauc_raw | prauc_oracle | delta_prauc | invariance_raw | invariance_oracle | invariance_improvement | dominance_train_mean | dominance_test_mean | shift_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| l4_product_x_product | 4 | shift_preserve_c5 | 0 | 30000 | oracle_s_only | depth4_light | 0.07868 | 0.9141 | 0.8354 | 0.1194 | 0.001359 | 0.118 |  |  | preserve |
| l4_product_x_product | 4 | shift_preserve_c5 | 0 | 30000 | oracle_coords_only | depth4_light | 0.07868 | 0.9067 | 0.828 | 0.1194 | 0.001149 | 0.1182 |  |  | preserve |
| l4_product_x_product | 4 | shift_preserve_c5 | 0 | 30000 | oracle_s_only | baseline | 0.1439 | 0.9143 | 0.7704 | 0.05098 | 0.005241 | 0.04574 |  |  | preserve |
| l4_product_x_product | 4 | shift_preserve_c5 | 0 | 30000 | oracle_coords_only | baseline | 0.1439 | 0.9028 | 0.7589 | 0.05098 | 0.003601 | 0.04738 |  |  | preserve |
| l4_product_x_product | 4 | shift_preserve_c5 | 0 | 30000 | oracle_s_only | depth7_high_capacity | 0.1923 | 0.9123 | 0.72 | 0.03492 | 0.005845 | 0.02908 |  |  | preserve |
| l4_product_x_product | 4 | shift_preserve_c5 | 0 | 30000 | oracle_coords_only | depth7_high_capacity | 0.1923 | 0.9001 | 0.7078 | 0.03492 | 0.003068 | 0.03185 |  |  | preserve |
| l4_product_x_product | 4 | shift_preserve_c5 | 2 | 30000 | oracle_s_only | baseline | 0.2609 | 0.9108 | 0.6499 | 0.03491 | 0.001925 | 0.03299 |  |  | preserve |
| l4_product_x_product | 4 | shift_preserve_c5 | 2 | 30000 | oracle_coords_only | depth4_light | 0.2695 | 0.9108 | 0.6414 | 0.04145 | 0.002106 | 0.03934 |  |  | preserve |
| l4_product_x_product | 4 | shift_preserve_c5 | 2 | 30000 | oracle_s_only | depth4_light | 0.2695 | 0.9087 | 0.6392 | 0.04145 | 0.002616 | 0.03883 |  |  | preserve |
| l4_product_x_product | 4 | shift_preserve_c5 | 2 | 30000 | oracle_s_only | depth7_high_capacity | 0.2663 | 0.9044 | 0.638 | 0.02197 | 0.001939 | 0.02003 |  |  | preserve |
| l4_product_x_product | 4 | shift_preserve_c5 | 2 | 30000 | oracle_coords_only | baseline | 0.2609 | 0.8989 | 0.638 | 0.03491 | 0.004625 | 0.03029 |  |  | preserve |
| l4_product_x_product | 4 | shift_preserve_c5 | 2 | 30000 | oracle_coords_only | depth7_high_capacity | 0.2663 | 0.8957 | 0.6293 | 0.02197 | 0.002557 | 0.01941 |  |  | preserve |
| l5_ratio_x_product | 5 | shift_preserve_c5 | 2 | 30000 | oracle_s_only | depth4_light | 0.05109 | 0.5008 | 0.4497 | 0.02471 | 0.002171 | 0.02254 |  |  | preserve |
| l5_ratio_x_product | 5 | shift_preserve_c5 | 0 | 30000 | oracle_s_only | depth4_light | 0.04431 | 0.4905 | 0.4462 | 0.02686 | 0.001065 | 0.02579 |  |  | preserve |
| l5_ratio_x_product | 5 | shift_preserve_c5 | 1 | 30000 | oracle_s_only | depth4_light | 0.05705 | 0.4939 | 0.4368 | 0.02059 | 0.002664 | 0.01793 |  |  | preserve |
| l5_ratio_x_product | 5 | shift_preserve_c5 | 2 | 30000 | oracle_s_only | baseline | 0.05144 | 0.4873 | 0.4358 | 0.0178 | 0.002357 | 0.01544 |  |  | preserve |
| l5_ratio_x_product | 5 | shift_preserve_c5 | 2 | 30000 | oracle_s_only | depth7_high_capacity | 0.05138 | 0.4783 | 0.4269 | 0.01109 | 0.004094 | 0.007 |  |  | preserve |
| l5_ratio_x_product | 5 | shift_preserve_c5 | 1 | 30000 | oracle_s_only | baseline | 0.06571 | 0.4905 | 0.4248 | 0.01326 | 0.001378 | 0.01188 |  |  | preserve |
| l5_ratio_x_product | 5 | shift_preserve_c5 | 0 | 30000 | oracle_s_only | depth7_high_capacity | 0.04474 | 0.4678 | 0.423 | 0.01676 | 0.003256 | 0.01351 |  |  | preserve |
| l5_ratio_x_product | 5 | shift_preserve_c5 | 1 | 30000 | oracle_s_only | depth7_high_capacity | 0.06462 | 0.4831 | 0.4185 | 0.01124 | 0.00224 | 0.009004 |  |  | preserve |
| l5_ratio_x_product | 5 | shift_preserve_c5 | 0 | 30000 | oracle_s_only | baseline | 0.04826 | 0.4596 | 0.4114 | 0.01932 | 0.004374 | 0.01495 |  |  | preserve |
| l5_ratio_x_product | 5 | shift_preserve_c5 | 0 | 30000 | oracle_coords_only | depth4_light | 0.04431 | 0.4514 | 0.4071 | 0.02686 | 0.0028 | 0.02406 |  |  | preserve |
| l5_ratio_x_product | 5 | shift_preserve_c5 | 1 | 30000 | oracle_coords_only | depth4_light | 0.05705 | 0.4629 | 0.4058 | 0.02059 | 0.002115 | 0.01848 |  |  | preserve |
| l5_ratio_x_product | 5 | shift_preserve_c5 | 0 | 30000 | oracle_coords_only | baseline | 0.04826 | 0.4438 | 0.3956 | 0.01932 | 0.00365 | 0.01567 |  |  | preserve |
| l5_ratio_x_product | 5 | shift_preserve_c5 | 2 | 30000 | oracle_coords_only | baseline | 0.05144 | 0.4461 | 0.3947 | 0.0178 | 0.002816 | 0.01498 |  |  | preserve |
## Smallest Δ runs

| task_id | level | regime_id | seed | n | oracle_mode | xgb_config_id | prauc_raw | prauc_oracle | delta_prauc | invariance_raw | invariance_oracle | invariance_improvement | dominance_train_mean | dominance_test_mean | shift_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| l2_ratio_of_sums | 2 | shift_naive_c5 | 0 | 30000 | oracle_s_only | depth4_light | 0.4568 | 0.3835 | -0.07324 | 0.01662 | 0.006024 | 0.0106 | 0.6305 | 0.723 | naive |
| l2_ratio_of_sums | 2 | shift_naive_c5 | 0 | 30000 | oracle_coords_only | baseline | 0.4556 | 0.3858 | -0.06985 | 0.01681 | 0.005772 | 0.01104 | 0.6305 | 0.723 | naive |
| l1_product | 1 | shift_naive_c5 | 2 | 30000 | oracle_coords_only | depth7_high_capacity | 0.8961 | 0.8348 | -0.06134 | 0.03718 | 0.00517 | 0.03201 |  |  | naive |
| l2_ratio_of_sums | 2 | shift_naive_c5 | 0 | 30000 | oracle_coords_only | depth4_light | 0.4568 | 0.4006 | -0.05616 | 0.01662 | 0.005845 | 0.01077 | 0.6305 | 0.723 | naive |
| l2_ratio_of_sums | 2 | shift_naive_c5 | 0 | 30000 | oracle_s_only | baseline | 0.4556 | 0.4057 | -0.04997 | 0.01681 | 0.005218 | 0.0116 | 0.6305 | 0.723 | naive |
| l1_ratio | 1 | shift_naive_c5 | 0 | 30000 | oracle_coords_only | depth4_light | 0.2784 | 0.2399 | -0.03853 | 0.0133 | 1.382e-05 | 0.01328 |  |  | naive |
| l1_product | 1 | ln_sigma0.7_rho0.5 | 0 | 30000 | oracle_s_only | depth7_high_capacity | 0.451 | 0.4195 | -0.03154 | 0.01319 | 0.0007891 | 0.0124 |  |  | none |
| l7_gated_ratio_vs_product | 7 | shift_naive_c5 | 2 | 30000 | oracle_s_only | depth4_light | 0.9296 | 0.8988 | -0.03084 | 0.03076 | 0.05823 | -0.02747 |  |  | naive |
| l1_product | 1 | shift_naive_c5 | 2 | 30000 | oracle_s_only | baseline | 0.909 | 0.879 | -0.03 | 0.04911 | 0.01179 | 0.03732 |  |  | naive |
| l1_ratio | 1 | ln_sigma0.7_rho0.5 | 0 | 30000 | oracle_coords_only | depth7_high_capacity | 0.2322 | 0.2035 | -0.0287 | 0.01039 | 2.138e-05 | 0.01037 |  |  | none |
| l7_gated_ratio_vs_product | 7 | shift_naive_c5 | 2 | 30000 | oracle_s_only | baseline | 0.9332 | 0.9069 | -0.02623 | 0.03721 | 0.05074 | -0.01353 |  |  | naive |
| l1_ratio | 1 | shift_naive_c5 | 0 | 30000 | oracle_s_only | depth7_high_capacity | 0.2318 | 0.2059 | -0.02594 | 0.0161 | 0.004515 | 0.01158 |  |  | naive |
| l2_product_of_sums | 2 | ln_sigma0.7_rho0.5 | 0 | 30000 | oracle_s_only | depth4_light | 0.419 | 0.3954 | -0.0236 | 0.01194 | 0.001146 | 0.0108 | 0.6303 | 0.6301 | none |
| l1_ratio | 1 | ln_sigma0.7_rho0.5 | 1 | 30000 | oracle_s_only | depth7_high_capacity | 0.2516 | 0.2294 | -0.0222 | 0.01101 | 0.001574 | 0.009433 |  |  | none |
| l1_product | 1 | ln_sigma0.3_rho0.0 | 2 | 30000 | oracle_s_only | depth4_light | 0.1489 | 0.1272 | -0.02169 | 0.008401 | 1.29e-05 | 0.008388 |  |  | none |
| l1_ratio | 1 | shift_naive_c5 | 2 | 30000 | oracle_s_only | depth7_high_capacity | 0.1905 | 0.1695 | -0.02103 | 0.01798 | 0.003097 | 0.01489 |  |  | naive |
| l1_ratio | 1 | shift_naive_c5 | 2 | 30000 | oracle_coords_only | baseline | 0.2082 | 0.1877 | -0.02047 | 0.01371 | 0.001779 | 0.01193 |  |  | naive |
| l1_ratio | 1 | ln_sigma1.2_rho0.9 | 1 | 30000 | oracle_coords_only | baseline | 0.1568 | 0.1369 | -0.01989 | 0.008535 | 0.002447 | 0.006088 |  |  | none |
| l1_ratio | 1 | ln_sigma1.2_rho0.9 | 1 | 30000 | oracle_s_only | baseline | 0.1568 | 0.1379 | -0.01889 | 0.008535 | 5.603e-05 | 0.008479 |  |  | none |
| l7_gated_ratio_vs_product | 7 | ln_sigma0.7_rho0.5 | 0 | 30000 | oracle_s_only | depth4_light | 0.3209 | 0.3021 | -0.01881 | 0.0204 | 0.002922 | 0.01748 |  |  | none |
| l1_product | 1 | shift_naive_c5 | 2 | 30000 | oracle_s_only | depth4_light | 0.9153 | 0.8967 | -0.01864 | 0.05125 | 0.0161 | 0.03515 |  |  | naive |
| l6_nonmonotone_product | 6 | ln_sigma0.7_rho0.5 | 0 | 30000 | oracle_coords_only | baseline | 0.07749 | 0.05929 | -0.0182 | 0.005696 | 0.0002637 | 0.005433 |  |  | none |
| l1_ratio | 1 | ln_sigma0.3_rho0.0 | 1 | 30000 | oracle_coords_only | baseline | 0.1207 | 0.1033 | -0.01742 | 0.01444 | 1.144e-05 | 0.01443 |  |  | none |
| l1_product | 1 | ln_sigma0.7_rho0.5 | 2 | 30000 | oracle_s_only | depth7_high_capacity | 0.5195 | 0.5023 | -0.01718 | 0.0137 | 0.004644 | 0.009058 |  |  | none |
| l3_product_diff | 3 | ln_sigma1.2_rho0.9 | 2 | 30000 | oracle_coords_only | baseline | 0.2018 | 0.1856 | -0.01626 | 0.01639 | 0.009141 | 0.007252 |  |  | none |
## Plots

### Δ PRAUC distribution

![](plots/delta_prauc_boxplot.png)

### Δ vs n

![](plots/delta_vs_n.png)

### Invariance vs Δ

![](plots/delta_vs_invariance.png)

### Dominance vs Δ

![](plots/delta_vs_dominance.png)

### Per-level Δ vs invariance

![](plots/delta_vs_invariance_level.png)

### Per-level Δ vs dominance

![](plots/delta_vs_dominance_level.png)

