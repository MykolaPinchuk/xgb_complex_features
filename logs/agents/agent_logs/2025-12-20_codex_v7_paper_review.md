# Agent log: v7 paper review (release readiness)

- Timestamp: 2025-12-20T16:34:33-08:00
- Agent: Codex CLI
- Model: GPT-5.2
- Artifact reviewed: `paper_draft/2025-12-14_110315/xgb_invariant_features_updated_v7.pdf`

## High-level assessment

v7 is coherent, scoped, and already reads like a reasonable initial working paper. The framing (targeted preserve shifts as an identifiability / shortcut-stress device; oracle coordinates as “feature engineering realism”) is scientifically defensible and communicates a clear message with modest tables.

The main release blocker is implementation alignment: a few mathematical definitions (coordinates / stabilizer ϵ / gating) are currently inconsistent with the code in `src/xgb_complex_features/dgp/tasks.py`. These are easy text edits, but they matter for credibility.

## Low-effort fixes recommended before public release

1) **Make coordinate definitions match the implementation**
   - In code, ϵ is **per-coordinate**, not a single global constant within a dataset:
     - ratio: `eps = epsilon_rel * median(denom)` where `denom` is the denominator raw feature (or denominator sum for ratio-of-sums).
     - product: `eps = epsilon_rel * median(product)` (or median of the product-of-sums base).
   - In code, ratio coordinates use `log(num / (denom + eps))` (no `+eps` in the numerator).
   - Update Eq. (3) and the paragraph below it accordingly. Also update Appendix A examples so Level 1 ratio uses singleton sets `{0}/{1}` rather than `{0,1}/{2,3}` unless the experiment truly does otherwise.

2) **Clarify what Level 7 “gating” is in the implementation (or de-emphasize further)**
   - In code, the gate is a **raw feature** `g = x_4` thresholded at a quantile, not a gate coordinate `u_g = uR(...)`.
   - If keeping Level 7, adjust Appendix A to match (`g = x_4`), and keep the “provisional” label prominent (already done in main text).

3) **Avoid an unsupported “fits training data” claim**
   - Abstract currently says raw mode “often fits training data but fails under targeted shifts”. If training metrics are not shown, change to: “often performs well on static regimes” or similar.

4) **Add a one-sentence note about prevalence and PRAUC under shifts**
   - In the implementation, `beta0` is calibrated on the unshifted signal but labels are sampled on the shifted signal, so test prevalence can change under shift. Since PRAUC is prevalence-sensitive, emphasize that the main estimand is the within-regime paired gap `ΔPRAUC`.

5) **Remove or reframe the “Implementation checklist” appendix for a public draft**
   - Appendix E reads internal. Either drop it, or convert it into a standard “Limitations and future work” paragraph list without implementation phrasing.

## Optional small polish (also low-effort)

- Add 1–2 headline numbers in the Abstract (e.g., “under preserve shifts, median ΔPRAUC ≈ 0.15 in the baseline setting”).
- Add a short reproducibility note (“Code + configs: …; run: `python -m xgb_complex_features workflow --config configs/exp_default.yaml`”).

