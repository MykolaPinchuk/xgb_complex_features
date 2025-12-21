# Feedback memo for paper-writing agent

- Paper: `paper_draft/2025-12-14_110315/xgb_invariant_features_updated_v2_dec16.pdf`
- Scope: incorporate recent discussion on (i) level framing and redundancy, (ii) why distribution shift is central, (iii) practitioner accessibility, (iv) clearer statement of what the benchmark measures.

## 0) Executive summary (what to change first)

1. **Correctness pass**: align task definitions (Appendix A + Sec 3.1) with what the code actually runs. Several level definitions in the PDF do not match the implemented tasks.
2. **Framing pass**: stop presenting “Level 1–7” as a strict complexity ordering. Treat levels as motif labels. Present Level 4 and Level 5 together as an interaction family.
3. **Motivation pass**: explain why distribution shift is necessary for the feature-engineering claim. Add a no-shift baseline view explicitly. The paper already contains the evidence (e.g., regime-family table), but it is not foregrounded for practitioners.
4. **Practitioner pass**: add plain-language explanation of distribution shift, invariance, and what to do in practice.

## 1) High priority: task-definition mismatches (paper vs code)

The working paper’s Appendix A and parts of Sec 3.1 currently describe tasks that are not the same as the benchmark implementation. This is a scientific validity issue because readers will interpret results under the written definitions.

### 1.1 What the code actually uses (high level)

The implemented benchmark works in *log-coordinates* (with an additive stabilizer), then builds the signal from those coordinates. In words:

- Define raw features $x \in \mathbb{R}^d_{>0}$.
- Construct coordinates that are **log-ratios** or **log-products**, e.g.
  - ratio coordinate: $u=\log\!\left(\frac{\sum_{i\in A} x_i}{\sum_{j\in B} x_j + \epsilon}\right)$,
  - product coordinate: $u=\log\!\left(\prod_{i\in A} x_i \prod_{j\in B} x_j + \epsilon\right)$,
  where $\epsilon$ is chosen as a small fraction of a median scale.
- Build the final signal $s$ from these coordinates by a small set of motifs: identity, differences, products, non-monotone transforms, gating.

### 1.2 Concrete mismatches to fix

- **Level 3**: the PDF describes a “ratio of differences” like $(x_0-x_1)/(x_2-x_3+\epsilon)$. The code uses a *difference of coordinates* (e.g., difference of log-ratios or difference of log-products).
- **Level 4 / Level 5**: the PDF describes “ratio of ratios” like $(x_0/x_1)/(x_2/x_3)$ and “mix” like $(x_0/x_1)/(x_2x_3)$. The code uses *coordinate interactions* of the form $s=u_1u_2$ where $u_k$ are log-coordinates (log-ratio, log-product). This matters because the “ratio-of-ratios” algebra collapses into a single monomial ratio, which would indeed undermine the claimed compositional complexity.
- **Preserve shift language**: the PDF says the preserve shift “keeps the ratio fixed”. With an additive $\epsilon$ in the denominator, the ratio coordinate is only approximately invariant unless $\epsilon=0$ or the shift scales $\epsilon$ as well. This can be handled with a short caveat.

Action: update Sec 3.1 and Appendix A so the math matches the implementation. If the goal is to keep the paper’s math as simple ratios/products, then the code and results need to be regenerated under those definitions. The easier path is to update the paper to match the code.

## 2) Level redundancy and updated framing (what we discussed)

### 2.1 The main point

Several “levels” share the same abstract signal form once one writes $z_i=\log x_i$ and ignores stabilizers. This does not make them redundant as learning problems, because the benchmark varies:

- the joint distribution (tails, correlation, mixtures),
- the feature view (raw-only vs oracle coordinates vs oracle signal),
- and targeted shifts that preserve the intended coordinate but destroy raw-scale shortcuts.

### 2.2 Specific clarification for Levels 4 and 5

The paper should **not** claim that Level 5 is “more compositional” than Level 4. In the code, both are interaction motifs of the form $s=u_1u_2$. The scientific distinction is the invariance type and induced geometry:

- ratio×ratio and product×product (Level 4 in the current labels),
- ratio×product (Level 5 in the current labels).

These are different *subtypes* within the same interaction family. They can behave differently under correlation and shift even though both are quadratic in $z$.

### 2.3 Proposed wording (drop-in replacement)

Replace language like “levels increase structural complexity” with something like:

> The benchmark assigns each task a level label that indexes a small set of functional motifs. The level index is a taxonomy. It is not a formal ordering by function-class complexity. In particular, Level 4 and Level 5 are both coordinate-interaction motifs of the form $s=u_1u_2$, but they differ in whether the interacting coordinates are ratio-type, product-type, or hybrid.

Optionally rename the narrative categories:

- “Coordinate motifs”: monotone coordinates, additive contrasts, interactions, non-monotone shapes, gating.
- “Aggregation tags”: sum-based vs non-sum-based.
- “Invariance tags”: ratio-type, product-type, hybrid.

## 3) Why distribution shift is central (addressing question 1)

The paper’s feature-engineering claim is about *robust recovery of the intended invariance*, not just in-distribution interpolation. Distribution shift is the tool that makes this identifiable.

### 3.1 What to explain explicitly

Without shift, a raw-only model can achieve good test PRAUC by exploiting distribution-specific cues that correlate with the latent coordinate on that particular regime. That does not mean it learned the coordinate. A preserve shift breaks those cues while leaving the latent signal (approximately) unchanged. Therefore, preserve shift is a targeted stress test for invariance learning.

### 3.2 What to add for practitioners

Add a short, concrete paragraph in the introduction:

> Distribution shift means the feature distribution at deployment differs from training. A common example is a change of units or scale. A ratio like debt-to-income should be invariant to multiplying both debt and income by the same constant. If the model learned the ratio, predictions should be stable. If it learned magnitude shortcuts, performance can collapse when units change.

### 3.3 “Do we need shift” and “what happens without shift”

Answer both explicitly in the main text:

- No-shift regimes (tail/correlation/mixture) often show smaller gaps.
- Preserve shift produces the largest gaps and is the cleanest evidence of shortcut learning.

The paper already contains regime-family summaries. Consider moving a compact version earlier (or add a small figure) so readers immediately see: “gap modest without shift; gap large under preserve shift”.

## 4) Practitioner accessibility (addressing question 2)

The current tone is readable, but it still feels like an academic robustness paper. Practical changes that help:

- Add a “When to care” box early (units change, inflation, sensor calibration, data pipeline normalization, vendor changes, geography/time drift).
- Add a “What to do” box: if a ratio/product is known and cheap, engineer it; otherwise test invariance with perturbations; consider augmentation or monotone/invariance constraints.
- Reduce jargon like “recover intended structure” by tying it to invariances practitioners already understand (unit invariance, scale invariance).
- Add one small end-to-end schematic of the benchmark pipeline: sample $x$, compute $u$, compute $s$, sample $y$, train raw vs oracle, evaluate, compute invariance diagnostics.

## 5) Clarify what the framework measures and why it matters (addressing question 3)

One sentence that should appear very early:

> The benchmark measures whether a strong tabular model trained on raw features learns the *invariant coordinate* that generates labels, or whether it relies on distribution-specific shortcuts that fail under targeted shifts.

Then make the measurement concrete:

- Primary estimand: $\Delta\mathrm{PRAUC}=\mathrm{PRAUC}_{oracle}-\mathrm{PRAUC}_{raw}$.
- Mechanistic evidence: invariance error + iso-coordinate variance.
- Stress test: preserve shift (plus naive shift as a control).

Also clarify what “oracle” is and is not:

- Oracle features are deterministic functions of $x$ used by the DGP.
- They are not the label.

## 6) Limitations and future work (make it less incremental)

Current limitations/future work read a bit generic. Suggest focusing on a few “credibility upgrades”:

- More seeds on a smaller subset (uncertainty).
- Independent sweeps of correlation vs tail-heaviness (remove confounding).
- Dataset size sweep for sample complexity.
- Baselines: raw-only on log(x), linear model on oracle coords, and a small neural net baseline.
- Fix and then evaluate Level 7 gating carefully (the current implementation likely needs adjustment before strong claims).

Also remove placeholders like “Appendix ??”.

