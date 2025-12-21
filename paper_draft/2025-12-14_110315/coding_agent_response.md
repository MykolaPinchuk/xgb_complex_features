# Coding agent response to feedback

This note responds to the feedback on `paper_draft/2025-12-14_110315/xgb_invariant_features_updated_v2_dec16.pdf`.

## 1. Why use distribution shift at all

The paper is not only asking whether oracle features increase in distribution test PRAUC. The paper is asking whether a raw feature model learns the intended invariant coordinate, or whether it relies on regime specific shortcuts.

In distribution performance alone cannot separate these two cases. A model can score well because magnitude and tail cues correlate with the latent coordinate under one regime. That does not mean the model learned the invariant structure.

The preserve shift is the identifiability device. It keeps the latent signal approximately fixed while removing scale cues. Large drops under preserve shift are evidence of non invariant learning rather than only insufficient capacity.

That said, the paper should answer the practical baseline question directly. It should state what the oracle gap looks like without explicit shift. It should show this early in the main text rather than only in appendices.

## 2. Make the paper more practitioner friendly

The paper should add a short plain language explanation of distribution shift and why it matters.

Concrete examples help. Units can change. Data pipelines can rescale variables. Inflation can change magnitudes. Sensor calibration can drift. Vendor and geography changes can shift feature distributions.

The paper should include a small box titled something like "When to care" and "What to do". It can recommend explicit ratio or product features when the invariance is known and cheap. It can also recommend invariance checks via targeted perturbations and augmentation when feature engineering is not obvious.

## 3. Clarify what the framework measures and why it matters

The paper should state a one sentence definition early.

The benchmark measures whether a strong tabular model trained on raw inputs learns the invariant coordinate that generates labels, or whether it uses shortcuts that fail under targeted shifts.

Then the paper should define the estimand and the mechanism checks.

The primary estimand is
$$
\Delta \mathrm{PRAUC} = \mathrm{PRAUC}_{oracle} - \mathrm{PRAUC}_{raw}.
$$

The mechanism checks are invariance error and iso coordinate variance. The preserve shift is the targeted stress test. The naive shift is a control that changes the latent coordinate distribution.

The paper should also state what the oracle mode is and is not. Oracle features are deterministic functions of $x$ used by the data generating process. They are not the label.

## 4. Improve limitations and future work

Limitations and future work should focus on credibility upgrades rather than small tweaks.

Examples include more seeds on a smaller subset, independent sweeps of correlation and tail heaviness, and a dataset size sweep for sample complexity. The paper should add a small set of baselines. Raw only on $\log(x)$ is one baseline. A linear model on oracle coordinates is another baseline. A small neural network baseline is another baseline.

The paper should also be careful about the level framing. Levels should be treated as motif labels rather than a strict complexity ladder. In particular, Level 4 and Level 5 are both interaction motifs of the form $s=u_1u_2$. They differ in invariance type rather than polynomial degree.
