# Mathematically correct but misleading posteriors

The library can reject structural or mathematical invalidity. It cannot infer
whether a valid probability model is appropriate for the real world. Inputs are
never clipped, reweighted, substituted, or otherwise corrected.

| Situation | Classification | Behavior |
|---|---|---|
| Bad likelihood model | `reject` or `record` | Invalid probabilities, ratios, or provenance are rejected. A numerically valid but poorly calibrated model is recorded unchanged because the library has no ground truth. |
| Stale model | `warn`, `reject`, or `record` | An explicit `FreshnessRule` determines warning versus rejection. Without one, the supplied timestamp is recorded and no freshness judgment is invented. |
| Wrong likelihood assumptions | `record` | The resolved likelihoods and their provenance are retained. Semantic correctness cannot be derived from two scalars. |
| Incorrect override | `reject` or `record` | An invalid override is rejected and never falls back to the model. A valid but semantically mistaken override wins by the documented precedence rule and both candidates remain recorded. |
| Low-confidence evidence | `record` | Confidence is validated and retained but never used to down-weight the likelihoods. No confidence threshold is chosen automatically. |
| Missing provenance | `reject` | `LikelihoodInput` requires typed provenance; source IDs and versions cannot be empty, and malformed enum values are rejected by validation. |
| Extreme likelihood ratios | `reject` or `record` | Non-finite and non-positive ratios are rejected by orchestration. Finite extreme ratios are calculated and recorded unchanged. |
| Dependent evidence treated as independent | `record` | Each update records provenance, but the stateless library cannot infer dependence across calls. Dependency control belongs in the upstream likelihood model or evidence pipeline. |

The only warning currently produced on the update path is a caller-configured
freshness warning. Adding automatic warning thresholds for confidence,
likelihood-ratio magnitude, or repeated source IDs would embed domain policy and
could misclassify valid inputs. Such diagnostics belong in optional analysis or
audit tooling with explicit caller-supplied policies.
