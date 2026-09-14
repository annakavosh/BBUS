# Optional analysis modules

The analysis targets are one-way consumers of immutable update records. They
are never linked by `belief_update::core`, `belief_update::resolver`, or
`belief_update::update`.

## Local sensitivity

`belief_update::local_sensitivity` reports the three analytic partial
derivatives of the binary posterior. With prior `p`, likelihoods `a = P(E|H)`
and `b = P(E|not H)`, and evidence mass `z = pa + (1-p)b`:

```text
d posterior / d p = ab / z^2
d posterior / d a = p(1-p)b / z^2
d posterior / d b = -p(1-p)a / z^2
```

The implementation scales the likelihoods before evaluating these expressions
to reduce avoidable underflow and retains the posterior from the supplied
immutable result. It uses no finite differencing, autodiff, or optimizer.

## Model versus explicit comparison

`belief_update::posterior_comparison` requires an `UpdateResult` whose original
inputs contain both candidates. It independently sends each candidate through
the existing update pipeline with the recorded prior, configuration, evaluation
time, and algorithm. The result owns both derived update records.

The reported difference has one fixed sign convention:

```text
difference = explicit posterior - model posterior
```

The module performs no formatting. A caller may render a result such as:

```text
Model posterior:       0.73
Explicit posterior:    0.64
Difference:           -0.09
```

## Threshold analysis

`belief_update::threshold_analysis` reports the prior and posterior positions,
the signed posterior margin, whether the threshold is met, and any upward or
downward transition. For a non-certain prior, the likelihood ratio that places
the posterior exactly at threshold `t` is calculated analytically:

```text
LR at threshold = t(1-p) / ((1-t)p)
```

Thresholds zero and one produce the corresponding limiting ratios but are not
marked reachable from a non-certain prior because validated likelihood ratios
must be finite and strictly positive. A certain prior can reach only the same
endpoint; in that case the required ratio is not unique and is left empty. No
range, tolerance, or threshold is selected automatically.
