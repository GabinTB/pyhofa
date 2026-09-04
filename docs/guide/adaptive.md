# Adaptive HFA & Rolling Estimation

## Adaptive HFA

```python
from pyhofa import adaptive_hfa

result = adaptive_hfa(X, r=2, max_order=4)
print(result.cumulant_order_f)
print(result.cumulant_order_u)
print(result.factor_contribution_ratios)
```

The estimator chooses between covariance, third-order and optionally fourth-order cumulant
estimators using factor contribution ratios (FCRs). Higher-order estimates use the split selected
by GER3 and GER4: the largest selected non-Gaussian count is retained and the remaining factors
are treated as Gaussian.

## Out-of-sample projection

Fit on the training block and reuse its frozen preprocessing statistics and loadings:

```python
from pyhofa import project

fit = adaptive_hfa(X_train, r=2, max_order=4, scale=True)
factors_test = project(
    X_test,
    fit.loadings,
    mean=fit.mean_,
    scale=fit.scale_,
)
```

!!! warning
    Do not fit on the full sample and then slice the returned factors. That uses test observations
    to estimate the column means, scales and loadings, causing look-ahead leakage.

## Rolling estimation

Fix the factor count, non-Gaussian/Gaussian split and selected cumulant order on an initial
training block; do not re-select them independently in every short rolling window. Refit each
window with that fixed estimator, align its loadings against the preceding fit, then project
observations with that window's frozen preprocessing statistics. For a third-order training
choice:

```python
from pyhofa import align_loadings, m3_als, project

reference_loadings = initial_fit.loadings

fit = m3_als(X_window, rh=fixed_rh, rg=fixed_rg)
aligned_loadings = align_loadings(reference_loadings, fit.loadings)
factors_window = project(
    X_window,
    aligned_loadings,
    mean=fit.mean_,
    scale=fit.scale_,
)
reference_loadings = aligned_loadings
```

The default alignment uses orthogonal Procrustes rotation and is the recommended choice for
chained rolling fits with weak or correlated factors. Set `allow_rotation=False` to use globally
optimal Hungarian matching on absolute column correlations followed by sign resolution when
factors should only be permuted and sign-flipped. Use `align_columns` only when column identities
are already fixed and the remaining indeterminacy is sign-only.

!!! warning
    Always recompute factor scores from the aligned loadings — permuting or rotating loadings
    changes the corresponding factor coordinates.

## API reference

::: pyhofa.adaptive

::: pyhofa.alignment

::: pyhofa.projection
