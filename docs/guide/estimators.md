# Estimators

## PCA and Projected PCA

```python
from pyhofa import m2_pca

pca = m2_pca(X, r=2)
ppca = m2_pca(X, r=3, method="P-PCA", characteristics=characteristics, sieve_terms=6)
```

`characteristics` has one row per panel variable; the projected-PCA path uses additive cubic
B-spline bases followed by an orthogonal least-squares projection.

## Maximum likelihood and QML

```python
from pyhofa import m2_mle

ml = m2_mle(X, r=2, method="ML")
em = m2_mle(X, r=2, method="ML-EM", ar_order=1)
```

Supported `method` values include `ML`, `QML`, `ML-GLS`, `ML-ITE` and `ML-EM`. ML-EM is implemented
as a state-space model with state `[f_t, f_{t-1}]`, filtered and smoothed with a Kalman filter plus
Rauch-Tung-Striebel smoother.

## GMM

```python
from pyhofa import m2_gmm, m3_gmm

gmm2 = m2_gmm(X, r=2)
gmm3 = m3_gmm(X, r=2)
```

## Higher-order ALS

```python
from pyhofa import m3_als, m4_als

hfa3 = m3_als(X, rh=2, rg=0)
hfa4 = m4_als(X, rh=2, rg=0)
```

Both use `n**4`-consistent scaling throughout and align eigenvector signs before evaluating the
ALS stopping criterion, so a sign flip between iterations does not report false non-convergence.

## API reference

::: pyhofa.estimators
