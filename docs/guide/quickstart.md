# Quick Start

## PCA and HFA

```python
import pyhofa

sim = pyhofa.dgp2(
    n=100,
    t=300,
    k=2,
    factor_lambda=[0.8, 0.8],
    factor_p=[1.0, 1.0],
    factor_ar=[0.5, 0.2],
    random_state=0,
)

pca = pyhofa.m2_pca(sim.X, r=2)
hfa3 = pyhofa.m3_als(sim.X, rh=2, rg=0)

print(pyhofa.trace_ratio(pca.factors, sim.factors))
print(pyhofa.trace_ratio(hfa3.factors, sim.factors))
```

## Factor-number selection

```python
from pyhofa import m2_select, m3_select, m4_select

r_er = m2_select(sim.X, method="ER", rmax=8)
r_ger3 = m3_select(sim.X, method="GER3", rmax=8)
r_ger4 = m4_select(sim.X, method="GER4", rmax=8)

print(r_er.n_factors)
print(r_ger3.n_nongaussian, r_ger3.n_gaussian)
print(r_ger4.n_nongaussian, r_ger4.n_gaussian)
```

The result objects also expose R-style compatibility fields:

```python
r_er.R
r_ger3.Rh
r_ger3.Rg
```

## Adaptive HFA

```python
from pyhofa import adaptive_hfa

result = adaptive_hfa(sim.X, r=2, max_order=4)
print(result.cumulant_order_f)
print(result.cumulant_order_u)
print(result.factor_contribution_ratios)
```

For out-of-sample evaluation, fit on the training block and reuse its frozen preprocessing
statistics and loadings — see [Adaptive HFA & Rolling Estimation](adaptive.md) for the full
pattern and why fitting on the full sample and slicing afterwards leaks look-ahead information.

## ML and GMM

```python
from pyhofa import m2_gmm, m2_mle, m3_gmm

ml = m2_mle(sim.X, r=2, method="ML")
em = m2_mle(sim.X, r=2, method="ML-EM", ar_order=1)
gmm2 = m2_gmm(sim.X, r=2)
gmm3 = m3_gmm(sim.X, r=2)
```

## Projected PCA

`characteristics` has one row per panel variable. The Python implementation uses additive cubic
B-spline bases for the sieve projection.

```python
ppca = pyhofa.m2_pca(
    X,
    r=3,
    method="P-PCA",
    characteristics=characteristics,
    sieve_terms=6,
)
```
