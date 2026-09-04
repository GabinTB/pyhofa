# Simulation

pyhofa includes a native sampler for the variance-adjusted skewed generalized-t parameterization
used by the R `hofa` package.

```python
import pyhofa

x = pyhofa.sgt_rvs(
    10_000,
    sigma=1.0,
    lam=0.8,
    p=1.0,
    q=float("inf"),
    rng=0,
)
```

For `lam=0`, `p=2`, `q=inf`, the distribution reduces to a Gaussian with the requested variance.

## DGP1 / DGP2

```python
sim = pyhofa.dgp2(
    n=100,
    t=300,
    k=2,
    factor_lambda=[0.8, 0.8],
    factor_p=[1.0, 1.0],
    factor_ar=[0.5, 0.2],
    random_state=0,
)
```

`dgp2` additionally accepts `loading_strength`, which is useful for direct strong/weak-factor
experiments without rewriting the DGP.

## API reference

::: pyhofa.simulation

::: pyhofa.moments
