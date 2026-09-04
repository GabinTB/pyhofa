# pyhofa — Higher-Order Factor Analysis for Python

**pyhofa** is a Python implementation of **higher-order multi-cumulant factor analysis
(HOFA/HFA)**, based on Guanglin Huang's [`hofa`](https://github.com/GuanglinHuang/hofa) R package
and the methodology in Huang, Lu and Boudt, *Estimation of factors using higher-order
multi-cumulants in weak factor models*.

---

## What's inside

- Second-order factor-number selection: ER, GR, Bai-Ng IC3/PC3, BIC3, Onatski, ACT
- Third- and fourth-order GER/GGR selectors, plus JJR simulation thresholds
- PCA and projected PCA
- Bai-Li ML, QML, ML-GLS, ML-ITE and ML-EM
- Second- and third-order GMM estimators
- Third- and fourth-order HFA alternating least squares
- Adaptive HFA, with out-of-sample projection for rolling estimation
- Synthetic DGP1/DGP2 simulations with skewed generalized-t innovations
- PC and IC higher-moment portfolio optimization
- Moment/cumulant utilities, covariance shrinkage, and simulation diagnostics

---

## Quick example

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

See [Quick Start](guide/quickstart.md) for factor selection, adaptive HFA and rolling estimation.

---

## Installation

```bash
pip install pyhofa
```

See [Installation](guide/installation.md) for building from source with `uv`.

---

## Relationship to the R `hofa` package

pyhofa ports the R [`hofa`](https://github.com/GuanglinHuang/hofa) package to Python. It is not a
blind transliteration — several points in the R source are internally inconsistent or unsafe for
production use, and the Python implementation follows the mathematical intent instead. See
[R Package Mapping & Corrections](guide/r-mapping.md) for the full function mapping and the list
of intentional differences.

## License and attribution

The upstream R package declares GPL-2. This Python implementation is therefore distributed under
**GPL-2.0-only** as a derivative implementation.

Primary upstream project:

- Guanglin Huang, [`GuanglinHuang/hofa`](https://github.com/GuanglinHuang/hofa)
- Huang, G., Lu, W., and Boudt, K., *Estimation of factors using higher-order multi-cumulants in
  weak factor models*.

This Python port is not an official release of the upstream authors.
