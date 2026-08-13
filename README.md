# pyhofa

A Python implementation of **higher-order multi-cumulant factor analysis (HOFA/HFA)**, based on Guanglin Huang's [`hofa`](https://github.com/GuanglinHuang/hofa) R package and the methodology in Huang, Lu and Boudt, *Estimation of factors using higher-order multi-cumulants in weak factor models*.

The library covers the full public surface of the R project in Python form:

- second-order factor-number selection: ER, GR, Bai-Ng IC3/PC3, BIC3, Onatski, ACT;
- third- and fourth-order GER/GGR selectors, plus JJR simulation thresholds;
- PCA and projected PCA;
- Bai-Li ML, QML, ML-GLS, ML-ITE and ML-EM;
- second- and third-order GMM estimators;
- third- and fourth-order HFA alternating least squares;
- Adaptive HFA;
- synthetic DGP1/DGP2 simulations with skewed generalized-t innovations;
- PC and IC higher-moment portfolio optimization;
- moment/cumulant utilities, covariance shrinkage, and simulation diagnostics.

This repository is currently structured as a local SDK, but the packaging metadata is compatible with a future PyPI publication.

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)

Core dependencies are NumPy, SciPy, pandas and scikit-learn. Notebook-only dependencies are kept in the development/example groups.

## Setup

```bash
uv sync --all-groups
```

Then run:

```bash
uv run pytest
uv run ruff check .
```

To use the package from this checkout:

```bash
uv run python
```

```python
import pyhofa
```

## Quick start

### PCA and HFA

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

### Factor-number selection

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

### Adaptive HFA

```python
from pyhofa import adaptive_hfa

result = adaptive_hfa(sim.X, r=2, max_order=4)
print(result.cumulant_order_f)
print(result.cumulant_order_u)
print(result.factor_contribution_ratios)
```

### ML and GMM

```python
from pyhofa import m2_gmm, m2_mle, m3_gmm

ml = m2_mle(sim.X, r=2, method="ML")
em = m2_mle(sim.X, r=2, method="ML-EM", ar_order=1)
gmm2 = m2_gmm(sim.X, r=2)
gmm3 = m3_gmm(sim.X, r=2)
```

### Projected PCA

`characteristics` has one row per panel variable.

```python
ppca = pyhofa.m2_pca(
    X,
    r=3,
    method="P-PCA",
    characteristics=characteristics,
    sieve_terms=6,
)
```

The Python implementation uses additive cubic B-spline bases for the sieve projection.

## API mapping from the R package

| R function | Python function |
|---|---|
| `M2.select` | `pyhofa.m2_select` / `pyhofa.M2_select` |
| `M3.select` | `pyhofa.m3_select` / `pyhofa.M3_select` |
| `M4.select` | `pyhofa.m4_select` / `pyhofa.M4_select` |
| `M2.pca` | `pyhofa.m2_pca` / `pyhofa.M2_pca` |
| `M2.mle` | `pyhofa.m2_mle` / `pyhofa.M2_mle` |
| `M2.gmm` | `pyhofa.m2_gmm` / `pyhofa.M2_gmm` |
| `M3.gmm` | `pyhofa.m3_gmm` / `pyhofa.M3_gmm` |
| `M3.als` | `pyhofa.m3_als` / `pyhofa.M3_als` |
| `M4.als` | `pyhofa.m4_als` / `pyhofa.M4_als` |
| `Adaptive.HFA` | `pyhofa.adaptive_hfa` / `pyhofa.Adaptive_HFA` |
| `Portfolio.IC` | `pyhofa.portfolio_ic` / `pyhofa.Portfolio_IC` |
| `Portfolio.PC` | `pyhofa.portfolio_pc` / `pyhofa.Portfolio_PC` |
| `hofa.DGP1` | `pyhofa.hofa_DGP1` or `pyhofa.dgp1` |
| `hofa.DGP2` | `pyhofa.hofa_DGP2` or `pyhofa.dgp2` |
| `M2M`, `M3M`, `M4M`, `C4M` | `m2m`, `m3m`, `m4m`, `c4m` |
| `TraceRatio` | `trace_ratio` |

Python result objects use descriptive attributes (`factors`, `loadings`, `residuals`) and also provide `f`, `u`, `e`, `ev`, `R`, `Rh`, and `Rg` compatibility properties where appropriate.

## Higher-order moment implementation

The third-order Gram matrix is computed without creating an explicit `n x n²` tensor:

```text
M3M(X) = X' [(XX') ⊙ (XX')] X
```

Similarly, the raw fourth-order product is

```text
M4M(X) = X' [(XX') ⊙ (XX') ⊙ (XX')] X
```

The fourth-cumulant product `c4m(X)` uses the algebraic expansion of `C4 @ C4.T`. This avoids materializing an `n x n³` cumulant matrix in the HFA selectors and ALS estimator. Explicit third/fourth moment and cumulant matricizations are still provided for small-dimensional tasks such as portfolio co-moments and validation.

## Intentional corrections and Python-specific choices

This is not a blind transliteration. Several points in the R source are internally inconsistent or unsafe for production use. The implementation follows the mathematical intent and documents the differences.

1. **Rows are always observations.** The R `M2.select` silently transposes the input when the number of columns exceeds the number of rows. Python never does this. A `t x n` panel remains a `t x n` panel even when `n > t`.

2. **M2 ER/GR names follow their formulas.** In the R source, the variables built from the eigenvalue-ratio and growth-ratio statistics are assigned to the opposite public names at the end of the branch. Python uses ER for the eigenvalue-ratio statistic and GR for the growth-ratio statistic.

3. **Fourth-order ALS uses `n**4` consistently.** The R initialization divides the fourth-order term by `n^4`, but one line inside its ALS loop divides it by `n^3`. Python treats the loop denominator as a typo and uses `n^4` throughout.

4. **ALS has a finite iteration cap and sign alignment.** Eigenvectors are sign-indeterminate. Comparing successive raw eigenvectors can therefore report false non-convergence after a sign flip. Python aligns signs before evaluating the stopping criterion and exposes `max_iter`.

5. **ML variance updates are positivity-protected.** Small negative idiosyncratic variance estimates caused by floating-point error are floored before inversion.

6. **ML-EM is implemented directly as a state-space model.** The state is `[f_t, f_{t-1}]`, with the AR(1)-prewhitened observation equation used in the R implementation. Filtering and smoothing use a Kalman filter plus Rauch-Tung-Striebel smoother.

7. **Projected PCA uses Python spline tooling.** R's `gam::s()` is replaced by additive cubic `SplineTransformer` bases followed by an orthogonal least-squares projection.

8. **GMM uses an actual orthogonal projection.** The factor-loading span is removed with `U (U'U)^+ U'`, rather than assuming the initialized loadings already form an orthonormal basis.

These choices are covered by tests where a direct mathematical identity is available.

## Simulation

The package includes a native sampler for the variance-adjusted skewed generalized-t parameterization used by the R package.

```python
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

The `dgp2` helper additionally accepts `loading_strength`, which is useful for direct strong/weak-factor experiments without rewriting the DGP.

## Portfolio routines

Both portfolio estimators support:

- `objective="MVaR"`: higher-moment modified VaR objective;
- `objective="EU"`: fourth-order CRRA expected-utility approximation;
- covariance adjustment `"NONE"`, `"LI"`, or `"DNL"`;
- constrained or short-selling portfolios.

```python
result = pyhofa.portfolio_pc(
    returns,
    r=3,
    objective="EU",
    covariance_adjustment="LI",
    shortselling=False,
)

weights = result.weights
```

## Development notebooks

`dev/` contains three unexecuted notebooks:

1. `01_weak_factor_simulation.ipynb`: PCA versus higher-order HFA on a weak non-Gaussian synthetic panel.
2. `02_fama_french_industries.ipynb`: Fama-French 49 industry portfolios obtained through `pandas-datareader`.
3. `03_fred_macro.ipynb`: a compact FRED macro panel and Adaptive HFA example.

The notebooks fetch public data when executed. No external datasets are committed to the repository.

## Testing

The test suite includes:

- exact `M3M = M3 M3'` checks;
- exact `M4M = M4 M4'` checks;
- exact `C4M = C4 C4'` checks;
- fourth moment/cumulant round trips;
- SGT analytical and Monte Carlo moment checks;
- DGP reconstruction checks;
- PCA/projected-PCA, ML, GMM and ALS smoke/invariant tests;
- second-, third- and fourth-order factor selectors;
- Adaptive HFA;
- PC/IC portfolio constraints.

Run:

```bash
uv run pytest --cov=pyhofa --cov-report=term-missing
```

## Repository layout

```text
.
├── .github/workflows/ci.yml
├── dev/
│   ├── 01_weak_factor_simulation.ipynb
│   ├── 02_fama_french_industries.ipynb
│   ├── 03_fred_macro.ipynb
│   └── README.md
├── src/pyhofa/
│   ├── __init__.py
│   ├── _types.py
│   ├── _utils.py
│   ├── adaptive.py
│   ├── estimators.py
│   ├── moments.py
│   ├── portfolio.py
│   ├── selection.py
│   └── simulation.py
├── tests/
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

## Build

The project uses a standard `src/` layout and setuptools build backend, so future distribution is straightforward:

```bash
uv build
```

A future PyPI release should decide the final distribution name first. The distribution and import package are both named `pyhofa`.

## License and attribution

The upstream R package declares GPL-2. This Python implementation is therefore distributed under **GPL-2.0-only** as a derivative implementation.

Primary upstream project:

- Guanglin Huang, `GuanglinHuang/hofa`
- Huang, G., Lu, W., and Boudt, K., *Estimation of factors using higher-order multi-cumulants in weak factor models*.

This Python port is not an official release of the upstream authors.
