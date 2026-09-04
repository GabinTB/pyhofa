# R Package Mapping & Corrections

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

Python result objects use descriptive attributes (`factors`, `loadings`, `residuals`) and also
provide `f`, `u`, `e`, `ev`, `R`, `Rh`, and `Rg` compatibility properties where appropriate.

## Higher-order moment implementation

The third-order Gram matrix is computed without creating an explicit `n x n²` tensor:

```text
M3M(X) = X' [(XX') ⊙ (XX')] X
```

Similarly, the raw fourth-order product is

```text
M4M(X) = X' [(XX') ⊙ (XX') ⊙ (XX')] X
```

The fourth-cumulant product `c4m(X)` uses the algebraic expansion of `C4 @ C4.T`. This avoids
materializing an `n x n³` cumulant matrix in the HFA selectors and ALS estimator. Explicit
third/fourth moment and cumulant matricizations are still provided for small-dimensional tasks
such as portfolio co-moments and validation.

## Intentional corrections and Python-specific choices

This is not a blind transliteration. Several points in the R source are internally inconsistent or
unsafe for production use. The implementation follows the mathematical intent and documents the
differences.

1. **Rows are always observations.** The R `M2.select` silently transposes the input when the
   number of columns exceeds the number of rows. Python never does this. A `t x n` panel remains a
   `t x n` panel even when `n > t`.

2. **M2 ER/GR names follow their formulas.** In the R source, the variables built from the
   eigenvalue-ratio and growth-ratio statistics are assigned to the opposite public names at the
   end of the branch. Python uses ER for the eigenvalue-ratio statistic and GR for the growth-ratio
   statistic.

3. **Fourth-order ALS uses `n**4` consistently.** The R initialization divides the fourth-order
   term by `n^4`, but one line inside its ALS loop divides it by `n^3`. Python treats the loop
   denominator as a typo and uses `n^4` throughout.

4. **ALS has a finite iteration cap and sign alignment.** Eigenvectors are sign-indeterminate.
   Comparing successive raw eigenvectors can therefore report false non-convergence after a sign
   flip. Python aligns signs before evaluating the stopping criterion and exposes `max_iter`.

5. **ML variance updates are positivity-protected.** Small negative idiosyncratic variance
   estimates caused by floating-point error are floored before inversion.

6. **ML-EM is implemented directly as a state-space model.** The state is `[f_t, f_{t-1}]`, with
   the AR(1)-prewhitened observation equation used in the R implementation. Filtering and
   smoothing use a Kalman filter plus Rauch-Tung-Striebel smoother.

7. **Projected PCA uses Python spline tooling.** R's `gam::s()` is replaced by additive cubic
   `SplineTransformer` bases followed by an orthogonal least-squares projection.

8. **GMM uses an actual orthogonal projection.** The factor-loading span is removed with
   `U (U'U)^+ U'`, rather than assuming the initialized loadings already form an orthonormal basis.

These choices are covered by tests where a direct mathematical identity is available.
