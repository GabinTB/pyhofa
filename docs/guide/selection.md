# Factor Selection

pyhofa provides second-, third- and fourth-order factor-number selection.

- **Second order** (`m2_select`): ER, GR, Bai-Ng IC3/PC3, BIC3, Onatski, ACT.
- **Third order** (`m3_select`): GER3/GGR3 selectors that split factors into a non-Gaussian and a
  Gaussian block, plus JJR simulation thresholds.
- **Fourth order** (`m4_select`): GER4/GGR4 analogues using fourth-cumulant information.

```python
from pyhofa import m2_select, m3_select, m4_select

r_er = m2_select(X, method="ER", rmax=8)
r_ger3 = m3_select(X, method="GER3", rmax=8)
r_ger4 = m4_select(X, method="GER4", rmax=8)
```

Result objects expose descriptive attributes (`n_factors`, `n_nongaussian`, `n_gaussian`) and also
R-compatibility fields (`R`, `Rh`, `Rg`) for parity with the upstream `hofa` package.

See [R Package Mapping & Corrections](r-mapping.md) for the M2 ER/GR naming fix relative to the R
source.

## API reference

::: pyhofa.selection
