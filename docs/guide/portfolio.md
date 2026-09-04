# Portfolio Optimization

`portfolio_ic` and `portfolio_pc` build higher-moment portfolios from IC- or PC-estimated factor
models. Both support:

- `objective="MVaR"`: higher-moment modified VaR objective
- `objective="EU"`: fourth-order CRRA expected-utility approximation
- covariance adjustment `"NONE"`, `"LI"`, or `"DNL"`
- constrained or short-selling portfolios

```python
import pyhofa

result = pyhofa.portfolio_pc(
    returns,
    r=3,
    objective="EU",
    covariance_adjustment="LI",
    shortselling=False,
)

weights = result.weights
```

## API reference

::: pyhofa.portfolio
