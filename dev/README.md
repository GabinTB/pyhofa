# Development notebooks

The notebooks are intentionally stored without outputs. They download public data at run time.

- `01_weak_factor_simulation.ipynb`: synthetic weak-factor comparison of PCA and HFA.
- `02_fama_french_industries.ipynb`: Fama-French 49 industry portfolios via `pandas-datareader`.
- `03_fred_macro.ipynb`: a compact FRED macro panel via `pandas-datareader`.

Install the example dependencies with:

```bash
uv sync --all-groups
```
