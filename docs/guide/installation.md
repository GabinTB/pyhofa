# Installation

## From PyPI

```bash
pip install pyhofa
```

Core dependencies are NumPy, SciPy, pandas and scikit-learn.

## From source

```bash
git clone https://github.com/GabinTB/pyhofa
cd pyhofa
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

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for source installs
