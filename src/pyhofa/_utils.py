"""Internal numerical utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def as_2d_float(x: ArrayLike, *, name: str = "X") -> FloatArray:
    """Convert input to a finite two-dimensional float64 array."""
    out = np.asarray(x, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {out.shape!r}")
    if min(out.shape) == 0:
        raise ValueError(f"{name} must have at least one row and one column")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return out


def center_scale(x: ArrayLike, *, scale: bool = False, ddof: int = 1) -> FloatArray:
    """Center columns and optionally scale them to unit sample variance."""
    out = as_2d_float(x).copy()
    out -= out.mean(axis=0, keepdims=True)
    if scale:
        std = out.std(axis=0, ddof=ddof)
        if np.any(std <= np.finfo(float).eps):
            raise ValueError("cannot scale a constant column")
        out /= std
    return out


def eigh_desc(a: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Eigenpairs of a symmetric matrix in descending eigenvalue order."""
    matrix = np.asarray(a, dtype=np.float64)
    matrix = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def safe_pinv(a: ArrayLike, *, rcond: float = 1e-12) -> FloatArray:
    """Numerically stable Moore-Penrose inverse."""
    return np.linalg.pinv(np.asarray(a, dtype=np.float64), rcond=rcond)


def align_columns(reference: FloatArray, candidate: FloatArray) -> FloatArray:
    """Resolve eigenvector sign indeterminacy column-wise."""
    aligned = candidate.copy()
    n_cols = min(reference.shape[1], candidate.shape[1])
    for j in range(n_cols):
        if float(reference[:, j] @ aligned[:, j]) < 0.0:
            aligned[:, j] *= -1.0
    return aligned


def top_eigenvectors(a: ArrayLike, r: int) -> tuple[FloatArray, FloatArray]:
    """Return the top ``r`` eigenvalues and vectors of a symmetric matrix."""
    if r < 0:
        raise ValueError("r must be non-negative")
    values, vectors = eigh_desc(a)
    if r > vectors.shape[1]:
        raise ValueError(f"r={r} exceeds matrix dimension {vectors.shape[1]}")
    return values[:r], vectors[:, :r]


def validate_rmax(rmax: int, n_eigenvalues: int) -> int:
    """Validate a maximum factor count while keeping ratio denominators available."""
    if rmax < 1:
        raise ValueError("rmax must be at least 1")
    if n_eigenvalues < 3:
        raise ValueError("at least three eigenvalues are required for factor selection")
    return min(int(rmax), n_eigenvalues - 2)


def pava(values: ArrayLike, *, increasing: bool = True) -> FloatArray:
    """Pool-adjacent-violators isotonic regression with unit weights."""
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if y.size == 0:
        return y.copy()
    work = y if increasing else -y
    levels: list[float] = []
    weights: list[int] = []
    for value in work:
        levels.append(float(value))
        weights.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            w = weights[-2] + weights[-1]
            level = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / w
            levels[-2:] = [level]
            weights[-2:] = [w]
    out = np.concatenate([np.full(w, level) for level, w in zip(levels, weights, strict=True)])
    return out if increasing else -out


def ar_coefficients(x: ArrayLike, order: int) -> FloatArray:
    """OLS autoregressive coefficients with no intercept on centered data."""
    series = np.asarray(x, dtype=np.float64)
    if order < 1 or order >= series.size:
        raise ValueError("AR order must be between 1 and len(x)-1")
    y = series[order:]
    design = np.column_stack([series[order - lag - 1 : -lag - 1] for lag in range(order)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef


def select_ar_order_aic(x: ArrayLike, max_order: int = 5) -> tuple[int, FloatArray]:
    """Choose an AR order by the Gaussian conditional AIC."""
    series = np.asarray(x, dtype=np.float64)
    best: tuple[float, int, FloatArray] | None = None
    for order in range(1, min(max_order, series.size - 2) + 1):
        coef = ar_coefficients(series, order)
        y = series[order:]
        design = np.column_stack(
            [series[order - lag - 1 : -lag - 1] for lag in range(order)]
        )
        resid = y - design @ coef
        sigma2 = max(float(resid @ resid) / resid.size, np.finfo(float).tiny)
        aic = resid.size * np.log(sigma2) + 2 * order
        if best is None or aic < best[0]:
            best = (aic, order, coef)
    if best is None:
        raise ValueError("series is too short for autoregression")
    return best[1], best[2]
