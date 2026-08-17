"""Out-of-sample factor projection."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._types import FloatArray
from ._utils import as_2d_float


def project(
    x_new: ArrayLike,
    loadings: ArrayLike,
    *,
    mean: ArrayLike,
    scale: ArrayLike | None = None,
) -> FloatArray:
    """Project new observations using frozen training preprocessing and loadings.

    The projection is ``((x_new - mean) / scale) @ loadings / n_variables``.
    ``scale`` is omitted for fits performed without standardization. Neither
    statistic is re-estimated from ``x_new``.
    """
    data = as_2d_float(x_new, name="x_new")
    loading_matrix = as_2d_float(loadings, name="loadings")
    if loading_matrix.shape[0] != data.shape[1]:
        raise ValueError("loadings must have one row per panel variable")

    fitted_mean = np.asarray(mean, dtype=np.float64)
    if fitted_mean.ndim != 1 or fitted_mean.shape[0] != data.shape[1]:
        raise ValueError("mean must be one-dimensional with one value per panel variable")
    if not np.all(np.isfinite(fitted_mean)):
        raise ValueError("mean contains NaN or infinite values")

    transformed = data - fitted_mean
    if scale is not None:
        fitted_scale = np.asarray(scale, dtype=np.float64)
        if fitted_scale.ndim != 1 or fitted_scale.shape[0] != data.shape[1]:
            raise ValueError("scale must be one-dimensional with one value per panel variable")
        if not np.all(np.isfinite(fitted_scale)) or np.any(fitted_scale <= 0.0):
            raise ValueError("scale must contain finite positive values")
        transformed = transformed / fitted_scale

    return transformed @ loading_matrix / data.shape[1]
