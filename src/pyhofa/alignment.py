"""Cross-fit loading alignment."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment

from ._types import FloatArray
from ._utils import align_columns, as_2d_float


def _normalized_columns(matrix: FloatArray) -> FloatArray:
    """Normalize centered columns, falling back to cosine similarity for constants."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    constant = norms <= np.finfo(float).eps
    if np.any(constant):
        centered[:, constant] = matrix[:, constant]
        norms[constant] = np.linalg.norm(centered[:, constant], axis=0)
    if np.any(norms <= np.finfo(float).eps):
        raise ValueError("reference and candidate columns must have nonzero variation or norm")
    return centered / norms


def align_loadings(
    reference: ArrayLike,
    candidate: ArrayLike,
    *,
    allow_rotation: bool = True,
) -> FloatArray:
    """Align candidate loadings to a reference fit.

    By default, an orthogonal Procrustes transformation aligns the candidate
    factor subspace directly. Set ``allow_rotation=False`` to instead use
    Hungarian matching on absolute column correlations followed by sign
    resolution. Both matrices must have the same shape, as expected when the
    factor count is fixed across rolling fits.
    """
    reference_matrix = as_2d_float(reference, name="reference")
    candidate_matrix = as_2d_float(candidate, name="candidate")
    if reference_matrix.shape != candidate_matrix.shape:
        raise ValueError("reference and candidate must have the same shape")

    if allow_rotation:
        rotation, _ = orthogonal_procrustes(candidate_matrix, reference_matrix)
        return align_columns(reference_matrix, candidate_matrix @ rotation)

    similarity = _normalized_columns(reference_matrix).T @ _normalized_columns(
        candidate_matrix
    )
    reference_indices, candidate_indices = linear_sum_assignment(-np.abs(similarity))
    permutation = candidate_indices[np.argsort(reference_indices)]
    aligned = candidate_matrix[:, permutation].copy()
    matched_similarity = similarity[np.arange(similarity.shape[0]), permutation]
    aligned[:, matched_similarity < 0.0] *= -1.0
    return aligned
