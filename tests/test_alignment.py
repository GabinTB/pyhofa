import numpy as np
import pytest

from pyhofa import align_columns, align_loadings


def test_align_columns_is_public_and_resolves_signs() -> None:
    reference = np.array([[1.0, 2.0], [3.0, -1.0], [2.0, 4.0]])
    candidate = reference * np.array([-1.0, 1.0])

    aligned = align_columns(reference, candidate)

    np.testing.assert_allclose(aligned, reference)
    np.testing.assert_allclose(candidate, reference * np.array([-1.0, 1.0]))


def test_align_loadings_resolves_permutation_and_sign() -> None:
    rng = np.random.default_rng(31)
    reference = rng.normal(size=(20, 4))
    permutation = np.array([2, 0, 3, 1])
    signs = np.array([-1.0, 1.0, -1.0, 1.0])
    candidate = reference[:, permutation] * signs

    aligned = align_loadings(reference, candidate)

    np.testing.assert_allclose(aligned, reference, atol=1e-12)


def test_align_loadings_resolves_orthogonal_rotation() -> None:
    rng = np.random.default_rng(32)
    reference = rng.normal(size=(30, 3))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    candidate = reference @ rotation

    aligned = align_loadings(reference, candidate, allow_rotation=True)

    np.testing.assert_allclose(aligned, reference, atol=1e-12)


def test_align_loadings_validates_comparable_fits() -> None:
    with pytest.raises(ValueError, match="same shape"):
        align_loadings(np.ones((5, 2)), np.ones((5, 3)))
    with pytest.raises(ValueError, match="nonzero variation or norm"):
        align_loadings(np.column_stack([np.ones(5), np.zeros(5)]), np.ones((5, 2)))
