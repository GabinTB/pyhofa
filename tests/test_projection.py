import numpy as np
import pytest

from pyhofa import adaptive_hfa, m2_pca, project


def test_project_reproduces_pca_training_scores() -> None:
    rng = np.random.default_rng(21)
    x = rng.normal(loc=np.arange(5), scale=np.arange(1, 6), size=(40, 5))
    result = m2_pca(x, 2, scale=True)

    projected = project(
        x,
        result.loadings,
        mean=result.mean_,
        scale=result.scale_,
    )

    np.testing.assert_allclose(projected, result.factors, atol=1e-12)
    np.testing.assert_allclose(result.mean_, x.mean(axis=0))
    np.testing.assert_allclose(result.scale_, x.std(axis=0, ddof=1))


def test_project_uses_training_statistics_for_new_observations() -> None:
    x_train = np.array([[0.0, 2.0], [2.0, 6.0], [4.0, 10.0]])
    x_new = np.array([[10.0, 20.0], [12.0, 24.0]])
    loadings = np.array([[2.0], [4.0]])
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0, ddof=1)

    result = project(x_new, loadings, mean=mean, scale=scale)

    expected = ((x_new - mean) / scale) @ loadings / x_new.shape[1]
    np.testing.assert_allclose(result, expected)
    assert not np.allclose(result.mean(axis=0), 0.0)


def test_project_validates_frozen_statistics() -> None:
    x = np.ones((3, 2))
    loadings = np.ones((2, 1))

    with pytest.raises(ValueError, match="mean must"):
        project(x, loadings, mean=np.zeros(3))
    with pytest.raises(ValueError, match="scale must contain"):
        project(x, loadings, mean=np.zeros(2), scale=np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="one row per panel variable"):
        project(x, np.ones((3, 1)), mean=np.zeros(2))


def test_adaptive_result_exposes_training_statistics() -> None:
    rng = np.random.default_rng(22)
    x = rng.normal(size=(30, 6))

    result = adaptive_hfa(x, r=2, max_order=3, scale=True, rmax=3)

    np.testing.assert_allclose(result.mean_, x.mean(axis=0))
    np.testing.assert_allclose(result.scale_, x.std(axis=0, ddof=1))


def test_adaptive_projection_round_trip_when_selected_orders_differ() -> None:
    rng = np.random.default_rng(0)
    t, n = 40, 8
    factors = np.column_stack(
        [
            rng.standard_t(3, size=t),
            rng.normal(size=t),
            rng.exponential(size=t) - 1.0,
        ]
    )
    loadings = rng.normal(size=(n, 3))
    x = factors @ loadings.T + 0.5 * rng.normal(size=(t, n))

    result = adaptive_hfa(x, r=3, max_order=3, rmax=4, tau_nt=10.0)

    assert result.cumulant_order_f != result.cumulant_order_u
    projected = project(
        x,
        result.loadings,
        mean=result.mean_,
        scale=result.scale_,
    )
    np.testing.assert_allclose(projected, result.factors, atol=1e-12)
