import numpy as np

from pyhofa import dgp2, sgt_moment, sgt_rvs


def test_sgt_gaussian_limit_moments() -> None:
    assert np.isclose(sgt_moment(1, lam=0, p=2, q=np.inf), 0.0)
    assert np.isclose(sgt_moment(2, lam=0, p=2, q=np.inf), 1.0)
    assert np.isclose(sgt_moment(3, lam=0, p=2, q=np.inf), 0.0)
    assert np.isclose(sgt_moment(4, lam=0, p=2, q=np.inf), 3.0)


def test_sgt_random_sample_is_variance_adjusted() -> None:
    sample = sgt_rvs(120_000, lam=0.6, p=1.2, q=np.inf, rng=123)
    assert abs(sample.mean()) < 0.03
    assert abs(sample.var() - 1.0) < 0.05
    assert np.mean((sample - sample.mean()) ** 3) > 0.5


def test_dgp2_shapes_and_reconstruction() -> None:
    result = dgp2(
        25,
        120,
        3,
        factor_ar=[0.5, 0.2, 0.0],
        factor_lambda=[0.8, 0.5, 0.0],
        random_state=42,
    )
    assert result.X.shape == (120, 25)
    assert result.factors.shape == (120, 3)
    assert result.loadings.shape == (25, 3)
    np.testing.assert_allclose(result.X, result.factors @ result.loadings.T + result.errors)
