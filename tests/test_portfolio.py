import numpy as np

from pyhofa import portfolio_ic, portfolio_pc


def _returns() -> np.ndarray:
    rng = np.random.default_rng(33)
    factors = rng.standard_t(df=7, size=(180, 2)) * 0.01
    loadings = rng.normal(size=(10, 2))
    errors = rng.normal(scale=0.006, size=(180, 10))
    return factors @ loadings.T + errors


def test_pc_portfolio_weights_sum_to_one() -> None:
    result = portfolio_pc(_returns(), r=2, objective="EU", covariance_adjustment="NONE")
    assert np.isclose(result.weights.sum(), 1.0)
    assert np.all(np.isfinite(result.portfolio_moments))


def test_ic_portfolio_weights_sum_to_one() -> None:
    result = portfolio_ic(
        _returns(),
        r=2,
        objective="MVaR",
        covariance_adjustment="NONE",
        random_state=1,
    )
    assert np.isclose(result.weights.sum(), 1.0)
    assert np.all(np.isfinite(result.weights))
