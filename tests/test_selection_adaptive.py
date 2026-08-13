import numpy as np

from pyhofa import adaptive_hfa, dgp2, m2_select, m3_select, m4_select


def _panel() -> np.ndarray:
    return dgp2(
        35,
        180,
        2,
        factor_lambda=[0.85, 0.7],
        factor_p=[1.0, 1.0],
        factor_ar=[0.5, 0.2],
        beta=0.1,
        neighborhood=2,
        error_ar=0.05,
        error_variance_range=(0.5, 1.0),
        random_state=19,
    ).X


def test_second_order_selectors_return_valid_counts() -> None:
    x = _panel()
    for method in ["ER", "GR", "BN-IC3", "BN-PC3", "BIC3", "ON", "ACT"]:
        result = m2_select(x, rmax=5, method=method)
        assert 0 <= result.n_factors <= 5
        assert result.eigenvalues is not None


def test_higher_order_selectors_return_decomposition() -> None:
    x = _panel()
    for selector, method in [(m3_select, "GER3"), (m4_select, "GER4")]:
        result = selector(x, rmax=4, method=method)
        assert result.n_nongaussian is not None
        assert result.n_gaussian is not None
        assert result.n_factors == result.n_nongaussian + result.n_gaussian


def test_adaptive_hfa_explicit_factor_count() -> None:
    x = _panel()
    result = adaptive_hfa(x, r=2, max_order=3)
    assert result.factors.shape == (x.shape[0], 2)
    assert result.loadings.shape == (x.shape[1], 2)
    assert result.cumulant_order_f in {2, 3}
    assert result.cumulant_order_u in {2, 3}
    assert set(result.factor_contribution_ratios) == {2, 3}
