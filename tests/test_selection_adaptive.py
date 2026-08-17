import numpy as np
import pytest

from pyhofa import adaptive_hfa, dgp2, m2_select, m3_select, m4_select
from pyhofa._types import FactorResult, SelectionResult


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
    assert result.n_nongaussian + result.n_gaussian == result.n_factors
    assert result.Rh == result.n_nongaussian
    assert result.Rg == result.n_gaussian


def test_adaptive_hfa_threads_selected_split_into_als(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyhofa.adaptive as adaptive_module

    x = np.arange(60, dtype=np.float64).reshape(12, 5)
    calls: list[tuple[str, int, int]] = []

    monkeypatch.setattr(
        adaptive_module,
        "m3_select",
        lambda *args, **kwargs: SelectionResult(3, n_nongaussian=2, n_gaussian=1),
    )
    monkeypatch.setattr(
        adaptive_module,
        "m4_select",
        lambda *args, **kwargs: SelectionResult(4, n_nongaussian=2, n_gaussian=2),
    )

    def fake_als(
        data: np.ndarray,
        *,
        rh: int,
        rg: int,
        gamma: tuple[float, ...],
    ) -> FactorResult:
        calls.append((f"m{len(gamma) + 1}", rh, rg))
        loadings = np.zeros((data.shape[1], rh + rg))
        factors = np.zeros((data.shape[0], rh + rg))
        return FactorResult(factors, loadings, data.copy())

    monkeypatch.setattr(adaptive_module, "m3_als", fake_als)
    monkeypatch.setattr(adaptive_module, "m4_als", fake_als)

    result = adaptive_hfa(x, r=4, max_order=4)

    assert calls == [("m3", 2, 2), ("m4", 2, 2)]
    assert result.n_nongaussian == 2
    assert result.n_gaussian == 2


def test_adaptive_hfa_uses_total_selected_factor_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyhofa.adaptive as adaptive_module

    x = np.arange(60, dtype=np.float64).reshape(12, 5)
    monkeypatch.setattr(
        adaptive_module,
        "m2_select",
        lambda *args, **kwargs: SelectionResult(3),
    )
    monkeypatch.setattr(
        adaptive_module,
        "m3_select",
        lambda *args, **kwargs: SelectionResult(3, n_nongaussian=3, n_gaussian=0),
    )
    monkeypatch.setattr(
        adaptive_module,
        "m4_select",
        lambda *args, **kwargs: SelectionResult(4, n_nongaussian=3, n_gaussian=1),
    )
    monkeypatch.setattr(
        adaptive_module,
        "m3_als",
        lambda data, *, rh, rg, gamma: FactorResult(
            np.zeros((data.shape[0], rh + rg)),
            np.zeros((data.shape[1], rh + rg)),
            data.copy(),
        ),
    )

    result = adaptive_hfa(x, max_order=3, rmax=4)

    assert result.n_factors == 4
    assert result.n_nongaussian == 3
    assert result.n_gaussian == 1
