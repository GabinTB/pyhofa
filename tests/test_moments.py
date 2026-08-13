import numpy as np

from pyhofa import (
    c4_to_m4,
    c4m,
    fourth_cumulant_matrix,
    fourth_moment_matrix,
    m3m,
    m4_to_c4,
    m4m,
    portfolio_moments_independent,
    super_diag,
    third_moment_matrix,
    trace_ratio,
)


def test_m3m_matches_explicit_third_moment_product() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 7))
    x -= x.mean(axis=0)
    m3 = third_moment_matrix(x, center=False)
    np.testing.assert_allclose(m3m(x) / x.shape[0] ** 2, m3 @ m3.T, atol=1e-12)


def test_m4m_matches_explicit_fourth_moment_product() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 5))
    x -= x.mean(axis=0)
    m4 = fourth_moment_matrix(x, center=False)
    np.testing.assert_allclose(m4m(x) / x.shape[0] ** 2, m4 @ m4.T, atol=1e-11)


def test_c4m_matches_explicit_fourth_cumulant_product() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(50, 6))
    x -= x.mean(axis=0)
    cov = x.T @ x / x.shape[0]
    c4 = fourth_cumulant_matrix(x, covariance=cov, center=False)
    np.testing.assert_allclose(c4m(x, cov), c4 @ c4.T, atol=1e-10, rtol=1e-10)


def test_cumulant_moment_round_trip() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_t(df=8, size=(100, 4))
    x -= x.mean(axis=0)
    cov = x.T @ x / x.shape[0]
    c4 = fourth_cumulant_matrix(x, covariance=cov, center=False)
    m4 = c4_to_m4(c4, cov)
    np.testing.assert_allclose(m4, fourth_moment_matrix(x, center=False), atol=1e-12)
    np.testing.assert_allclose(m4_to_c4(m4, cov), c4, atol=1e-12)


def test_super_diag_locations() -> None:
    x = np.array([2.0, 3.0])
    d3 = super_diag(x, 3)
    assert d3.shape == (2, 4)
    assert d3[0, 0] == 2.0
    assert d3[1, 3] == 3.0
    assert np.count_nonzero(d3) == 2


def test_trace_ratio_is_one_for_same_subspace() -> None:
    rng = np.random.default_rng(5)
    f = rng.normal(size=(100, 3))
    rotated = f @ np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    assert np.isclose(trace_ratio(rotated, f), 1.0)


def test_portfolio_independent_moments_match_tensor_construction() -> None:
    rng = np.random.default_rng(6)
    n, k = 8, 2
    w = np.full(n, 1 / n)
    a = rng.normal(size=(n, k))
    m2f = rng.uniform(0.5, 2.0, size=k)
    m3f = rng.uniform(-0.5, 0.5, size=k) * m2f**1.5
    m4f = rng.uniform(3.0, 6.0, size=k) * m2f**2
    m2e = rng.uniform(0.5, 2.0, size=n)
    m3e = rng.uniform(-0.5, 0.5, size=n) * m2e**1.5
    m4e = rng.uniform(3.0, 6.0, size=n) * m2e**2

    m2x = a @ np.diag(m2f) @ a.T + np.diag(m2e)
    m3x = a @ super_diag(m3f, 3) @ np.kron(a.T, a.T) + super_diag(m3e, 3)
    c4x = (
        a
        @ super_diag(m4f - 3 * m2f**2, 4)
        @ np.kron(np.kron(a.T, a.T), a.T)
        + super_diag(m4e - 3 * m2e**2, 4)
    )
    m4x = c4_to_m4(c4x, m2x)
    expected = np.array(
        [
            w @ m2x @ w,
            w @ m3x @ np.kron(w, w),
            w @ m4x @ np.kron(np.kron(w, w), w),
        ]
    )
    actual = portfolio_moments_independent(w, (m2f, m3f, m4f), (m2e, m3e, m4e), a)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
