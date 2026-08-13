import numpy as np

from pyhofa import dgp2, m2_gmm, m2_mle, m2_pca, m3_als, m3_gmm, m4_als, trace_ratio


def _sample() -> tuple[np.ndarray, np.ndarray]:
    sim = dgp2(
        24,
        140,
        2,
        factor_lambda=[0.8, 0.5],
        factor_p=[1.0, 1.0],
        factor_ar=[0.4, 0.1],
        beta=0.1,
        neighborhood=2,
        error_ar=0.1,
        random_state=7,
    )
    return sim.X, sim.factors


def test_pca_shapes_and_normalization() -> None:
    x, _ = _sample()
    result = m2_pca(x, 2)
    assert result.factors.shape == (140, 2)
    assert result.loadings.shape == (24, 2)
    np.testing.assert_allclose(result.loadings.T @ result.loadings / 24, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(
        result.residuals,
        (x - x.mean(axis=0)) - result.factors @ result.loadings.T,
        atol=1e-12,
    )


def test_projected_pca_runs() -> None:
    rng = np.random.default_rng(8)
    n, t = 30, 60
    c = rng.normal(size=(n, 2))
    true_loadings = np.column_stack([c[:, 0] ** 2 - 1, np.sin(c[:, 1])])
    factors = rng.normal(size=(t, 2))
    x = factors @ true_loadings.T + 0.2 * rng.normal(size=(t, n))
    result = m2_pca(x, 2, method="P-PCA", characteristics=c, sieve_terms=5)
    assert result.factors.shape == (t, 2)
    assert result.loadings.shape == (n, 2)
    assert result.metadata["G"].shape == (n, 2)


def test_bai_li_ml_runs_and_has_positive_idiosyncratic_variances() -> None:
    x, _ = _sample()
    result = m2_mle(x, 2, method="ML", max_iter=150)
    assert result.factors.shape == (140, 2)
    assert np.all(np.asarray(result.metadata["idiosyncratic_variance"]) > 0)


def test_gls_variants_run() -> None:
    x, _ = _sample()
    for method in ["ML-GLS", "ML-ITE", "ML-EM"]:
        result = m2_mle(x, 2, method=method, max_iter=5, tol=1e-5)
        assert result.factors.shape == (140, 2)
        assert "rho" in result.metadata


def test_gmm_estimators_run() -> None:
    x, _ = _sample()
    result2 = m2_gmm(x, 2, delta=1e-8)
    result3 = m3_gmm(x, 2, delta=1e-8)
    assert result2.loadings.shape == (24, 2)
    assert result3.loadings.shape == (24, 2)
    assert np.all(np.isfinite(result2.factors))
    assert np.all(np.isfinite(result3.factors))


def test_higher_order_als_recovers_nontrivial_factor_subspace() -> None:
    x, truth = _sample()
    result3 = m3_als(x, rh=2, rg=0, tol=1e-6, max_iter=100)
    result4 = m4_als(x, rh=2, rg=0, tol=1e-6, max_iter=100)
    assert trace_ratio(result3.factors, truth) > 0.25
    assert trace_ratio(result4.factors, truth) > 0.25
