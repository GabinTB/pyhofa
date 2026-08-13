"""Factor estimators based on second-, third- and fourth-order information."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.preprocessing import SplineTransformer

from ._types import FactorResult
from ._utils import (
    align_columns,
    as_2d_float,
    center_scale,
    eigh_desc,
    safe_pinv,
    select_ar_order_aic,
)
from .moments import c4m, m2m, m3m

FloatArray = NDArray[np.float64]


def m2_pca(
    x: ArrayLike,
    r: int,
    *,
    characteristics: ArrayLike | None = None,
    center: bool = True,
    scale: bool = False,
    method: str = "PCA",
    sieve_terms: int | None = None,
) -> FactorResult:
    """PCA or projected-PCA factor estimation.

    For ``P-PCA``, additive cubic B-spline bases are constructed separately for
    each characteristic and the panel is projected onto their span before PCA.
    """
    if r < 1:
        raise ValueError("r must be positive")
    raw = as_2d_float(x)
    t, n = raw.shape
    if r > min(t, n):
        raise ValueError("r exceeds the rank bound min(t, n)")
    method = method.upper()
    data = raw.copy()
    if center:
        data -= data.mean(axis=0, keepdims=True)
    if scale:
        std = data.std(axis=0, ddof=1)
        if np.any(std <= np.finfo(float).eps):
            raise ValueError("cannot scale a constant column")
        data /= std

    if method == "PCA":
        covariance = data.T @ data / max(t - 1, 1)
        values, vectors = eigh_desc(covariance)
        loadings = vectors[:, :r] * np.sqrt(n)
        factors = data @ loadings / n
        residuals = data - factors @ loadings.T
        return FactorResult(factors, loadings, residuals, values / t, {"method": "PCA"})

    if method not in {"P-PCA", "PPCA", "PROJECTED-PCA"}:
        raise ValueError(f"unknown PCA method: {method!r}")
    if characteristics is None:
        raise ValueError("characteristics are required for projected PCA")
    c = as_2d_float(characteristics, name="characteristics")
    if c.shape[0] != n:
        raise ValueError("characteristics must have one row per panel variable")
    if sieve_terms is None:
        sieve_terms = max(4, round(3.0 * (n * min(n, t)) ** 0.25) - 1)
    if sieve_terms < 4:
        raise ValueError("sieve_terms must be at least 4 for cubic splines")

    bases = [np.ones((n, 1), dtype=np.float64)]
    for j in range(c.shape[1]):
        unique = np.unique(c[:, j]).size
        knots = max(2, min(int(sieve_terms), unique))
        transformer = SplineTransformer(
            n_knots=knots,
            degree=min(3, max(1, unique - 1)),
            include_bias=False,
        )
        bases.append(transformer.fit_transform(c[:, [j]]))
    design = np.column_stack(bases)
    projection_coefficients = np.linalg.lstsq(design, data.T, rcond=None)[0]
    projected = design @ projection_coefficients  # n x t

    gram = projected.T @ projected / n
    values, vectors = eigh_desc(gram)
    factors = vectors[:, :r] * np.sqrt(t)
    g_hat = projected @ factors / t
    w_hat = data.T @ factors / t
    residuals = (data.T - w_hat @ factors.T).T
    gamma_hat = w_hat - g_hat
    return FactorResult(
        factors,
        w_hat,
        residuals,
        values,
        {"method": "P-PCA", "G": g_hat, "gamma": gamma_hat, "sieve_terms": sieve_terms},
    )


def _pca_ml_initial(data: FloatArray, r: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    t, n = data.shape
    covariance = data.T @ data / t
    values, vectors = eigh_desc(covariance)
    d = np.maximum(values[:r] / n, np.finfo(float).tiny)
    loadings = vectors[:, :r] * np.sqrt(d)[None, :] * np.sqrt(n)
    factors = data @ loadings @ np.diag(1.0 / np.sqrt(d)) / n
    residuals = data - factors @ loadings.T
    variances = np.maximum(np.mean(residuals**2, axis=0), 1e-10)
    return loadings, factors, variances


def _bai_li_ml(
    covariance: FloatArray,
    loadings: FloatArray,
    variances: FloatArray,
    *,
    tol: float,
    max_iter: int,
) -> tuple[FloatArray, FloatArray, int]:
    r = loadings.shape[1]
    identity = np.eye(r)
    b = loadings.copy()
    psi = np.maximum(variances.copy(), 1e-10)
    for iteration in range(1, max_iter + 1):
        inv_psi_b = b / psi[:, None]
        inv_sigma_b = inv_psi_b @ safe_pinv(identity + b.T @ inv_psi_b)
        eff = inv_sigma_b.T @ covariance @ inv_sigma_b + identity - b.T @ inv_sigma_b
        ezf = covariance @ inv_sigma_b
        b_new = ezf @ safe_pinv(eff)
        psi_new = np.diag(covariance - b_new @ inv_sigma_b.T @ covariance)
        psi_new = np.maximum(psi_new, 1e-10)
        error = np.linalg.norm(b_new - b, ord="fro") + np.linalg.norm(psi_new - psi)
        b, psi = b_new, psi_new
        if error <= tol:
            return b, psi, iteration
    warnings.warn("Bai-Li ML iteration reached max_iter without convergence", stacklevel=2)
    return b, psi, max_iter


def _weighted_factor_scores(data: FloatArray, loadings: FloatArray, variances: FloatArray) -> FloatArray:
    inv_weighted = loadings / np.maximum(variances[:, None], 1e-12)
    normal = loadings.T @ inv_weighted
    return data @ inv_weighted @ safe_pinv(normal)


def _prewhiten_regression(
    y: FloatArray,
    factors: FloatArray,
    coefficients: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    order = coefficients.size
    target = y[order:].copy()
    design = factors[order:].copy()
    for lag, rho in enumerate(coefficients, start=1):
        target -= rho * y[order - lag : -lag]
        design -= rho * factors[order - lag : -lag]
    return target, design


def _gls_refine(
    data: FloatArray,
    loadings: FloatArray,
    factors: FloatArray,
    variances: FloatArray,
    *,
    ar_order: int | None,
    iterations: int,
    tol: float,
) -> tuple[FloatArray, FloatArray, FloatArray, list[FloatArray], int]:
    n = data.shape[1]
    u = loadings.copy()
    f = factors.copy()
    psi = variances.copy()
    rhos: list[FloatArray] = [np.array([0.0])] * n
    for iteration in range(1, iterations + 1):
        old = u.copy()
        residuals = data - f @ u.T
        next_rhos: list[FloatArray] = []
        for i in range(n):
            if ar_order is None:
                _, coefficients = select_ar_order_aic(residuals[:, i], max_order=5)
            else:
                if ar_order < 1:
                    raise ValueError("ar_order must be >= 1 or None")
                from ._utils import ar_coefficients

                coefficients = ar_coefficients(residuals[:, i], ar_order)
            target, design = _prewhiten_regression(data[:, i], f, coefficients)
            coef, *_ = np.linalg.lstsq(design, target, rcond=None)
            u[i] = coef
            next_rhos.append(coefficients)
        f = _weighted_factor_scores(data, u, psi)
        residuals = data - f @ u.T
        psi = np.maximum(np.mean(residuals**2, axis=0), 1e-10)
        rhos = next_rhos
        if np.linalg.norm(u - old, ord="fro") <= tol:
            return u, f, psi, rhos, iteration
    return u, f, psi, rhos, iterations



def _kalman_smooth_random_walk_factors(
    observations: FloatArray,
    loadings: FloatArray,
    rho: FloatArray,
    variances: FloatArray,
    factor_innovation_cov: FloatArray,
) -> FloatArray:
    """RTS smoother for the state [f_t, f_{t-1}] used by the source ML-EM routine."""
    r = loadings.shape[1]
    state_dim = 2 * r
    transition = np.zeros((state_dim, state_dim), dtype=np.float64)
    transition[:r, :r] = np.eye(r)
    transition[r:, :r] = np.eye(r)
    process_cov = np.zeros((state_dim, state_dim), dtype=np.float64)
    process_cov[:r, :r] = factor_innovation_cov
    observation_matrix = np.column_stack([loadings, -rho[:, None] * loadings])
    observation_cov = np.diag(np.maximum(variances, 1e-10))

    n_steps = observations.shape[0]
    filtered_mean = np.zeros((n_steps, state_dim), dtype=np.float64)
    filtered_cov = np.zeros((n_steps, state_dim, state_dim), dtype=np.float64)
    predicted_mean = np.zeros_like(filtered_mean)
    predicted_cov = np.zeros_like(filtered_cov)
    mean = np.zeros(state_dim, dtype=np.float64)
    covariance = np.diag(np.r_[np.ones(r), np.zeros(r)])
    identity = np.eye(state_dim)

    for idx, observation in enumerate(observations):
        mean_pred = transition @ mean
        cov_pred = transition @ covariance @ transition.T + process_cov
        innovation_cov = observation_matrix @ cov_pred @ observation_matrix.T + observation_cov
        gain = cov_pred @ observation_matrix.T @ safe_pinv(innovation_cov)
        innovation = observation - observation_matrix @ mean_pred
        mean = mean_pred + gain @ innovation
        # Joseph form is more stable than the one-sided covariance update.
        kh = identity - gain @ observation_matrix
        covariance = kh @ cov_pred @ kh.T + gain @ observation_cov @ gain.T
        covariance = (covariance + covariance.T) / 2.0
        predicted_mean[idx] = mean_pred
        predicted_cov[idx] = cov_pred
        filtered_mean[idx] = mean
        filtered_cov[idx] = covariance

    smoothed_mean = filtered_mean.copy()
    smoothed_cov = filtered_cov.copy()
    for idx in range(n_steps - 2, -1, -1):
        smoother_gain = filtered_cov[idx] @ transition.T @ safe_pinv(predicted_cov[idx + 1])
        smoothed_mean[idx] = filtered_mean[idx] + smoother_gain @ (
            smoothed_mean[idx + 1] - predicted_mean[idx + 1]
        )
        smoothed_cov[idx] = filtered_cov[idx] + smoother_gain @ (
            smoothed_cov[idx + 1] - predicted_cov[idx + 1]
        ) @ smoother_gain.T
    return smoothed_mean


def _ml_em_refine(
    data: FloatArray,
    loadings: FloatArray,
    factors: FloatArray,
    variances: FloatArray,
    rho_initial: FloatArray,
    *,
    tol: float,
    max_iter: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, int]:
    """State-space EM refinement corresponding to the R package's ML-EM construction."""
    t, n = data.shape
    r = loadings.shape[1]
    u = loadings.copy()
    psi = np.maximum(variances.copy(), 1e-10)
    rho = np.clip(rho_initial.copy(), -0.98, 0.98)
    factor_innovation_cov = np.cov(np.diff(factors, axis=0), rowvar=False, ddof=1)
    factor_innovation_cov = np.atleast_2d(factor_innovation_cov)
    factor_innovation_cov += 1e-8 * np.eye(r)

    for iteration in range(1, max_iter + 1):
        observations = data[1:] - data[:-1] * rho[None, :]
        states = _kalman_smooth_random_walk_factors(
            observations, u, rho, psi, factor_innovation_cov
        )
        f_now = states[:, :r]
        f_lag = states[:, r:]
        state_cov = np.cov(states, rowvar=False, ddof=1)
        state_cov = np.atleast_2d(state_cov)
        v00 = state_cov[:r, :r]
        v01 = state_cov[:r, r:]
        v11 = state_cov[r:, r:]

        u_new = u.copy()
        rho_new = rho.copy()
        psi_new = psi.copy()
        for i in range(n):
            ri = rho[i]
            moment = v00 - ri * v01 - ri * v01.T + ri**2 * v11
            transformed_factor = f_now - ri * f_lag
            transformed_x = data[1:, i] - ri * data[:-1, i]
            rhs = np.mean(transformed_factor * transformed_x[:, None], axis=0)
            u_new[i] = safe_pinv(moment) @ rhs

            ui = u_new[i]
            cross01 = float(ui @ v01 @ ui)
            lag_var = float(ui @ v11 @ ui)
            numerator = np.sum(
                data[1:, i] * data[:-1, i]
                - data[1:, i] * (f_lag @ ui)
                - data[:-1, i] * (f_now @ ui)
                + cross01
            )
            denominator = np.sum(
                data[:-1, i] ** 2 - 2.0 * data[:-1, i] * (f_lag @ ui) + lag_var
            )
            if abs(denominator) > 1e-12:
                rho_new[i] = np.clip(numerator / denominator, -0.98, 0.98)

            ri_new = rho_new[i]
            z = data[1:, i] - ri_new * data[:-1, i]
            fitted_now = f_now @ ui
            fitted_lag = f_lag @ ui
            quadratic = (
                z**2
                - 2.0 * z * fitted_now
                + 2.0 * ri_new * z * fitted_lag
                + float(ui @ v00 @ ui)
                - 2.0 * ri_new * float(ui @ v01 @ ui)
                + ri_new**2 * float(ui @ v11 @ ui)
            )
            psi_new[i] = max(float(np.mean(quadratic)), 1e-10)

        psi_lag = v01 @ safe_pinv(v11)
        innovations = f_now - f_lag @ psi_lag.T
        factor_innovation_cov = np.cov(innovations, rowvar=False, ddof=1)
        factor_innovation_cov = np.atleast_2d(factor_innovation_cov) + 1e-8 * np.eye(r)
        error = (
            np.linalg.norm(u_new - u, ord="fro") ** 2
            + np.linalg.norm(rho_new - rho) ** 2
            + np.linalg.norm(psi_new - psi) ** 2
        )
        u, rho, psi = u_new, rho_new, psi_new
        if error <= tol:
            break
    f = _weighted_factor_scores(data, u, psi)
    return u, f, psi, rho, iteration

def m2_mle(
    x: ArrayLike,
    r: int,
    *,
    scale: bool = False,
    method: str = "ML",
    tol: float = 1e-6,
    ar_order: int | None = 1,
    max_iter: int = 500,
) -> FactorResult:
    """Bai-Li ML/QML and serial-correlation refinements.

    ``ML`` and ``QML`` port the original EM-like Bai-Li loading/variance update.
    ``ML-GLS`` performs one AR-prewhitened loading update; ``ML-ITE`` iterates
    it. ``ML-EM`` uses a Kalman/RTS smoother for the source model with state
    ``[f_t, f_{t-1}]`` and AR(1) idiosyncratic errors.
    """
    if r < 1:
        raise ValueError("r must be positive")
    data = center_scale(x, scale=scale)
    t, n = data.shape
    if r > min(t, n):
        raise ValueError("r exceeds the rank bound")
    method = method.upper()
    valid = {"ML", "QML", "ML-GLS", "ML-ITE", "ML-EM"}
    if method not in valid:
        raise ValueError(f"unknown ML method: {method!r}")

    covariance = data.T @ data / t
    initial_u, _, initial_psi = _pca_ml_initial(data, r)
    u, psi, ml_iterations = _bai_li_ml(
        covariance,
        initial_u,
        initial_psi,
        tol=np.sqrt(tol),
        max_iter=max_iter,
    )
    f = _weighted_factor_scores(data, u, psi)
    e = data - f @ u.T
    metadata: dict[str, object] = {
        "method": method,
        "idiosyncratic_variance": psi,
        "ml_iterations": ml_iterations,
    }
    if method in {"ML", "QML"}:
        return FactorResult(f, u, e, metadata=metadata)

    if method == "ML-EM":
        if ar_order != 1:
            raise ValueError("ML-EM is defined for ar_order=1")
        # Match the source workflow: obtain AR(1) values from one GLS pass, then
        # use the state-space [f_t, f_{t-1}] EM refinement.
        u_gls, f_gls, psi_gls, rhos, _ = _gls_refine(
            data, u, f, psi, ar_order=1, iterations=1, tol=tol
        )
        rho_initial = np.array([coef[0] for coef in rhos], dtype=np.float64)
        u, f, psi, rho, dynamic_iterations = _ml_em_refine(
            data,
            u_gls,
            f_gls,
            psi_gls,
            rho_initial,
            tol=tol,
            max_iter=min(max_iter, 50),
        )
        rhos_out: object = rho
    else:
        refinements = 1 if method == "ML-GLS" else min(max_iter, 10)
        u, f, psi, rhos, dynamic_iterations = _gls_refine(
            data,
            u,
            f,
            psi,
            ar_order=ar_order,
            iterations=refinements,
            tol=tol,
        )
        rhos_out = rhos
    e = data - f @ u.T
    metadata.update(
        {
            "idiosyncratic_variance": psi,
            "rho": rhos_out,
            "dynamic_iterations": dynamic_iterations,
        }
    )
    return FactorResult(f, u, e, metadata=metadata)


def _gmm_initial(
    data: FloatArray,
    r: int,
    initial: str,
    tol: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    pca = m2_pca(data, r, center=True, method="PCA")
    if initial.upper() == "PCA":
        loadings = pca.loadings
        factors = pca.factors
        residuals = pca.residuals
        variances = np.mean(residuals**2, axis=0)
        return loadings, factors, residuals, variances
    if initial.upper() != "MLE":
        raise ValueError("initial must be 'PCA' or 'MLE'")
    mle = m2_mle(data, r, method="ML", tol=tol)
    variances = np.asarray(mle.metadata["idiosyncratic_variance"], dtype=np.float64)
    return mle.loadings, mle.factors, mle.residuals, variances


def _spectral_inverse(weight: FloatArray, delta: float) -> FloatArray:
    values, vectors = eigh_desc(weight)
    keep = values > delta
    if not np.any(keep):
        # A hard failure is more useful than returning an all-zero GMM criterion.
        keep[np.argmax(values)] = True
    return (vectors[:, keep] / values[keep][None, :]) @ vectors[:, keep].T


def m2_gmm(
    x: ArrayLike,
    r: int,
    *,
    kappa: float = 0.0,
    sigma_e: ArrayLike | None = None,
    initial: str = "PCA",
    weight_diagonal: bool = False,
    identity: bool = False,
    delta: float | None = None,
    tol: float = 1e-6,
) -> FactorResult:
    """Fan-Zhong generalized moment estimator using first/second moments."""
    raw = as_2d_float(x)
    data = center_scale(raw)
    t, n = data.shape
    if r < 1 or r > min(t, n):
        raise ValueError("invalid factor count r")
    init_u, _, init_e, init_var = _gmm_initial(data, r, initial, tol)
    variances = init_var if sigma_e is None else np.asarray(sigma_e, dtype=np.float64).reshape(-1)
    if variances.size != n:
        raise ValueError("sigma_e must have length n")
    if identity:
        variances = np.full(n, np.mean(variances))

    v1 = raw.mean(axis=0)[:, None]
    v2 = data.T @ data / t - np.diag(variances)
    v = np.column_stack([v1, v2])
    projector = np.eye(n) - init_u @ safe_pinv(init_u.T @ init_u) @ init_u.T
    weight = np.zeros((n + 1, n + 1), dtype=np.float64)
    for row_raw, row in zip(raw, data, strict=True):
        vi = np.column_stack([row_raw[:, None], np.outer(row, row) - np.diag(variances)])
        weight += vi.T @ projector @ vi
    weight /= t
    if weight_diagonal:
        weight = np.diag(np.diag(weight))
    threshold = 1.0 / np.log(max(t, 3)) if delta is None else float(delta)
    inv_weight = _spectral_inverse(weight, threshold)
    criterion = kappa * (data.T @ data / t) + v @ inv_weight @ v.T
    values, vectors = eigh_desc(criterion)
    loadings = vectors[:, :r] * np.sqrt(n)
    factors = data @ loadings / n
    residuals = data - factors @ loadings.T
    return FactorResult(
        factors,
        loadings,
        residuals,
        values,
        {"method": "M2-GMM", "weight": weight, "initial_residuals": init_e},
    )


def m3_gmm(
    x: ArrayLike,
    r: int,
    *,
    kappa: float = 0.0,
    initial: str = "PCA",
    weight_diagonal: bool = False,
    identity: bool = False,
    delta: float | None = None,
    tol: float = 1e-6,
) -> FactorResult:
    """Fan-Zhong generalized moment estimator augmented by third moments."""
    raw = as_2d_float(x)
    data = center_scale(raw)
    t, n = data.shape
    if r < 1 or r > min(t, n):
        raise ValueError("invalid factor count r")
    init_u, init_f, init_e, variances = _gmm_initial(data, r, initial, tol)
    skew_e = np.mean(init_e**3, axis=0)
    if identity:
        variances = np.full(n, np.mean(variances))
        skew_e = np.full(n, np.mean(skew_e))

    v1 = raw.mean(axis=0)[:, None]
    v2 = data.T @ data / t - np.diag(variances)
    v3 = (data * data).T @ data / t - np.diag(skew_e)
    v = np.column_stack([v1, v2, v3])
    projector = np.eye(n) - init_u @ safe_pinv(init_u.T @ init_u) @ init_u.T
    weight = np.zeros((2 * n + 1, 2 * n + 1), dtype=np.float64)
    for row_raw, row in zip(raw, data, strict=True):
        vi = np.column_stack(
            [
                row_raw[:, None],
                np.outer(row, row) - np.diag(variances),
                np.outer(row * row, row) - np.diag(skew_e),
            ]
        )
        weight += vi.T @ projector @ vi
    weight /= t
    if weight_diagonal:
        weight = np.diag(np.diag(weight))
    threshold = 1.0 / np.log(max(t, 3)) if delta is None else float(delta)
    inv_weight = _spectral_inverse(weight, threshold)
    criterion = kappa * (data.T @ data / t) + v @ inv_weight @ v.T
    values, vectors = eigh_desc(criterion)
    loadings = vectors[:, :r] * np.sqrt(n)
    factors = data @ loadings / n
    residuals = data - factors @ loadings.T
    return FactorResult(
        factors,
        loadings,
        residuals,
        values,
        {
            "method": "M3-GMM",
            "weight": weight,
            "initial_factor_skew": np.mean(init_f**3, axis=0),
        },
    )


def _gaussian_components(residual: FloatArray, rg: int, n: int) -> tuple[FloatArray, FloatArray]:
    if rg == 0:
        return np.empty((n, 0)), np.empty((residual.shape[0], 0))
    matrix = m2m(residual) / n**2
    _, vectors = eigh_desc(matrix)
    loadings = vectors[:, :rg] * np.sqrt(n)
    factors = residual @ loadings / n
    return loadings, factors


def m3_als(
    x: ArrayLike,
    *,
    rh: int,
    rg: int,
    scale: bool = False,
    gamma: tuple[float, float] = (0.0, 1.0),
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> FactorResult:
    """Third-order alternating least-squares HFA estimator."""
    data = center_scale(x, scale=scale)
    t, n = data.shape
    del t
    if rh < 0 or rg < 0 or rh + rg < 1 or rh + rg > n:
        raise ValueError("rh and rg must be non-negative and sum to 1..n")
    g2, g3 = map(float, gamma)

    if rh == 0:
        covariance = np.cov(data, rowvar=False, ddof=1) / n
        values, vectors = eigh_desc(covariance)
        loadings = vectors[:, :rg] * np.sqrt(n)
        factors = data @ loadings / n
        residuals = data - factors @ loadings.T
        return FactorResult(factors, loadings, residuals, values, {"iterations": 0})

    joint = g2 * m2m(data) / n**2 + g3 * m3m(data) / n**3
    initial_values, vectors = eigh_desc(joint)
    uh = vectors[:, :rh] * np.sqrt(n)
    fh = data @ uh / n
    if rg == 0:
        residuals = data - fh @ uh.T
        return FactorResult(fh, uh, residuals, initial_values, {"iterations": 0})

    ug, _ = _gaussian_components(data - fh @ uh.T, rg, n)
    for iteration in range(1, max_iter + 1):
        fg = (data - fh @ uh.T) @ ug / n
        wh = data - fg @ ug.T
        joint = g2 * m2m(wh) / n**2 + g3 * m3m(wh) / n**3
        _, vectors_h = eigh_desc(joint)
        uh_new = align_columns(uh, vectors_h[:, :rh] * np.sqrt(n))
        fh_new = data @ uh_new / n
        ug_new, _ = _gaussian_components(data - fh_new @ uh_new.T, rg, n)
        ug_new = align_columns(ug, ug_new)
        error = np.linalg.norm(uh_new - uh, ord="fro") + np.linalg.norm(ug_new - ug, ord="fro")
        uh, fh, ug = uh_new, fh_new, ug_new
        if error <= tol:
            break
    else:
        warnings.warn("M3 ALS reached max_iter without convergence", stacklevel=2)
        iteration = max_iter
    loadings = np.column_stack([uh, ug])
    factors = data @ loadings / n
    residuals = data - factors @ loadings.T
    return FactorResult(factors, loadings, residuals, initial_values, {"iterations": iteration})


def m4_als(
    x: ArrayLike,
    *,
    rh: int,
    rg: int,
    scale: bool = False,
    gamma: tuple[float, float, float] = (0.0, 0.0, 1.0),
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> FactorResult:
    """Fourth-order alternating least-squares HFA estimator.

    The fourth-order term is divided by ``n**4`` both at initialization and in
    the ALS loop. The R source uses ``n**3`` only inside the loop, which is an
    internal inconsistency and is treated here as a typo.
    """
    data = center_scale(x, scale=scale)
    t, n = data.shape
    if rh < 0 or rg < 0 or rh + rg < 1 or rh + rg > n:
        raise ValueError("rh and rg must be non-negative and sum to 1..n")
    g2, g3, g4 = map(float, gamma)

    if rh == 0:
        covariance = np.cov(data, rowvar=False, ddof=1) / n
        values, vectors = eigh_desc(covariance)
        loadings = vectors[:, :rg] * np.sqrt(n)
        factors = data @ loadings / n
        residuals = data - factors @ loadings.T
        return FactorResult(factors, loadings, residuals, values, {"iterations": 0})

    covariance = data.T @ data / t
    joint = (
        g2 * m2m(data) / n**2
        + g3 * m3m(data) / n**3
        + g4 * c4m(data, covariance) / n**4
    )
    initial_values, vectors = eigh_desc(joint)
    uh = vectors[:, :rh] * np.sqrt(n)
    fh = data @ uh / n
    if rg == 0:
        residuals = data - fh @ uh.T
        return FactorResult(fh, uh, residuals, initial_values, {"iterations": 0})

    ug, _ = _gaussian_components(data - fh @ uh.T, rg, n)
    for iteration in range(1, max_iter + 1):
        fg = (data - fh @ uh.T) @ ug / n
        wh = data - fg @ ug.T
        cov_h = wh.T @ wh / wh.shape[0]
        joint = (
            g2 * m2m(wh) / n**2
            + g3 * m3m(wh) / n**3
            + g4 * c4m(wh, cov_h) / n**4
        )
        _, vectors_h = eigh_desc(joint)
        uh_new = align_columns(uh, vectors_h[:, :rh] * np.sqrt(n))
        fh_new = data @ uh_new / n
        ug_new, _ = _gaussian_components(data - fh_new @ uh_new.T, rg, n)
        ug_new = align_columns(ug, ug_new)
        error = np.linalg.norm(uh_new - uh, ord="fro") + np.linalg.norm(ug_new - ug, ord="fro")
        uh, fh, ug = uh_new, fh_new, ug_new
        if error <= tol:
            break
    else:
        warnings.warn("M4 ALS reached max_iter without convergence", stacklevel=2)
        iteration = max_iter
    loadings = np.column_stack([uh, ug])
    factors = data @ loadings / n
    residuals = data - factors @ loadings.T
    return FactorResult(factors, loadings, residuals, initial_values, {"iterations": iteration})


# Compatibility aliases for the R-style function names.
M2_pca = m2_pca
M2_mle = m2_mle
M2_gmm = m2_gmm
M3_gmm = m3_gmm
M3_als = m3_als
M4_als = m4_als
