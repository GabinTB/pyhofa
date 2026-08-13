"""Higher-moment portfolio construction from PC or IC factor models."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from sklearn.decomposition import FastICA

from ._types import PortfolioResult
from ._utils import center_scale, eigh_desc
from .moments import (
    direct_nonlinear_shrinkage,
    expected_utility_objective,
    fourth_cumulant_matrix,
    linear_identity_shrinkage,
    modified_var_objective,
    portfolio_moments_independent,
    portfolio_moments_matrix,
    third_moment_matrix,
    variable_moment,
)
from .selection import m2_select

FloatArray = NDArray[np.float64]


def _covariance(data: FloatArray, adjustment: str) -> FloatArray:
    adjustment = adjustment.upper()
    if adjustment == "NONE":
        return np.cov(data, rowvar=False, ddof=1)
    if adjustment == "LI":
        return linear_identity_shrinkage(data)
    if adjustment == "DNL":
        return direct_nonlinear_shrinkage(data)
    raise ValueError("covariance_adjustment must be 'NONE', 'LI' or 'DNL'")


def _select_r(data: FloatArray, r: int | None, rmax: int, method: str) -> int:
    if r is not None:
        if r < 1 or r > min(data.shape):
            raise ValueError("invalid r")
        return int(r)
    mapping = {"IC3": "BN-IC3", "ED": "ON"}
    selected = m2_select(data, rmax=rmax, method=mapping.get(method.upper(), method.upper()))
    return max(1, selected.n_factors)


def _optimize_weights(
    n: int,
    objective,
    *,
    shortselling: bool,
) -> tuple[FloatArray, float, bool, str]:
    lower = -1.0 if shortselling else 0.0
    bounds = [(lower, 1.0)] * n
    initial = np.full(n, 1.0 / n)
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    weights = np.asarray(result.x, dtype=np.float64)
    total = weights.sum()
    if abs(total) > np.finfo(float).eps:
        weights /= total
    return weights, float(objective(weights)), bool(result.success), str(result.message)


def portfolio_ic(
    x: ArrayLike,
    *,
    r: int | None = None,
    rmax: int = 10,
    factor_selection: str = "ER",
    objective: str = "MVaR",
    alpha: float = 0.01,
    gamma: float = 10.0,
    covariance_adjustment: str = "NONE",
    shortselling: bool = True,
    random_state: int | None = 0,
) -> PortfolioResult:
    """Independent-component portfolio with parsimonious higher moments."""
    data = center_scale(x)
    t, n = data.shape
    del t
    n_factors = _select_r(data, r, rmax, factor_selection)
    cov = _covariance(data, covariance_adjustment)

    if n_factors == 1:
        _, vectors = eigh_desc(cov)
        direction = vectors[:, 0]
        factor = data @ direction
        std = np.sqrt(np.mean((factor - factor.mean()) ** 2))
        factor = factor / max(std, np.finfo(float).eps)
        loadings = np.linalg.lstsq(factor[:, None], data, rcond=None)[0].T
        factors = factor[:, None]
    else:
        ica = FastICA(
            n_components=n_factors,
            whiten="unit-variance",
            fun="cube",
            max_iter=2000,
            tol=1e-7,
            random_state=random_state,
        )
        factors = ica.fit_transform(data)
        loadings = np.asarray(ica.mixing_, dtype=np.float64)
    residuals = data - factors @ loadings.T

    m2f = variable_moment(factors, 2)
    m3f = variable_moment(factors, 3)
    m4f = variable_moment(factors, 4)
    factor_moments = (m2f, m3f, m4f)
    m2e = np.diag(_covariance(residuals, covariance_adjustment))
    m3e = variable_moment(residuals, 3)
    m4e = variable_moment(residuals, 4)
    idio_moments = (m2e, m3e, m4e)

    objective_upper = objective.upper()

    def loss(w: FloatArray) -> float:
        moments = portfolio_moments_independent(w, factor_moments, idio_moments, loadings)
        if objective_upper == "MVAR":
            return modified_var_objective(moments, alpha=alpha)
        if objective_upper == "EU":
            return expected_utility_objective(moments, gamma=gamma)
        raise ValueError("objective must be 'MVaR' or 'EU'")

    weights, value, success, message = _optimize_weights(
        n, loss, shortselling=shortselling
    )
    portfolio_moments = portfolio_moments_independent(
        weights, factor_moments, idio_moments, loadings
    )
    return PortfolioResult(
        weights,
        value,
        n_factors,
        factor_moments,
        idio_moments,
        portfolio_moments,
        success,
        message,
    )


def portfolio_pc(
    x: ArrayLike,
    *,
    r: int | None = None,
    rmax: int = 10,
    factor_selection: str = "ER",
    objective: str = "MVaR",
    alpha: float = 0.01,
    gamma: float = 10.0,
    covariance_adjustment: str = "NONE",
    shortselling: bool = True,
) -> PortfolioResult:
    """Principal-component portfolio with full factor co-moments."""
    data = center_scale(x)
    _, n = data.shape
    n_factors = _select_r(data, r, rmax, factor_selection)
    cov = _covariance(data, covariance_adjustment)
    _, vectors = eigh_desc(cov)
    loadings = vectors[:, :n_factors] * np.sqrt(n)
    factors = data @ loadings / n
    residuals = data - factors @ loadings.T

    m2f = np.cov(factors, rowvar=False, ddof=1)
    if n_factors == 1:
        m2f = np.asarray(m2f).reshape(1, 1)
    m3f = third_moment_matrix(factors)
    c4f = fourth_cumulant_matrix(factors)
    factor_moments = (m2f, m3f, c4f)
    m2e = np.diag(_covariance(residuals, covariance_adjustment))
    m3e = variable_moment(residuals, 3)
    m4e = variable_moment(residuals, 4)
    idio_moments = (m2e, m3e, m4e)

    objective_upper = objective.upper()

    def loss(w: FloatArray) -> float:
        moments = portfolio_moments_matrix(w, factor_moments, idio_moments, loadings)
        if objective_upper == "MVAR":
            return modified_var_objective(moments, alpha=alpha)
        if objective_upper == "EU":
            return expected_utility_objective(moments, gamma=gamma)
        raise ValueError("objective must be 'MVaR' or 'EU'")

    weights, value, success, message = _optimize_weights(
        n, loss, shortselling=shortselling
    )
    portfolio_moments = portfolio_moments_matrix(
        weights, factor_moments, idio_moments, loadings
    )
    return PortfolioResult(
        weights,
        value,
        n_factors,
        factor_moments,
        idio_moments,
        portfolio_moments,
        success,
        message,
    )


Portfolio_IC = portfolio_ic
Portfolio_PC = portfolio_pc
