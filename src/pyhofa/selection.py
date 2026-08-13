"""Factor-number selection criteria."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from scipy.stats import kurtosis, skew

from ._types import SelectionResult
from ._utils import center_scale, eigh_desc, validate_rmax
from .moments import c4m, jmca, m3m
from .simulation import sgt_moment, sgt_rvs

FloatArray = NDArray[np.float64]


def _ratio_selector(
    eigenvalues: FloatArray,
    rmax: int,
    *,
    modified: bool,
    zero_anchor: float,
    growth: bool,
) -> int:
    """Port the ER/GR family while handling zero-factor candidates safely."""
    ev = np.asarray(eigenvalues, dtype=np.float64)
    kmax = validate_rmax(rmax, ev.size)
    eps = np.finfo(float).tiny

    if not growth:
        if modified:
            sequence = np.r_[zero_anchor, ev]
            ratios = sequence[: kmax + 1] / np.maximum(sequence[1 : kmax + 2], eps)
            return int(np.nanargmax(ratios))
        ratios = ev[:kmax] / np.maximum(ev[1 : kmax + 1], eps)
        return int(np.nanargmax(ratios) + 1)

    # Ahn-Horenstein growth-ratio construction based on residual eigenvalue sums.
    # V_r is the unexplained spectral mass after retaining r factors.
    tail = np.array([np.sum(ev[r:]) for r in range(kmax + 2)], dtype=np.float64)
    tail = np.maximum(tail, eps)
    gains = np.log(tail[:-1]) - np.log(tail[1:])
    ratios = gains[:-1] / np.maximum(gains[1:], eps)
    if modified:
        # The modified zero-factor anchor competes with the first factor through
        # the same device used by the original package.
        zero_gain = np.log(max(np.sum(ev) + zero_anchor, eps)) - np.log(max(np.sum(ev), eps))
        denom = max(gains[0], eps)
        candidates = np.r_[zero_gain / denom, ratios[:kmax]]
        return int(np.nanargmax(candidates))
    return int(np.nanargmax(ratios[:kmax]) + 1)


def _onatski(eigenvalues: FloatArray, rmax: int) -> int:
    ev = np.asarray(eigenvalues, dtype=np.float64)
    kmax = min(int(rmax), ev.size - 5)
    if kmax < 1:
        raise ValueError("not enough eigenvalues for the Onatski selector")
    j = kmax + 1  # 1-based index in the source algorithm
    for _ in range(100):
        previous = j
        start = max(1, j) - 1
        if start + 5 > ev.size:
            start = ev.size - 5
        y = ev[start : start + 5]
        x_idx = np.arange(start, start + 5, dtype=np.float64)
        x = np.power(np.maximum(x_idx, 0.0), 2.0 / 3.0)
        slope = np.polyfit(x, y, deg=1)[0]
        delta = 2.0 * abs(float(slope))
        gaps = ev[:kmax] - ev[1 : kmax + 1]
        hits = np.flatnonzero(gaps > delta)
        if hits.size == 0:
            break
        j = int(hits.max() + 2)  # convert gap index into source's j=r+1
        if j == previous:
            break
    return max(0, j - 1)


def _act(correlation_eigenvalues: FloatArray, n_obs: int, n_vars: int, rmax: int) -> int:
    lamb = np.asarray(correlation_eigenvalues, dtype=np.float64)
    kmax = validate_rmax(rmax, lamb.size)
    pp = min(kmax + 2, lamb.size - 1)
    mz = np.zeros(pp, dtype=np.float64)
    q = 0.75
    for kk in range(pp):
        # R uses lambdaZ[-seq(1, kk)] with 1-based indexing.
        remaining = np.delete(lamb, np.arange(kk + 1))
        z0 = q * lamb[kk] + (1.0 - q) * lamb[kk + 1]
        diffs = np.r_[remaining - lamb[kk], z0 - lamb[kk]]
        y0 = remaining.size / max(n_obs - 1, 1)
        inv_mean = np.mean(1.0 / diffs[np.abs(diffs) > np.finfo(float).eps])
        mz[kk] = -(1.0 - y0) / lamb[kk] + y0 * inv_mean
    temp = (-1.0 / mz)[1:] - 1.0 - np.sqrt(n_vars / max(n_obs - 1, 1))
    hits = np.flatnonzero(temp[:kmax] > 0.0)
    # Source adds one to the largest retained adjusted eigenvalue index.
    return min(int(hits.max() + 2) if hits.size else 1, kmax)


def m2_select(
    x: ArrayLike,
    *,
    scale: bool = False,
    rmax: int = 8,
    method: str = "ER",
    modified: bool = False,
) -> SelectionResult:
    """Select factor count from second-order information.

    Supported methods are ``ER``, ``GR``, ``BN-IC3``, ``BN-PC3``, ``BIC3``,
    ``ON`` and ``ACT``.

    Unlike the R implementation, this function never silently transposes the
    input when ``n > t``. Rows are always observations and columns variables.
    """
    data = center_scale(x, scale=scale)
    t, n = data.shape
    method = method.upper()
    covariance = data.T @ data / t
    ev, evec = eigh_desc(covariance)
    scaled_ev = ev / n
    kmax = validate_rmax(rmax, ev.size)
    m = min(n, t)

    if method in {"ER", "GR"}:
        zero = float(np.sum(scaled_ev) / np.log(max(m, 2))) if modified else 0.0
        selected = _ratio_selector(
            scaled_ev,
            kmax,
            modified=modified,
            zero_anchor=zero,
            growth=method == "GR",
        )
        return SelectionResult(selected, scaled_ev)

    if method in {"BN-IC3", "BN-PC3", "BIC3"}:
        u = evec[:, :kmax]
        sigma_hat = np.sum((data @ u @ u.T - data) ** 2) / (n * t)
        criteria: dict[str, list[float]] = {"BN-IC3": [], "BN-PC3": [], "BIC3": []}
        for r in range(1, kmax + 1):
            ur = u[:, :r]
            v_r = np.sum((data @ ur @ ur.T - data) ** 2) / (n * t)
            bic_penalty = r * sigma_hat * ((n + t - r) * np.log(n * t)) / (n * t)
            bn_penalty = r * np.log((n * t) / (n + t)) * (n + t) / (n * t)
            criteria["BIC3"].append(v_r + bic_penalty)
            criteria["BN-IC3"].append(np.log(max(v_r, np.finfo(float).tiny)) + bn_penalty)
            criteria["BN-PC3"].append(v_r + sigma_hat * bn_penalty)
        selected = int(np.argmin(criteria[method]) + 1)
        return SelectionResult(selected, scaled_ev, metadata={"criterion": criteria[method]})

    if method in {"ON", "ED"}:
        selected = _onatski(ev, kmax)
        return SelectionResult(selected, scaled_ev)

    if method == "ACT":
        std = np.sqrt(np.diag(covariance))
        if np.any(std <= np.finfo(float).eps):
            raise ValueError("ACT cannot be used with a zero-variance variable")
        corr = covariance / np.outer(std, std)
        corr_ev, _ = eigh_desc(corr)
        selected = _act(corr_ev, t, n, kmax)
        return SelectionResult(selected, corr_ev)

    raise ValueError(f"unknown M2 selection method: {method!r}")


def _residual_gaussian_count(
    data: FloatArray,
    vectors: FloatArray,
    rh: int,
    rmax: int,
    *,
    growth: bool,
) -> int:
    if rh > 0:
        u = vectors[:, :rh]
        residual = data - data @ u @ u.T
    else:
        residual = data
    covariance = np.cov(residual, rowvar=False, ddof=1)
    ev, _ = eigh_desc(covariance)
    ev = np.clip(ev / data.shape[1], 0.0, None)
    anchor = float(np.sum(ev) / np.log(max(min(data.shape), 2)))
    return _ratio_selector(
        ev,
        rmax,
        modified=True,
        zero_anchor=anchor,
        growth=growth,
    )


def _fit_sgt_shape(data: FloatArray) -> tuple[float, float, float]:
    standardized = center_scale(data, scale=True)
    target_skew = float(np.sqrt(np.mean(skew(standardized, axis=0, bias=True) ** 2)))
    target_kurt = float(
        np.sqrt(np.mean(kurtosis(standardized, axis=0, fisher=False, bias=True) ** 2))
    )

    def objective(theta: FloatArray) -> float:
        lam, p, q = theta
        try:
            m3 = sgt_moment(3, sigma=1.0, lam=lam, p=p, q=q)
            m4 = sgt_moment(4, sigma=1.0, lam=lam, p=p, q=q)
        except (ValueError, FloatingPointError):
            return 1e12
        return float((m3 - target_skew) ** 2 + (m4 - target_kurt) ** 2)

    fit = minimize(
        objective,
        x0=np.array([0.0, 2.0, 10.0]),
        method="L-BFGS-B",
        bounds=[(-0.95, 0.95), (1.0, 3.0), (3.001, 200.0)],
    )
    if not fit.success:
        warnings.warn(f"SGT calibration did not fully converge: {fit.message}", stacklevel=2)
    return float(fit.x[0]), float(fit.x[1]), float(fit.x[2])


def _jjr_selector(
    data: FloatArray,
    rmax: int,
    simulations: int,
    order: int,
    rng: np.random.Generator,
) -> int:
    if simulations < 1:
        raise ValueError("simulations must be positive")
    lam, p, q = _fit_sgt_shape(data)
    t, n = data.shape
    gamma = (0.0, 1.0, 0.0) if order == 3 else (1.0, 1.0, 1.0)
    maxima = np.empty(simulations)
    for i in range(simulations):
        simulated = sgt_rvs((t, n), lam=lam, p=p, q=q, rng=rng)
        simulated = center_scale(simulated, scale=True)
        eigenvalues, _ = jmca(simulated, rmax, gamma=gamma)
        maxima[i] = np.max(eigenvalues)
    threshold = float(np.mean(maxima))
    observed, _ = jmca(center_scale(data, scale=True), rmax, gamma=gamma)
    return int(np.sum(observed > threshold))


def m3_select(
    x: ArrayLike,
    *,
    scale: bool = False,
    rmax: int = 8,
    method: str = "GER3",
    modified: bool = False,
    simulations: int = 100,
    random_state: int | np.random.Generator | None = None,
) -> SelectionResult:
    """Select non-Gaussian and Gaussian factors from third-order information."""
    data = center_scale(x, scale=scale)
    t, n = data.shape
    method = method.upper()
    kmax = validate_rmax(rmax, min(n, t))

    if method in {"GER3", "GGR3"}:
        matrix = m3m(data) / t**2 / n**3
        raw_ev, vectors = eigh_desc(matrix)
        ev = np.sqrt(np.clip(raw_ev, 0.0, None))
        anchor = float(np.sum(ev) / max(t / n, np.finfo(float).eps)) if modified else 0.0
        rh = _ratio_selector(
            ev,
            kmax,
            modified=modified,
            zero_anchor=anchor,
            growth=method == "GGR3",
        )
        rg = _residual_gaussian_count(
            data, vectors, rh, kmax, growth=method == "GGR3"
        )
        return SelectionResult(rh + rg, ev, rh, rg)

    if method == "JJR3":
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        selected = _jjr_selector(data, kmax, simulations, 3, rng)
        return SelectionResult(selected)

    raise ValueError(f"unknown M3 selection method: {method!r}")


def m4_select(
    x: ArrayLike,
    *,
    scale: bool = False,
    rmax: int = 8,
    method: str = "GER4",
    modified: bool = False,
    simulations: int = 100,
    random_state: int | np.random.Generator | None = None,
) -> SelectionResult:
    """Select non-Gaussian and Gaussian factors from fourth-order information."""
    data = center_scale(x, scale=scale)
    t, n = data.shape
    method = method.upper()
    kmax = validate_rmax(rmax, min(n, t))

    if method in {"GER4", "GGR4"}:
        cov = data.T @ data / t
        raw_ev, vectors = eigh_desc(c4m(data, cov))
        ev = np.sqrt(np.clip(raw_ev / n**4, 0.0, None))
        anchor = float(np.sum(ev) / max(t / n, np.finfo(float).eps)) if modified else 0.0
        rh = _ratio_selector(
            ev,
            kmax,
            modified=modified,
            zero_anchor=anchor,
            growth=method == "GGR4",
        )
        rg = _residual_gaussian_count(
            data, vectors, rh, kmax, growth=method == "GGR4"
        )
        return SelectionResult(rh + rg, ev, rh, rg)

    if method == "JJR4":
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        selected = _jjr_selector(data, kmax, simulations, 4, rng)
        return SelectionResult(selected)

    raise ValueError(f"unknown M4 selection method: {method!r}")


# Compatibility aliases using legal Python identifiers.
M2_select = m2_select
M3_select = m3_select
M4_select = m4_select
