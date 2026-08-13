"""Synthetic data-generating processes used by the HOFA simulations."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import beta as beta_fn
from scipy.special import gamma as gamma_fn

from ._types import SimulationResult

FloatArray = NDArray[np.float64]


def _sgt_scale_constants(
    sigma: float,
    lam: float,
    p: float,
    q: float,
) -> tuple[float, float]:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not -1.0 < lam < 1.0:
        raise ValueError("lam must lie strictly between -1 and 1")
    if p <= 0:
        raise ValueError("p must be positive")

    if math.isinf(q):
        a = gamma_fn(3.0 / p) / gamma_fn(1.0 / p)
        b = gamma_fn(2.0 / p) / gamma_fn(1.0 / p)
        variance_term = (1.0 + 3.0 * lam**2) * a - 4.0 * lam**2 * b**2
        v = variance_term ** (-0.5)
        mean_shift = 2.0 * v * sigma * lam * b
        return float(v), float(mean_shift)

    if q <= 2.0 / p:
        raise ValueError("finite q must satisfy q > 2/p for a finite variance")
    b1 = beta_fn(1.0 / p, q)
    b2 = beta_fn(2.0 / p, q - 1.0 / p) / b1
    b3 = beta_fn(3.0 / p, q - 2.0 / p) / b1
    variance_term = (1.0 + 3.0 * lam**2) * b3 - 4.0 * lam**2 * b2**2
    v = q ** (-1.0 / p) * variance_term ** (-0.5)
    mean_shift = 2.0 * v * sigma * lam * q ** (1.0 / p) * b2
    return float(v), float(mean_shift)


def sgt_rvs(
    size: int | tuple[int, ...],
    *,
    mu: float = 0.0,
    sigma: float = 1.0,
    lam: float = 0.0,
    p: float = 2.0,
    q: float = math.inf,
    rng: np.random.Generator | int | None = None,
) -> FloatArray:
    """Draw from the variance-adjusted skewed generalized-t distribution.

    The parameterization matches the ``sgt`` R package convention used by the
    original HOFA simulations. ``q=np.inf`` gives the skewed generalized-error
    limiting family; ``lam=0, p=2, q=np.inf`` is standard Gaussian.
    """
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    v, mean_shift = _sgt_scale_constants(sigma, lam, p, q)
    signs = np.where(generator.random(size) < (1.0 + lam) / 2.0, 1.0, -1.0)
    if math.isinf(q):
        magnitude = generator.gamma(shape=1.0 / p, scale=1.0, size=size) ** (1.0 / p)
    else:
        u = generator.beta(1.0 / p, q, size=size)
        magnitude = (q * u / np.maximum(1.0 - u, np.finfo(float).tiny)) ** (1.0 / p)
    raw = signs * (1.0 + signs * lam) * v * sigma * magnitude
    return np.asarray(mu + raw - mean_shift, dtype=np.float64)


def sgt_moment(
    order: int,
    *,
    sigma: float = 1.0,
    lam: float = 0.0,
    p: float = 2.0,
    q: float = math.inf,
) -> float:
    """Central moment of the variance-adjusted SGT distribution."""
    if order < 0:
        raise ValueError("order must be non-negative")
    if not math.isinf(q) and q <= order / p:
        raise ValueError("requested moment does not exist")
    v, mean_shift = _sgt_scale_constants(sigma, lam, p, q)

    def raw_moment(j: int) -> float:
        if j == 0:
            return 1.0
        if math.isinf(q):
            radial = gamma_fn((j + 1.0) / p) / gamma_fn(1.0 / p)
        else:
            radial = q ** (j / p) * beta_fn((j + 1.0) / p, q - j / p) / beta_fn(
                1.0 / p, q
            )
        skew_factor = ((1.0 + lam) ** (j + 1) + (-1.0) ** j * (1.0 - lam) ** (j + 1)) / 2.0
        return float((v * sigma) ** j * radial * skew_factor)

    total = 0.0
    for j in range(order + 1):
        total += math.comb(order, j) * (-mean_shift) ** (order - j) * raw_moment(j)
    return float(total)


def _ar1_from_innovations(
    innovations: FloatArray,
    rho: float,
    *,
    burnin: int = 200,
    rng: np.random.Generator,
    innovation_factory: Callable[[int], ArrayLike] | None = None,
) -> FloatArray:
    if abs(rho) >= 1.0:
        raise ValueError("AR(1) coefficient must have absolute value below one")
    if burnin <= 0 or innovation_factory is None:
        out = np.empty_like(innovations)
        state = 0.0
        for i, innovation in enumerate(innovations):
            state = rho * state + innovation
            out[i] = state
        return out
    extra = np.asarray(innovation_factory(burnin), dtype=np.float64)
    all_innovations = np.concatenate([extra, innovations])
    out = np.empty_like(all_innovations)
    state = 0.0
    for i, innovation in enumerate(all_innovations):
        state = rho * state + innovation
        out[i] = state
    return out[burnin:]


def _as_factor_parameter(value: ArrayLike | float, k: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(k, float(array))
    array = array.reshape(-1)
    if array.size != k:
        raise ValueError(f"{name} must be scalar or have length k={k}")
    return array


def dgp1(
    n: int,
    t: int,
    k: int,
    *,
    factor_sigma: ArrayLike | float = 1.0,
    factor_lambda: ArrayLike | float = 0.8,
    factor_p: ArrayLike | float = 1.0,
    factor_q: ArrayLike | float = math.inf,
    error_sigma: float = 1.0,
    error_lambda: float = 0.0,
    error_p: float = 2.0,
    error_q: float = math.inf,
    alpha: float = 0.0,
    factor_ar: ArrayLike | float = 0.0,
    error_ar: float = 0.2,
    covariance_decay: float = 0.5,
    covariance_scale: float = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> SimulationResult:
    """Generate the original package's first factor-model DGP."""
    if min(n, t, k) <= 0:
        raise ValueError("n, t and k must be positive")
    del alpha  # Retained for source-API compatibility; the source code does not use it.
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    sig_f = _as_factor_parameter(factor_sigma, k, "factor_sigma")
    lam_f = _as_factor_parameter(factor_lambda, k, "factor_lambda")
    p_f = _as_factor_parameter(factor_p, k, "factor_p")
    q_f = _as_factor_parameter(factor_q, k, "factor_q")
    rho_f = _as_factor_parameter(factor_ar, k, "factor_ar")

    factors = np.empty((t, k), dtype=np.float64)
    for j in range(k):
        factory = lambda m, j=j: sgt_rvs(
            m,
            sigma=sig_f[j],
            lam=lam_f[j],
            p=p_f[j],
            q=q_f[j],
            rng=rng,
        )
        innovations = factory(t)
        factors[:, j] = _ar1_from_innovations(
            innovations, float(rho_f[j]), rng=rng, innovation_factory=factory
        )

    errors_raw = np.empty((t, n), dtype=np.float64)
    for j in range(n):
        factory = lambda m: sgt_rvs(
            m,
            sigma=error_sigma,
            lam=error_lambda,
            p=error_p,
            q=error_q,
            rng=rng,
        )
        innovations = factory(t)
        errors_raw[:, j] = _ar1_from_innovations(
            innovations, error_ar, rng=rng, innovation_factory=factory
        )

    loadings = rng.normal(size=(n, k))
    eigen_scale = covariance_scale * np.arange(1, n + 1, dtype=np.float64) ** (-covariance_decay)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(n, n)))
    error_transform = np.diag(np.sqrt(eigen_scale)) @ orthogonal.T
    errors = errors_raw @ error_transform
    x = factors @ loadings.T + errors
    return SimulationResult(x, loadings, factors, errors)


def dgp2(
    n: int,
    t: int,
    k: int,
    *,
    factor_sigma: ArrayLike | float = 1.0,
    factor_lambda: ArrayLike | float = 0.8,
    factor_p: ArrayLike | float = 1.0,
    factor_q: ArrayLike | float = math.inf,
    error_sigma: float = 1.0,
    error_lambda: float = 0.0,
    error_p: float = 2.0,
    error_q: float = math.inf,
    beta: float = 0.2,
    neighborhood: int | None = None,
    error_ar: float = 0.2,
    error_variance_range: tuple[float, float] = (1.0, 5.0),
    factor_ar: ArrayLike | float = 0.0,
    loading_strength: ArrayLike | float = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> SimulationResult:
    """Generate a cross-sectionally and serially dependent weak-factor panel.

    ``loading_strength`` is a Python extension useful for reproducing weak-factor
    experiments directly. A scalar rescales every factor loading column; a
    length-k vector controls each factor separately.
    """
    if min(n, t, k) <= 0:
        raise ValueError("n, t and k must be positive")
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    sig_f = _as_factor_parameter(factor_sigma, k, "factor_sigma")
    lam_f = _as_factor_parameter(factor_lambda, k, "factor_lambda")
    p_f = _as_factor_parameter(factor_p, k, "factor_p")
    q_f = _as_factor_parameter(factor_q, k, "factor_q")
    rho_f = _as_factor_parameter(factor_ar, k, "factor_ar")
    strength = _as_factor_parameter(loading_strength, k, "loading_strength")
    neighborhood = max(0, int(round(n / 10))) if neighborhood is None else int(neighborhood)
    if neighborhood < 0:
        raise ValueError("neighborhood must be non-negative")
    if not error_variance_range[0] > 0 or error_variance_range[1] < error_variance_range[0]:
        raise ValueError("error_variance_range must be positive and ordered")

    loadings = rng.normal(size=(n, k)) * strength
    theta = rng.uniform(error_variance_range[0], error_variance_range[1], size=n)

    factors = np.empty((t, k), dtype=np.float64)
    for j in range(k):
        factory = lambda m, j=j: sgt_rvs(
            m,
            sigma=sig_f[j],
            lam=lam_f[j],
            p=p_f[j],
            q=q_f[j],
            rng=rng,
        )
        factors[:, j] = _ar1_from_innovations(
            factory(t), float(rho_f[j]), rng=rng, innovation_factory=factory
        )

    iid = sgt_rvs(
        (t, n),
        sigma=error_sigma,
        lam=error_lambda,
        p=error_p,
        q=error_q,
        rng=rng,
    )
    cross = iid.copy()
    if neighborhood > 0 and beta != 0.0:
        for j in range(n):
            left = max(0, j - neighborhood)
            right = min(n, j + neighborhood + 1)
            neighbors = np.sum(iid[:, left:right], axis=1) - iid[:, j]
            cross[:, j] += beta * neighbors

    serial = np.empty_like(cross)
    if abs(error_ar) >= 1.0:
        raise ValueError("error_ar must have absolute value below one")
    for j in range(n):
        state = 0.0
        for i in range(t):
            state = error_ar * state + cross[i, j]
            serial[i, j] = state

    neighbor_count = max(1, 2 * neighborhood)
    normalization = math.sqrt((1.0 - error_ar**2) / (1.0 + neighbor_count * beta**2))
    errors = serial * np.sqrt(theta)[None, :] * normalization
    x = factors @ loadings.T + errors
    return SimulationResult(x, loadings, factors, errors)


def hofa_DGP1(
    n: int,
    t: int,
    k: int,
    par_f: Sequence[ArrayLike | float],
    par_e: Sequence[ArrayLike | float],
    alpha: float,
    rho_f: ArrayLike,
    *,
    rho_e: float = 0.2,
    random_state: int | np.random.Generator | None = None,
) -> SimulationResult:
    """Compatibility wrapper matching the R ``hofa.DGP1`` parameter bundles."""
    return dgp1(
        n,
        t,
        k,
        factor_sigma=par_f[0],
        factor_lambda=par_f[1],
        factor_p=par_f[2],
        factor_q=par_f[3],
        error_sigma=float(np.asarray(par_e[0])),
        error_lambda=float(np.asarray(par_e[1])),
        error_p=float(np.asarray(par_e[2])),
        error_q=float(np.asarray(par_e[3])),
        alpha=alpha,
        factor_ar=rho_f,
        error_ar=rho_e,
        random_state=random_state,
    )


def hofa_DGP2(
    n: int,
    t: int,
    k: int,
    par_f: Sequence[ArrayLike | float],
    par_e: Sequence[ArrayLike | float],
    par_cove: Sequence[ArrayLike | float] | dict[str, ArrayLike | float],
    rho_f: ArrayLike,
    *,
    random_state: int | np.random.Generator | None = None,
) -> SimulationResult:
    """Compatibility wrapper matching the R ``hofa.DGP2`` parameter bundles."""
    if isinstance(par_cove, dict):
        beta = float(par_cove["beta"])
        neighborhood = int(par_cove["J"])
        error_ar = float(par_cove["rho"])
        variance_range = tuple(np.asarray(par_cove["msig_e"], dtype=float))
    else:
        beta = float(np.asarray(par_cove[0]))
        neighborhood = int(np.asarray(par_cove[1]))
        error_ar = float(np.asarray(par_cove[2]))
        variance_range = tuple(np.asarray(par_cove[3], dtype=float))
    return dgp2(
        n,
        t,
        k,
        factor_sigma=par_f[0],
        factor_lambda=par_f[1],
        factor_p=par_f[2],
        factor_q=par_f[3],
        error_sigma=float(np.asarray(par_e[0])),
        error_lambda=float(np.asarray(par_e[1])),
        error_p=float(np.asarray(par_e[2])),
        error_q=float(np.asarray(par_e[3])),
        beta=beta,
        neighborhood=neighborhood,
        error_ar=error_ar,
        error_variance_range=(float(variance_range[0]), float(variance_range[1])),
        factor_ar=rho_f,
        random_state=random_state,
    )
