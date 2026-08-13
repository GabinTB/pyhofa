"""Higher-order moment and cumulant matrices used by HOFA."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm

from ._utils import as_2d_float, center_scale, eigh_desc, pava

FloatArray = NDArray[np.float64]


def super_diag(x: ArrayLike, order: int) -> FloatArray:
    """Super-diagonal matricization used for diagonal higher-order tensors."""
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    n = values.size
    if order == 2:
        return np.diag(values)
    if order not in {3, 4}:
        raise ValueError("order must be 2, 3 or 4")
    out = np.zeros((n, n ** (order - 1)), dtype=np.float64)
    for i, value in enumerate(values):
        index = sum(i * n**power for power in range(order - 1))
        out[i, index] = value
    return out


def m2m(x: ArrayLike, *, center: bool = False) -> FloatArray:
    """Gram product of the second-order sample moment matrix.

    This is ``X.T @ (X @ X.T) @ X`` and equals the unnormalised
    covariance-matricization product used by the original package.
    """
    data = center_scale(x) if center else as_2d_float(x)
    gram = data @ data.T
    return data.T @ gram @ data


def m3m(x: ArrayLike, *, center: bool = False) -> FloatArray:
    """Product of the third-order moment matricization with its transpose."""
    data = center_scale(x) if center else as_2d_float(x)
    gram = data @ data.T
    return data.T @ (gram * gram) @ data


def m4m(x: ArrayLike, *, center: bool = False) -> FloatArray:
    """Product of the fourth-order raw-moment matricization with its transpose."""
    data = center_scale(x) if center else as_2d_float(x)
    gram = data @ data.T
    return data.T @ (gram * gram * gram) @ data


def third_moment_matrix(x: ArrayLike, *, center: bool = True) -> FloatArray:
    """Return the third central-moment tensor matricized as ``n x n^2``."""
    data = center_scale(x) if center else as_2d_float(x)
    tensor = np.einsum("ti,tj,tk->ijk", data, data, data, optimize=True) / data.shape[0]
    return tensor.reshape(data.shape[1], -1)


def fourth_moment_matrix(x: ArrayLike, *, center: bool = True) -> FloatArray:
    """Return the fourth central-moment tensor matricized as ``n x n^3``."""
    data = center_scale(x) if center else as_2d_float(x)
    tensor = np.einsum("ti,tj,tk,tl->ijkl", data, data, data, data, optimize=True)
    tensor /= data.shape[0]
    return tensor.reshape(data.shape[1], -1)


def fourth_cumulant_matrix(
    x: ArrayLike,
    covariance: ArrayLike | None = None,
    *,
    center: bool = True,
) -> FloatArray:
    """Fourth cumulant tensor matricized as ``n x n^3``.

    The definition is

    ``cum(i,j,k,l) = E[x_i x_j x_k x_l] - S_ij S_kl - S_ik S_jl - S_il S_jk``.
    """
    data = center_scale(x) if center else as_2d_float(x)
    t = data.shape[0]
    if covariance is None:
        covariance = data.T @ data / t
    cov = np.asarray(covariance, dtype=np.float64)
    m4 = fourth_moment_matrix(data, center=False).reshape(
        data.shape[1], data.shape[1], data.shape[1], data.shape[1]
    )
    c4 = (
        m4
        - np.einsum("ij,kl->ijkl", cov, cov)
        - np.einsum("ik,jl->ijkl", cov, cov)
        - np.einsum("il,jk->ijkl", cov, cov)
    )
    return c4.reshape(data.shape[1], -1)


def c4m(x: ArrayLike, covariance: ArrayLike | None = None) -> FloatArray:
    """Efficiently compute ``C4 @ C4.T`` without materialising ``C4``.

    This is the Python counterpart of the original package's ``C4M`` helper.
    """
    data = center_scale(x)
    t = data.shape[0]
    if covariance is None:
        cov = data.T @ data / t
    else:
        cov = np.asarray(covariance, dtype=np.float64)
    gram = data @ data.T
    raw4_product = data.T @ (gram**3) @ data / t**2

    q = np.einsum("ti,ij,tj->t", data, cov, data, optimize=True)
    correction = 3.0 * data.T @ ((q[:, None] + q[None, :]) * gram) @ data / t**2

    cov2 = cov @ cov
    e1 = 3.0 * float(np.sum(cov * cov)) * cov2
    e2 = 6.0 * cov2 @ cov2
    result = raw4_product - correction + e1 + e2
    return (result + result.T) / 2.0


def c4_to_m4(c4: ArrayLike, covariance: ArrayLike) -> FloatArray:
    """Convert a matricized fourth cumulant into a fourth central moment."""
    c4_mat = np.asarray(c4, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    n = cov.shape[0]
    tensor = c4_mat.reshape(n, n, n, n)
    m4 = (
        tensor
        + np.einsum("ij,kl->ijkl", cov, cov)
        + np.einsum("ik,jl->ijkl", cov, cov)
        + np.einsum("il,jk->ijkl", cov, cov)
    )
    return m4.reshape(n, -1)


def m4_to_c4(m4: ArrayLike, covariance: ArrayLike) -> FloatArray:
    """Convert a matricized fourth central moment into a fourth cumulant."""
    m4_mat = np.asarray(m4, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    n = cov.shape[0]
    tensor = m4_mat.reshape(n, n, n, n)
    c4 = (
        tensor
        - np.einsum("ij,kl->ijkl", cov, cov)
        - np.einsum("ik,jl->ijkl", cov, cov)
        - np.einsum("il,jk->ijkl", cov, cov)
    )
    return c4.reshape(n, -1)


def variable_moment(x: ArrayLike, order: int) -> FloatArray:
    """Column-wise central moment of an arbitrary positive integer order."""
    if order < 1:
        raise ValueError("order must be positive")
    data = center_scale(x)
    return np.mean(data**order, axis=0)


def trace_ratio(estimated: ArrayLike, truth: ArrayLike) -> float:
    """Subspace trace ratio used in the HOFA simulation study."""
    f = as_2d_float(estimated, name="estimated")
    f0 = as_2d_float(truth, name="truth")
    if f.shape[0] != f0.shape[0]:
        raise ValueError("estimated and truth must have the same number of rows")
    projection = f @ np.linalg.pinv(f.T @ f) @ f.T
    numerator = np.trace(f0.T @ projection @ f0)
    denominator = np.trace(f0.T @ f0)
    return float(numerator / denominator)


def jmca(
    x: ArrayLike,
    kmax: int,
    gamma: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[FloatArray, FloatArray]:
    """Joint moment component analysis matrix used by the JJR selectors."""
    data = center_scale(x, scale=True)
    n_obs, n_vars = data.shape
    m2 = m2m(data) / n_vars**2
    m3 = m3m(data) / n_vars**2 / n_obs**2
    m4 = m4m(data) / n_vars**2 / n_obs**4
    joint = gamma[0] * m2 + gamma[1] * m3 + gamma[2] * m4
    values, vectors = eigh_desc(joint)
    values = np.sqrt(np.clip(values, 0.0, None))
    return values[:kmax], vectors[:, :kmax]


def portfolio_moments_independent(
    w: ArrayLike,
    factor_moments: tuple[ArrayLike, ArrayLike, ArrayLike],
    idiosyncratic_moments: tuple[ArrayLike, ArrayLike, ArrayLike],
    loadings: ArrayLike,
) -> FloatArray:
    """Portfolio moments under independent factor/idiosyncratic components."""
    weights = np.asarray(w, dtype=np.float64)
    a = np.asarray(loadings, dtype=np.float64)
    m2f, m3f, m4f = (np.asarray(v, dtype=np.float64) for v in factor_moments)
    m2e, m3e, m4e = (np.asarray(v, dtype=np.float64) for v in idiosyncratic_moments)
    b = weights @ a
    m2p = float(np.sum(b**2 * m2f) + np.sum(weights**2 * m2e))
    m3p = float(np.sum(b**3 * m3f) + np.sum(weights**3 * m3e))
    m4p = float(
        np.sum(b**4 * (m4f - 3.0 * m2f**2))
        + np.sum(weights**4 * (m4e - 3.0 * m2e**2))
        + 3.0 * m2p**2
    )
    return np.array([m2p, m3p, m4p])


def portfolio_moments_matrix(
    w: ArrayLike,
    factor_moments: tuple[ArrayLike, ArrayLike, ArrayLike],
    idiosyncratic_moments: tuple[ArrayLike, ArrayLike, ArrayLike],
    loadings: ArrayLike,
) -> FloatArray:
    """Portfolio moments with full factor co-moment matrices."""
    weights = np.asarray(w, dtype=np.float64)
    a = np.asarray(loadings, dtype=np.float64)
    m2f, m3f, c4f = (np.asarray(v, dtype=np.float64) for v in factor_moments)
    m2e, m3e, m4e = (np.asarray(v, dtype=np.float64) for v in idiosyncratic_moments)
    b = weights @ a
    m2p = float(b @ m2f @ b + np.sum(weights**2 * m2e))
    m3p = float(b @ m3f @ np.kron(b, b) + np.sum(weights**3 * m3e))
    m4p = float(
        b @ c4f @ np.kron(np.kron(b, b), b)
        + np.sum(weights**4 * (m4e - 3.0 * m2e**2))
        + 3.0 * m2p**2
    )
    return np.array([m2p, m3p, m4p])


def modified_var_objective(moments: ArrayLike, alpha: float = 0.05) -> float:
    """Modified VaR objective used by the original portfolio routines."""
    m2p, m3p, m4p = np.asarray(moments, dtype=np.float64)
    if m2p <= 0:
        return np.inf
    z = float(norm.ppf(alpha))
    std = np.sqrt(m2p)
    skew = m3p / std**3
    kurt = m4p / std**4
    correction = (
        -(z**2 - 1.0) * skew / 6.0
        - (z**3 - 3.0 * z) * kurt / 24.0
        + (2.0 * z**3 - 5.0 * z) * skew**2 / 36.0
    )
    return float(-std * z + std * correction)


def expected_utility_objective(moments: ArrayLike, gamma: float = 10.0) -> float:
    """Fourth-order CRRA expected-utility loss approximation."""
    m2p, m3p, m4p = np.asarray(moments, dtype=np.float64)
    return float(
        gamma * m2p / 2.0
        - gamma * (gamma + 1.0) * m3p / 6.0
        + gamma * (gamma + 1.0) * (gamma + 2.0) * m4p / 24.0
    )


def linear_identity_shrinkage(x: ArrayLike, k: int = 0) -> FloatArray:
    """Ledoit-Wolf linear shrinkage toward a scaled identity matrix."""
    data = as_2d_float(x).copy()
    if k == 0:
        data -= data.mean(axis=0, keepdims=True)
    n, p = data.shape
    effective_n = n - k
    if effective_n <= 0:
        raise ValueError("n - k must be positive")
    sample = data.T @ data / effective_n
    mean_eigenvalue = float(np.trace(sample) / p)
    target = mean_eigenvalue * np.eye(p)
    d2 = float(np.sum((sample - target) ** 2) / p)
    if d2 <= np.finfo(float).tiny:
        return sample
    bbar2 = 0.0
    for row in data:
        outer = np.outer(row, row)
        bbar2 += np.sum((outer - sample) ** 2)
    bbar2 /= p * effective_n**2
    b2 = min(d2, float(bbar2))
    a2 = d2 - b2
    return (b2 / d2) * target + (a2 / d2) * sample


def direct_nonlinear_shrinkage(x: ArrayLike, k: float = 1.0) -> FloatArray:
    """Direct nonlinear covariance shrinkage following the original HOFA helper.

    This ports the kernel estimator used by the R package and applies a decreasing
    pool-adjacent-violators projection to the cleaned eigenvalues.
    """
    data = center_scale(x)
    n, p = data.shape
    sample = data.T @ data / n
    eigenvalues, eigenvectors = eigh_desc(sample)
    tiny = 1e-8
    positive = eigenvalues[eigenvalues >= tiny]
    replacement = positive[-1] if positive.size else tiny
    eigenvalues = np.where(eigenvalues < tiny, replacement, eigenvalues)

    m = min(p, n)
    lam = eigenvalues[-m:]
    lmat = np.repeat(lam[:, None], m, axis=1)
    h = n ** (-0.35)
    lt = lmat.T
    denom = 2.0 * np.pi * lt**2 * h**2
    with np.errstate(divide="ignore", invalid="ignore"):
        f_tilde = np.nanmean(
            np.sqrt(np.maximum(0.0, 4.0 * lt**2 * h**2 - (lmat - lt) ** 2)) / denom,
            axis=1,
        )
        hf_tilde = np.nanmean(
            (
                np.sign(lmat - lt)
                * np.sqrt(np.maximum(0.0, (lmat - lt) ** 2 - 4.0 * lt**2 * h**2))
                - lmat
                + lt
            )
            / denom,
            axis=1,
        )

    concentration = p / n
    if p <= n:
        d_tilde = lam / (
            (np.pi * concentration * lam * f_tilde) ** 2
            + (1.0 - concentration - np.pi * concentration * lam * hf_tilde) ** 2
        )
    else:
        hf0 = (1.0 - np.sqrt(max(0.0, 1.0 - 4.0 * h**2))) / (2.0 * np.pi * h**2)
        hf0 *= np.mean(1.0 / lam)
        d0 = 1.0 / (np.pi * (p - n) / n * hf0)
        d1 = lam / (np.pi**2 * lam**2 * (f_tilde**2 + hf_tilde**2))
        d_tilde = np.concatenate([np.full(p - n, d0), d1])

    d_hat = pava(d_tilde, increasing=False)
    d_hat = np.maximum(d_hat, np.finfo(float).tiny) ** k
    # The R implementation associates d_tilde with ascending small-to-large lambdas.
    # Reverse back to the descending eigenvector ordering used by eigh_desc.
    if d_hat.size == p:
        cleaned = d_hat[::-1]
    else:
        cleaned = np.concatenate([eigenvalues[: p - d_hat.size], d_hat[::-1]])
    result = eigenvectors @ np.diag(cleaned) @ eigenvectors.T
    return (result + result.T) / 2.0
