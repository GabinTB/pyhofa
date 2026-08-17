"""Adaptive higher-order factor analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._types import AdaptiveResult
from ._utils import eigh_desc, fit_center_scale
from .estimators import m2_pca, m3_als, m4_als
from .moments import c4m, m3m
from .selection import m2_select, m3_select, m4_select


def adaptive_hfa(
    x: ArrayLike,
    *,
    scale: bool = False,
    r: int | None = None,
    max_order: int = 3,
    tau_nt: float | None = None,
    rmax: int = 8,
) -> AdaptiveResult:
    """Adaptive HFA for weak non-Gaussian factor models.

    The estimator chooses between covariance, third-order and optionally
    fourth-order cumulant estimators using factor contribution ratios (FCRs).

    Higher-order estimates use the split selected by GER3 and GER4: the
    largest selected non-Gaussian count is retained and the remaining factors
    are estimated from residual covariance as Gaussian components. The
    second-order candidate has no such distinction, so its FCR compares a PCA
    model with the structured higher-order candidates whenever Gaussian
    factors are selected.

    For out-of-sample evaluation, fit only on the training block and call
    :func:`pyhofa.project` with the returned ``loadings``, ``mean_`` and
    ``scale_``. Fitting on the full sample and slicing ``factors`` leaks test
    observations into both preprocessing and loading estimation.

    ``cumulant_order_f`` records the factor-order winner by FCR, while the
    returned ``factors`` are projected onto the selected loading order
    ``cumulant_order_u``. This keeps ``factors`` consistent with ``loadings``
    and with :func:`pyhofa.project` when the two selected orders differ.
    """
    if max_order not in {3, 4}:
        raise ValueError("max_order must be 3 or 4")
    data, fitted_mean, fitted_scale = fit_center_scale(x, scale=scale)
    t, n = data.shape
    if tau_nt is None:
        tau_nt = 2.0 * t**0.25 / n

    selection3 = m3_select(data, rmax=rmax, method="GER3")
    # Preserve the original adaptive rule, which also checks fourth-order
    # selection even when max_order=3.
    selection4 = m4_select(data, rmax=rmax, method="GER4")
    if r is None:
        r2 = m2_select(data, rmax=rmax, method="ER", modified=True).n_factors
        r = max(r2, selection3.n_factors, selection4.n_factors)
    if r < 0 or r > min(t, n):
        raise ValueError("r must lie between 0 and min(t, n)")
    if r == 0:
        return AdaptiveResult(
            np.empty((t, 0)),
            np.empty((n, 0)),
            2,
            2,
            0,
            {2: 0.0, 3: 0.0, 4: 0.0},
            0,
            0,
            fitted_mean,
            fitted_scale,
        )

    n_nongaussian = min(
        r,
        max(selection3.n_nongaussian or 0, selection4.n_nongaussian or 0),
    )
    n_gaussian = r - n_nongaussian

    cov = np.cov(data, rowvar=False, ddof=1)
    ev2, _ = eigh_desc(cov)
    ev2 = np.clip(ev2 / n, 0.0, None)
    ev3, _ = eigh_desc(m3m(data))
    ev3 = np.clip(ev3, 0.0, None)
    ev4: np.ndarray | None = None
    if max_order == 4:
        ev4, _ = eigh_desc(c4m(data, data.T @ data / t))
        ev4 = np.clip(ev4, 0.0, None)

    ns = max(r, min(n, max(1, int(np.floor(0.5 * n)))))

    def fcr(ev: np.ndarray) -> float:
        denominator = float(np.sum(ev[:ns]))
        if denominator <= np.finfo(float).tiny:
            return 0.0
        return float(np.sum(ev[:r]) / denominator)

    fcrs = {2: fcr(ev2), 3: fcr(ev3)}
    if ev4 is not None:
        fcrs[4] = fcr(ev4)

    res2 = m2_pca(data, r, method="PCA")
    res3 = m3_als(
        data,
        gamma=(0.0, 1.0),
        rh=n_nongaussian,
        rg=n_gaussian,
    )
    estimates = {2: res2, 3: res3}
    if max_order == 4:
        estimates[4] = m4_als(
            data,
            gamma=(0.0, 0.0, 1.0),
            rh=n_nongaussian,
            rg=n_gaussian,
        )

    order_f = max(estimates, key=lambda order: fcrs[order])
    loading_scores = dict(fcrs)
    loading_scores[2] += float(tau_nt)
    order_u = max(estimates, key=lambda order: loading_scores[order])

    return AdaptiveResult(
        estimates[order_u].factors,
        estimates[order_u].loadings,
        order_f,
        order_u,
        r,
        fcrs,
        n_nongaussian,
        n_gaussian,
        fitted_mean,
        fitted_scale,
    )


Adaptive_HFA = adaptive_hfa
