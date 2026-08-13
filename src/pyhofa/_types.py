"""Result containers used by pyhofa."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class FactorResult:
    """Estimated factors, loadings and residuals."""

    factors: FloatArray
    loadings: FloatArray
    residuals: FloatArray
    eigenvalues: FloatArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def f(self) -> FloatArray:
        """Compatibility alias for the R package's ``f`` field."""
        return self.factors

    @property
    def u(self) -> FloatArray:
        """Compatibility alias for the R package's ``u`` field."""
        return self.loadings

    @property
    def e(self) -> FloatArray:
        """Compatibility alias for the R package's ``e`` field."""
        return self.residuals

    @property
    def ev(self) -> FloatArray | None:
        """Compatibility alias for the R package's ``ev`` field."""
        return self.eigenvalues


@dataclass(slots=True)
class SelectionResult:
    """Factor-number selection result."""

    n_factors: int
    eigenvalues: FloatArray | None = None
    n_nongaussian: int | None = None
    n_gaussian: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def R(self) -> int:  # noqa: N802 - compatibility with the R API
        """Compatibility alias for the R package's ``R`` field."""
        return self.n_factors

    @property
    def Rh(self) -> int | None:  # noqa: N802
        """Compatibility alias for the R package's ``Rh`` field."""
        return self.n_nongaussian

    @property
    def Rg(self) -> int | None:  # noqa: N802
        """Compatibility alias for the R package's ``Rg`` field."""
        return self.n_gaussian


@dataclass(slots=True)
class AdaptiveResult:
    """Adaptive HFA factor and loading estimates."""

    factors: FloatArray
    loadings: FloatArray
    cumulant_order_f: int
    cumulant_order_u: int
    n_factors: int
    factor_contribution_ratios: dict[int, float]

    @property
    def f(self) -> FloatArray:
        return self.factors

    @property
    def u(self) -> FloatArray:
        return self.loadings


@dataclass(slots=True)
class SimulationResult:
    """Synthetic factor-model sample."""

    X: FloatArray
    loadings: FloatArray
    factors: FloatArray
    errors: FloatArray

    @property
    def W(self) -> FloatArray:
        return self.loadings

    @property
    def FF(self) -> FloatArray:
        return self.factors

    @property
    def E(self) -> FloatArray:
        return self.errors


@dataclass(slots=True)
class PortfolioResult:
    """Portfolio optimization result."""

    weights: FloatArray
    objective: float
    n_factors: int
    factor_moments: tuple[FloatArray, ...]
    idiosyncratic_moments: tuple[FloatArray, ...]
    portfolio_moments: FloatArray
    success: bool
    message: str

    @property
    def w(self) -> FloatArray:
        return self.weights

    @property
    def r(self) -> int:
        return self.n_factors
