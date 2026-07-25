"""Cubic B-spline basis for the continuous-time forecast head.

A physics-informed model needs ``dG/dt``. There are two usual ways to get it and
both have costs: autodiff with respect to a time input needs one backward pass
per collocation point, and finite differencing reintroduces discretisation error
(the legacy code differenced across 30-minute gaps against per-minute rate
constants, which is why its residual was meaningless).

Instead the network emits **coefficients of a cubic B-spline** over the forecast
interval. Then

* ``G(t)``      is a fixed linear map applied to the coefficients,
* ``dG/dt``     is another fixed linear map -- exact and analytic,
* the horizon predictions are just ``G(30), G(60), G(90), G(120)``,

so the discrete targets and the continuous function are the same object. There is
no train/report mismatch and no solver in the loop. Both design matrices are
precomputed once and held as buffers.

Anchoring
---------
The forecast must start from the last observed CGM value. Rather than adding a
penalty for that, it is imposed by construction::

    G(t) = G_0 + sum_k c_k * (B_k(t) - B_k(0))

so ``G(0) = G_0`` identically, for any coefficients. ``dG/dt`` is unaffected
because the subtracted term is constant in ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

DEGREE = 3


def clamped_uniform_knots(t_max: float, n_basis: int, degree: int = DEGREE) -> np.ndarray:
    """Clamped knot vector with uniformly spaced interior knots on ``[0, t_max]``.

    A clamped vector repeats the end knots ``degree + 1`` times so the basis
    spans the endpoints, which is what lets ``G(0)`` and ``G(t_max)`` be
    controlled directly. ``n_basis = len(knots) - degree - 1``, hence
    ``n_interior = n_basis - degree - 1``.
    """
    n_interior = n_basis - degree - 1
    if n_interior < 0:
        raise ValueError(
            f"n_basis={n_basis} too small for degree {degree}; need at least {degree + 1}"
        )
    interior = np.linspace(0.0, t_max, n_interior + 2)[1:-1]
    return np.concatenate(
        [np.zeros(degree + 1), interior, np.full(degree + 1, t_max)]
    )


def design_matrices(
    eval_min: np.ndarray, t_max: float, n_basis: int, degree: int = DEGREE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Value, derivative, and value-at-zero design matrices.

    Returns ``(basis, basis_derivative, basis_at_zero)`` with shapes
    ``(len(eval_min), n_basis)``, ``(len(eval_min), n_basis)``, and
    ``(n_basis,)``.

    Built with :mod:`scipy.interpolate` rather than a hand-rolled Cox-de Boor
    recursion, and only once at construction, so correctness matters more than
    speed here.
    """
    from scipy.interpolate import BSpline

    knots = clamped_uniform_knots(t_max, n_basis, degree)
    basis = np.zeros((len(eval_min), n_basis))
    derivative = np.zeros((len(eval_min), n_basis))
    at_zero = np.zeros(n_basis)

    for index in range(n_basis):
        coefficients = np.zeros(n_basis)
        coefficients[index] = 1.0
        spline = BSpline(knots, coefficients, degree, extrapolate=False)
        basis[:, index] = np.nan_to_num(spline(eval_min))
        derivative[:, index] = np.nan_to_num(spline.derivative(1)(eval_min))
        at_zero[index] = np.nan_to_num(spline(0.0))

    return basis, derivative, at_zero


@dataclass(frozen=True)
class SplineGrid:
    """The evaluation grid shared by the head and the physics residual.

    ``collocation_min`` is the dense grid where the ODE residual is evaluated.
    ``horizon_min`` are the reported forecast horizons. Both are evaluated from
    the same coefficients, which is the point: the residual constrains exactly
    the function that is reported.
    """

    collocation_min: np.ndarray
    horizon_min: np.ndarray

    @classmethod
    def build(
        cls,
        horizons_min: tuple[int, ...] = (30, 60, 90, 120),
        n_collocation: int = 121,
    ) -> "SplineGrid":
        t_max = float(max(horizons_min))
        collocation = np.linspace(0.0, t_max, n_collocation)
        return cls(collocation_min=collocation, horizon_min=np.asarray(horizons_min, dtype=float))

    @property
    def t_max(self) -> float:
        return float(self.horizon_min.max())


class SplineEvaluator(torch.nn.Module):
    """Maps spline coefficients to glucose values and derivatives.

    Holds no learnable weights -- only constant design matrices, registered as
    buffers so they move with ``.to(device)`` and are saved in the state dict
    (which makes a checkpoint self-describing about the grid it was trained on).

    Shapes: coefficients ``(B, n_basis)``, ``G_0`` ``(B,)``. Outputs are
    ``(B, n_collocation)`` or ``(B, n_horizons)``.
    """

    def __init__(
        self,
        grid: SplineGrid,
        n_basis: int = 12,
        degree: int = DEGREE,
    ) -> None:
        super().__init__()
        self.n_basis = n_basis
        self.degree = degree
        self.grid = grid

        coll_basis, coll_derivative, at_zero = design_matrices(
            grid.collocation_min, grid.t_max, n_basis, degree
        )
        horizon_basis, _, _ = design_matrices(
            grid.horizon_min, grid.t_max, n_basis, degree
        )

        # Stored in float64: these are exact constants, and the derivative map in
        # particular is differenced against analytic references in the tests, so
        # float32 storage would cap achievable precision at ~1e-7 relative. All
        # consumers cast to the coefficient dtype at use, so a float32 model pays
        # no runtime cost.
        self.register_buffer("collocation_basis", torch.tensor(coll_basis, dtype=torch.float64))
        self.register_buffer(
            "collocation_derivative", torch.tensor(coll_derivative, dtype=torch.float64)
        )
        self.register_buffer("horizon_basis", torch.tensor(horizon_basis, dtype=torch.float64))
        self.register_buffer("basis_at_zero", torch.tensor(at_zero, dtype=torch.float64))
        self.register_buffer(
            "collocation_min", torch.tensor(grid.collocation_min, dtype=torch.float64)
        )
        self.register_buffer("horizon_min", torch.tensor(grid.horizon_min, dtype=torch.float64))

    def _anchor(self, basis: Tensor) -> Tensor:
        """Subtract the value at ``t=0`` so the anchoring identity holds."""
        return basis - self.basis_at_zero.unsqueeze(0)

    def value(self, coefficients: Tensor, G0: Tensor, *, at: str = "collocation") -> Tensor:
        """Glucose ``G(t)`` [mg/dL] at the requested grid."""
        basis = self.collocation_basis if at == "collocation" else self.horizon_basis
        anchored = self._anchor(basis).to(coefficients.dtype)
        return G0.unsqueeze(-1) + coefficients @ anchored.T

    def derivative(self, coefficients: Tensor) -> Tensor:
        """``dG/dt`` [mg/dL/min] at the collocation grid -- analytic, exact.

        The anchoring offset is constant in ``t`` so it does not appear here.
        """
        basis = self.collocation_derivative.to(coefficients.dtype)
        return coefficients @ basis.T

    def forward(self, coefficients: Tensor, G0: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(G_collocation, dG_dt_collocation, G_horizons)``."""
        return (
            self.value(coefficients, G0, at="collocation"),
            self.derivative(coefficients),
            self.value(coefficients, G0, at="horizon"),
        )


__all__ = [
    "DEGREE",
    "SplineEvaluator",
    "SplineGrid",
    "clamped_uniform_knots",
    "design_matrices",
]
