"""The Bergman minimal model glucose equation: residual and forward solution.

Full model, all three states integrated (contrast with the legacy
``PhysicsInformedLoss``, which never integrated ``dX/dt`` at all and set
``X = p3 * IOB`` algebraically)::

    dG/dt = -(p1 + X)*G + p1*G_b + Ra(t)/V_G      [mg/dL/min]
    dX/dt = -p2*X + p3*(I - I_b)                   [1/min^2]
    dI/dt = -n*(I - I_b) + u_I(t)/V_I              [uU/mL/min]

``X`` and ``I`` are advanced exactly by :mod:`twin.physio.compartments`. This
module handles glucose.

Two uses:

**Residual** (:func:`glucose_residual`) -- the collocation term of the physics
loss. The network's spline head supplies both ``G`` and ``dG/dt`` analytically,
so the residual is evaluated without any finite differencing.

**Forward solution** (:func:`integrate_glucose`) -- the mechanistic prior of the
hybrid model. Because ``X(t)`` is already known, the glucose equation is a
*linear* first-order ODE with time-varying coefficients, so it too needs no
nonlinear solver: with ``k(t) = p1 + X(t)`` and ``c(t) = p1*G_b + Ra(t)/V_G``,

.. math::

    G(t+\\Delta) = G(t) e^{-k\\Delta} + \\frac{c}{k}\\left(1 - e^{-k\\Delta}\\right)

exactly, for ``k`` and ``c`` piecewise-constant on the step. On a 1-minute grid
that is accurate far below the residual's noise floor.
"""

from __future__ import annotations

import torch
from torch import Tensor

from twin.physio.params import PatientParams

#: Below this magnitude of ``k*dt`` the closed-form update is evaluated by its
#: Taylor expansion. ``p1`` is conventionally ~0 in type 1 diabetes and ``X``
#: starts near zero, so ``k -> 0`` is the normal case, not an edge case.
_SMALL = 1e-8


def effective_disposal(X: Tensor, params: PatientParams) -> Tensor:
    """Net fractional glucose disposal rate ``p1 + X``, floored at zero.

    Remote insulin action is non-negative by definition: insulin cannot act in
    reverse. A reduced or suspended basal drops plasma insulin below ``I_b``, which
    makes the linear ``X`` state briefly negative, and if ``p1 + X < 0`` the glucose
    equation becomes ``dG/dt = +|k| G`` -- exponential divergence. Over a 121-point
    collocation grid that overflows to NaN, which is exactly how a training run was
    first observed to fail.

    Flooring at zero is the physiologically correct constraint, not a numerical
    patch: it says insulin action cannot fall below none at all. Endogenous glucose
    production above basal, the real mechanism behind a rise during a basal
    reduction, is represented by the ``p1 * G_b`` term rather than by negative
    insulin action.
    """
    return (params.p1.unsqueeze(-1) + X).clamp(min=0.0)


def _one_minus_exp_over(x: Tensor) -> Tensor:
    """``(1 - exp(-x)) / x``, numerically stable including at ``x = 0``.

    The limit is 1; the second-order expansion ``1 - x/2 + x^2/6`` is used near
    zero, where the naive quotient loses all significance.
    """
    small = x.abs() < _SMALL
    safe = torch.where(small, torch.ones_like(x), x)
    exact = -torch.expm1(-safe) / safe
    approx = 1.0 - x / 2.0 + x * x / 6.0
    return torch.where(small, approx, exact)


def glucose_residual(
    G: Tensor,
    dG_dt: Tensor,
    X: Tensor,
    Ra_mgdl_per_min: Tensor,
    params: PatientParams,
) -> Tensor:
    """Bergman glucose-equation residual [mg/dL/min].

    ``r = dG/dt + (p1 + X)*G - p1*G_b - Ra/V_G``

    Zero for an exact solution. All arguments broadcast to a common shape,
    typically ``(B, n_collocation)``; ``params`` fields are ``(B,)`` and are
    unsqueezed to match.

    Note the residual is returned in physical units. Non-dimensionalise with
    :func:`residual_scale` before forming a loss, otherwise the physics weight is
    not comparable across subjects.
    """
    p1 = params.p1.unsqueeze(-1)
    G_b = params.G_b.unsqueeze(-1)
    return dG_dt + effective_disposal(X, params) * G - p1 * G_b - Ra_mgdl_per_min


def residual_scale(params: PatientParams) -> Tensor:
    """Characteristic glucose rate [mg/dL/min] used to non-dimensionalise.

    With the reference scales ``G ~ G_b`` and ``t ~ T``, a residual divided by
    ``G_b / T`` is dimensionless and O(1). ``T = 60`` min is the natural
    timescale of the forecast problem.
    """
    reference_time_min = 60.0
    return (params.G_b / reference_time_min).unsqueeze(-1)


def integrate_glucose(
    G0: Tensor,
    X: Tensor,
    Ra_mgdl_per_min: Tensor,
    params: PatientParams,
    *,
    dt: float = 1.0,
) -> Tensor:
    """Forward-solve the glucose equation on a uniform grid.

    Parameters
    ----------
    G0
        Initial glucose [mg/dL], shape ``(B,)``.
    X
        Remote insulin action [1/min] on the grid, shape ``(B, T)``.
    Ra_mgdl_per_min
        Glucose appearance as a rate [mg/dL/min], shape ``(B, T)``.
    params
        Batched patient parameters.
    dt
        Step in minutes, matching the ``X`` / ``Ra`` grid.

    Returns
    -------
    Tensor
        Shape ``(B, T)``. Element 0 is ``G0``; element ``t`` is glucose at
        ``t * dt`` minutes after the start.

    This is the mechanistic prior of the hybrid model. The network learns only
    the discrepancy from it, which is both more accurate and easier to defend
    than asking a penalty term to impose physiology on an unconstrained output.

    Notes
    -----
    ``k`` and ``c`` are not genuinely piecewise-constant -- ``X`` and ``Ra`` vary
    continuously within a step -- so sampling them at the step start would leave
    an O(dt) bias, which is visible as a sustained residual after a meal. Both
    coefficients are therefore averaged across the step, giving the midpoint rule
    and O(dt^2) accuracy at no extra cost.
    """
    if X.shape != Ra_mgdl_per_min.shape:
        raise ValueError(
            f"X {tuple(X.shape)} != Ra {tuple(Ra_mgdl_per_min.shape)}"
        )
    steps = X.shape[-1]
    p1 = params.p1.unsqueeze(-1)
    G_b = params.G_b.unsqueeze(-1)

    k = effective_disposal(X, params)  # (B, T), floored at zero
    c = p1 * G_b + Ra_mgdl_per_min  # (B, T)

    # Step-averaged coefficients: element t governs the interval [t, t+1).
    k_step = 0.5 * (k[:, :-1] + k[:, 1:])
    c_step = 0.5 * (c[:, :-1] + c[:, 1:])

    decay = torch.exp(-k_step * dt)
    # c/k * (1 - exp(-k*dt)) written as c*dt*(1-exp(-x))/x to stay finite at k=0.
    forcing = c_step * dt * _one_minus_exp_over(k_step * dt)

    trajectory = [G0]
    current = G0
    for step in range(steps - 1):
        current = current * decay[:, step] + forcing[:, step]
        trajectory.append(current)
    return torch.stack(trajectory, dim=-1)


def steady_state_glucose(
    X: Tensor, Ra_mgdl_per_min: Tensor, params: PatientParams
) -> Tensor:
    """Instantaneous equilibrium ``G`` for the given ``X`` and ``Ra``.

    Not a forecast -- a diagnostic. If the model's forecast wanders far from this
    for a sustained period the parameter estimates are not describing the data.
    """
    p1 = params.p1.unsqueeze(-1)
    G_b = params.G_b.unsqueeze(-1)
    k = p1 + X
    c = p1 * G_b + Ra_mgdl_per_min
    return c / k.clamp(min=_SMALL)


def insulin_sensitivity(params: PatientParams) -> Tensor:
    """``S_I = p3 / p2`` [mL/(uU*min)] -- the reported patient-specific parameter."""
    return params.S_I


__all__ = [
    "effective_disposal",
    "glucose_residual",
    "insulin_sensitivity",
    "integrate_glucose",
    "residual_scale",
    "steady_state_glucose",
]
