"""Exact propagation of the linear physiological compartments.

The insulin and carbohydrate sub-models are **linear** ODEs driven by known
inputs (pump delivery and meal records). Only the glucose equation is nonlinear,
because of the ``X * G`` product — and glucose is supplied by the network's
spline head rather than integrated. So the entire mechanistic state needed for
the physics residual can be advanced *exactly*, with no numerical integrator in
the training loop.

State vector (6 components, in this fixed order):

===  ========  =========  ==================================================
idx  symbol    unit       meaning
===  ========  =========  ==================================================
0    ``S1``    U          subcutaneous insulin, compartment 1
1    ``S2``    U          subcutaneous insulin, compartment 2
2    ``i``     uU/mL      plasma insulin above basal, ``I - I_b``
3    ``Qsto``  mg         glucose in stomach
4    ``Qgut``  mg         glucose in gut
5    ``X``     1/min      remote insulin action
===  ========  =========  ==================================================

Equations
---------
With ``a = 1 / tmax_I``::

    dS1/dt   = -a*S1 + u_ins(t)
    dS2/dt   =  a*S1 - a*S2
    di/dt    = -n*i + (1000/V_I) * a * S2
    dQsto/dt = -k_gri*Qsto + u_carb(t)
    dQgut/dt =  k_gri*Qsto - k_abs*Qgut
    dX/dt    = -p2*X + p3*i

The ``1000/V_I`` factor converts ``a*S2`` [U/min] into [uU/mL/min]: multiply by
1e6 uU/U and divide by ``V_I`` litres times 1000 mL/L.

Derived quantities
------------------
``IOB  = S1 + S2``          [U]   insulin delivered but not yet absorbed
``COB  = Qsto + Qgut``      [mg]  carbohydrate ingested but not yet appeared
``Ra   = f * k_abs * Qgut``  [mg/min] rate of glucose appearance
``I    = i + I_b``           [uU/mL]

``IOB`` and ``COB`` are mass-conserving by construction, which is the point:
they replace the hand-rolled convolution kernels of the legacy pipeline (whose
insulin kernel was time-reversed, putting almost no weight on a bolus at the
moment it was given) and they share one parameterisation with the physics loss.

Exact discretisation
--------------------
For a constant input ``u`` over a step ``dt``::

    x(t+dt) = Ad @ x(t) + Bd @ u,
    Ad = expm(A*dt),   Bd = (integral_0^dt expm(A*s) ds) @ B

Both blocks come from a single matrix exponential of the augmented system::

    M = expm([[A, B], [0, 0]] * dt)
    Ad = M[:6, :6]      Bd = M[:6, 6:]

Because ``dt`` is fixed, this is **one** 8x8 ``matrix_exp`` per patient for an
entire trajectory. Pump and meal records are genuinely piecewise-constant on the
1-minute grid, so the discretisation is exact rather than approximate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from twin.physio.params import PatientParams

N_STATES = 6
N_INPUTS = 2

IDX_S1, IDX_S2, IDX_I, IDX_QSTO, IDX_QGUT, IDX_X = range(N_STATES)
STATE_NAMES: tuple[str, ...] = ("S1", "S2", "i", "Qsto", "Qgut", "X")


@dataclass
class CompartmentTrajectory:
    """Mechanistic states on a uniform time grid.

    ``states`` has shape ``(B, T, 6)`` and ``time_min`` shape ``(T,)``, with
    ``time_min[0]`` the trajectory start. Element ``t`` is the state *at* that
    time, so ``states[:, 0]`` is the supplied initial condition.
    """

    states: Tensor
    time_min: Tensor
    params: PatientParams

    def _s(self, index: int) -> Tensor:
        return self.states[..., index]

    @property
    def S1(self) -> Tensor:
        return self._s(IDX_S1)

    @property
    def S2(self) -> Tensor:
        return self._s(IDX_S2)

    @property
    def X(self) -> Tensor:
        return self._s(IDX_X)

    @property
    def Qsto(self) -> Tensor:
        return self._s(IDX_QSTO)

    @property
    def Qgut(self) -> Tensor:
        return self._s(IDX_QGUT)

    @property
    def iob_u(self) -> Tensor:
        """Insulin on board [U] -- delivered but not yet absorbed."""
        return self.S1 + self.S2

    @property
    def cob_g(self) -> Tensor:
        """Carbohydrate on board [g] -- ingested but not yet appeared."""
        return (self.Qsto + self.Qgut) / 1000.0

    @property
    def plasma_insulin(self) -> Tensor:
        """Plasma insulin ``I`` [uU/mL], including basal."""
        return self._s(IDX_I) + self.params.I_b.unsqueeze(-1)

    @property
    def insulin_above_basal(self) -> Tensor:
        """``i = I - I_b`` [uU/mL]."""
        return self._s(IDX_I)

    @property
    def Ra_mg_per_min(self) -> Tensor:
        """Rate of glucose appearance [mg/min]."""
        f = self.params.f.unsqueeze(-1)
        k_abs = self.params.k_abs.unsqueeze(-1)
        return f * k_abs * self.Qgut

    @property
    def Ra_mgdl_per_min(self) -> Tensor:
        """Rate of appearance expressed as a glucose rate [mg/dL/min]."""
        return self.Ra_mg_per_min / self.params.V_G.unsqueeze(-1)

    def interpolate(self, query_min: Tensor) -> Tensor:
        """Linearly interpolate states at arbitrary times [min].

        The grid is 1-minute and the states are smooth on that scale, so linear
        interpolation is accurate to well below the residual's noise floor.
        Returns shape ``(B, len(query_min), 6)``.
        """
        grid = self.time_min
        query = query_min.to(grid.dtype).clamp(min=grid[0], max=grid[-1])
        step = grid[1] - grid[0]
        pos = (query - grid[0]) / step
        lower = pos.floor().long().clamp(max=grid.shape[0] - 2)
        frac = (pos - lower.to(pos.dtype)).unsqueeze(0).unsqueeze(-1)
        left = self.states[:, lower, :]
        right = self.states[:, lower + 1, :]
        return left + frac * (right - left)


def system_matrices(params: PatientParams) -> tuple[Tensor, Tensor]:
    """Build the continuous-time ``(A, B)`` for a batch of patients.

    Returns ``A`` of shape ``(B, 6, 6)`` and ``B`` of shape ``(B, 6, 2)``.
    Input vector is ``[u_ins (U/min), u_carb (mg/min)]``.
    """
    p2 = params.p2.reshape(-1)
    p3 = params.p3.reshape(-1)
    n = params.n.reshape(-1)
    V_I = params.V_I.reshape(-1)
    k_gri = params.k_gri.reshape(-1)
    k_abs = params.k_abs.reshape(-1)
    a = 1.0 / params.tmax_I.reshape(-1)

    batch = p2.shape[0]
    device, dtype = p2.device, p2.dtype
    A = torch.zeros(batch, N_STATES, N_STATES, device=device, dtype=dtype)
    B = torch.zeros(batch, N_STATES, N_INPUTS, device=device, dtype=dtype)

    # Subcutaneous insulin cascade.
    A[:, IDX_S1, IDX_S1] = -a
    A[:, IDX_S2, IDX_S1] = a
    A[:, IDX_S2, IDX_S2] = -a

    # Plasma insulin above basal. 1e6 uU/U divided by (V_I litres * 1000 mL/L).
    A[:, IDX_I, IDX_S2] = a * 1000.0 / V_I
    A[:, IDX_I, IDX_I] = -n

    # Gastric emptying and intestinal absorption.
    A[:, IDX_QSTO, IDX_QSTO] = -k_gri
    A[:, IDX_QGUT, IDX_QSTO] = k_gri
    A[:, IDX_QGUT, IDX_QGUT] = -k_abs

    # Remote insulin action.
    A[:, IDX_X, IDX_I] = p3
    A[:, IDX_X, IDX_X] = -p2

    # Inputs enter the first compartment of each cascade.
    B[:, IDX_S1, 0] = 1.0
    B[:, IDX_QSTO, 1] = 1.0
    return A, B


def discretise(A: Tensor, B: Tensor, dt: float) -> tuple[Tensor, Tensor]:
    """Exact zero-order-hold discretisation via one augmented matrix exponential.

    ``M = expm([[A, B], [0, 0]] * dt)`` yields ``Ad = M[:6, :6]`` and
    ``Bd = M[:6, 6:]`` simultaneously. Exact for piecewise-constant inputs.
    """
    batch = A.shape[0]
    size = N_STATES + N_INPUTS
    augmented = torch.zeros(batch, size, size, device=A.device, dtype=A.dtype)
    augmented[:, :N_STATES, :N_STATES] = A
    augmented[:, :N_STATES, N_STATES:] = B
    expanded = torch.linalg.matrix_exp(augmented * dt)
    return expanded[:, :N_STATES, :N_STATES], expanded[:, :N_STATES, N_STATES:]


def basal_steady_state(params: PatientParams, basal_u_per_min: Tensor) -> Tensor:
    """Analytic steady state under a constant basal rate, with empty gut.

    Solving ``A x + B u = 0`` for the insulin limb gives::

        S1* = S2* = u / a = u * tmax_I
        i*  = (1000 / V_I) * a * S2* / n = (1000 / V_I) * u / n
        X*  = p3 * i* / p2

    with ``Qsto* = Qgut* = 0``. Used as the initial condition at the start of the
    burn-in window; the burn-in then replays the subject's real insulin and meal
    history so the state entering the forecast window is data-driven rather than
    assumed.
    """
    u = basal_u_per_min.reshape(-1)
    tmax = params.tmax_I.reshape(-1)
    V_I = params.V_I.reshape(-1)
    n = params.n.reshape(-1)
    p2 = params.p2.reshape(-1)
    p3 = params.p3.reshape(-1)

    s = u * tmax
    i = (1000.0 / V_I) * u / n
    x = p3 * i / p2
    zero = torch.zeros_like(u)
    return torch.stack([s, s, i, zero, zero, x], dim=-1)


def simulate_compartments(
    params: PatientParams,
    u_ins: Tensor,
    u_carb: Tensor,
    *,
    dt: float = 1.0,
    x0: Tensor | None = None,
    t0_min: float = 0.0,
) -> CompartmentTrajectory:
    """Advance the linear compartments over a uniform grid.

    Parameters
    ----------
    params
        Batched patient parameters, batch size ``B``.
    u_ins
        Insulin delivery rate [U/min], shape ``(B, T)``. Element ``t`` is held
        constant over ``[t, t+dt)``. A bolus of ``d`` units at a single step is
        represented as ``d / dt`` for that step, which conserves mass exactly.
    u_carb
        Carbohydrate ingestion rate [mg/min], shape ``(B, T)``, same convention.
    dt
        Step in minutes. Must match the resolution of ``u_ins`` / ``u_carb``.
    x0
        Initial state ``(B, 6)``. Defaults to the zero state; callers should
        normally pass :func:`basal_steady_state`.
    t0_min
        Time label of the initial state.

    Returns
    -------
    CompartmentTrajectory
        ``states`` of shape ``(B, T+1, 6)`` -- the initial state followed by one
        state per input step.
    """
    if u_ins.shape != u_carb.shape:
        raise ValueError(f"u_ins {tuple(u_ins.shape)} != u_carb {tuple(u_carb.shape)}")
    if u_ins.ndim != 2:
        raise ValueError(f"expected (B, T) inputs, got {tuple(u_ins.shape)}")

    batch, steps = u_ins.shape
    A, B = system_matrices(params)
    if A.shape[0] != batch:
        raise ValueError(
            f"parameter batch {A.shape[0]} does not match input batch {batch}"
        )
    Ad, Bd = discretise(A, B, dt)

    state = (
        torch.zeros(batch, N_STATES, device=A.device, dtype=A.dtype)
        if x0 is None
        else x0.to(device=A.device, dtype=A.dtype)
    )
    inputs = torch.stack(
        [u_ins.to(A.dtype), u_carb.to(A.dtype)], dim=-1
    )  # (B, T, 2)

    # Sequential scan. The per-step work is a 6x6 matvec, so the cost is
    # dominated by Python overhead rather than arithmetic; T is a few hundred.
    out = [state]
    for step in range(steps):
        state = torch.einsum("bij,bj->bi", Ad, state) + torch.einsum(
            "bij,bj->bi", Bd, inputs[:, step, :]
        )
        out.append(state)

    time_min = t0_min + dt * torch.arange(
        steps + 1, device=A.device, dtype=A.dtype
    )
    return CompartmentTrajectory(
        states=torch.stack(out, dim=1), time_min=time_min, params=params
    )


__all__ = [
    "IDX_I",
    "IDX_QGUT",
    "IDX_QSTO",
    "IDX_S1",
    "IDX_S2",
    "IDX_X",
    "N_INPUTS",
    "N_STATES",
    "STATE_NAMES",
    "CompartmentTrajectory",
    "basal_steady_state",
    "discretise",
    "simulate_compartments",
    "system_matrices",
]
