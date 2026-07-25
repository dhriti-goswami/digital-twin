"""Mechanistic physiology: Bergman minimal model and its input sub-models.

The same parameterisation serves three purposes, which is deliberate:

1. **Features.** ``IOB = S1 + S2`` and ``COB = Qsto + Qgut`` come out of the
   compartment model, so they are mass-conserving by construction rather than
   hand-rolled convolution kernels.
2. **Physics loss.** :func:`~twin.physio.bergman.glucose_residual` is evaluated
   at collocation points using the same states.
3. **Mechanistic prior.** :func:`~twin.physio.bergman.integrate_glucose` gives a
   forward forecast the network corrects, rather than being asked to rediscover
   physiology from a penalty term.

Everything is batched, differentiable, and free of numerical integrators: the
insulin and gut limbs are linear (exact matrix-exponential propagation) and the
glucose equation is linear once ``X(t)`` is known.
"""

from twin.physio.bergman import (
    glucose_residual,
    insulin_sensitivity,
    integrate_glucose,
    residual_scale,
    steady_state_glucose,
)
from twin.physio.compartments import (
    N_STATES,
    STATE_NAMES,
    CompartmentTrajectory,
    basal_steady_state,
    discretise,
    simulate_compartments,
    system_matrices,
)
from twin.physio.params import (
    BOUNDS,
    ESTIMATED,
    N_ESTIMATED,
    POPULATION_MEANS,
    Bound,
    PatientParams,
    assert_bounds_sourced,
    population_params,
    population_unconstrained,
    unconstrained_to_params,
)
from twin.physio.spline import SplineEvaluator, SplineGrid

__all__ = [
    "BOUNDS",
    "ESTIMATED",
    "N_ESTIMATED",
    "N_STATES",
    "POPULATION_MEANS",
    "STATE_NAMES",
    "Bound",
    "CompartmentTrajectory",
    "PatientParams",
    "SplineEvaluator",
    "SplineGrid",
    "assert_bounds_sourced",
    "basal_steady_state",
    "discretise",
    "glucose_residual",
    "insulin_sensitivity",
    "integrate_glucose",
    "population_params",
    "population_unconstrained",
    "residual_scale",
    "simulate_compartments",
    "steady_state_glucose",
    "system_matrices",
    "unconstrained_to_params",
]
