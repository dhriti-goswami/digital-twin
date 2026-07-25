"""Verification of the continuous-time B-spline forecast head.

The head's contract is what makes the physics loss meaningful, so each clause is
tested:

* ``G(0)`` equals the last observed CGM value *identically*, for any
  coefficients (anchoring by construction, not by penalty).
* ``dG/dt`` is the exact analytic derivative of ``G(t)``, not a difference.
* The reported horizon values are evaluations of the same function the residual
  constrains -- so there is no train/report mismatch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from twin.physio.spline import (
    DEGREE,
    SplineEvaluator,
    SplineGrid,
    clamped_uniform_knots,
    design_matrices,
)

HORIZONS = (30, 60, 90, 120)


@pytest.fixture
def evaluator() -> SplineEvaluator:
    grid = SplineGrid.build(horizons_min=HORIZONS, n_collocation=121)
    return SplineEvaluator(grid, n_basis=12)


# --------------------------------------------------------------------------- #
# Knots and basis
# --------------------------------------------------------------------------- #


def test_knot_count_matches_basis_count():
    """``n_basis = len(knots) - degree - 1`` must hold exactly."""
    for n_basis in (4, 8, 12, 20):
        knots = clamped_uniform_knots(120.0, n_basis)
        assert len(knots) - DEGREE - 1 == n_basis


def test_knots_are_clamped():
    """End knots must repeat ``degree + 1`` times so the basis spans endpoints."""
    knots = clamped_uniform_knots(120.0, 12)
    assert np.all(knots[: DEGREE + 1] == 0.0)
    assert np.all(knots[-(DEGREE + 1) :] == 120.0)
    assert np.all(np.diff(knots) >= 0), "knot vector must be non-decreasing"


def test_basis_too_small_is_rejected():
    with pytest.raises(ValueError, match="too small"):
        clamped_uniform_knots(120.0, n_basis=2)


def test_basis_forms_partition_of_unity():
    """Cubic B-spline bases sum to 1 everywhere on a clamped knot vector.

    A failure here means the basis is malformed and every coefficient would be
    scaled by an unknown, position-dependent factor.
    """
    grid = np.linspace(0.0, 120.0, 241)
    basis, _, _ = design_matrices(grid, 120.0, 12)
    assert np.allclose(basis.sum(axis=1), 1.0, atol=1e-10)


def test_derivative_basis_sums_to_zero():
    """The derivative of a partition of unity is identically zero."""
    grid = np.linspace(0.0, 120.0, 241)
    _, derivative, _ = design_matrices(grid, 120.0, 12)
    assert np.allclose(derivative.sum(axis=1), 0.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# Anchoring
# --------------------------------------------------------------------------- #


def test_value_at_zero_is_exactly_G0(evaluator: SplineEvaluator):
    """The anchoring identity must hold for arbitrary coefficients."""
    torch.manual_seed(0)
    coefficients = torch.randn(16, evaluator.n_basis) * 50.0
    G0 = torch.rand(16) * 200.0 + 60.0

    values = evaluator.value(coefficients, G0, at="collocation")
    assert torch.allclose(values[:, 0], G0, atol=1e-4)


def test_zero_coefficients_give_flat_forecast(evaluator: SplineEvaluator):
    """With zero coefficients the forecast is the persistence prediction.

    A useful property: the model's initialisation is the naive baseline, so it
    starts from a sensible place rather than from noise.
    """
    G0 = torch.tensor([150.0, 90.0])
    coefficients = torch.zeros(2, evaluator.n_basis)

    values = evaluator.value(coefficients, G0, at="horizon")
    derivative = evaluator.derivative(coefficients)

    assert torch.allclose(values, G0.unsqueeze(-1).expand_as(values), atol=1e-5)
    assert torch.allclose(derivative, torch.zeros_like(derivative), atol=1e-6)


# --------------------------------------------------------------------------- #
# Derivative correctness
# --------------------------------------------------------------------------- #


def test_derivative_matches_finite_difference(evaluator: SplineEvaluator):
    """The analytic derivative must agree with a fine central difference.

    This is the property the legacy pipeline lacked: its residual differenced
    across 30-minute gaps against per-minute rate constants.
    """
    torch.manual_seed(1)
    coefficients = (torch.randn(4, evaluator.n_basis) * 30.0).double()
    G0 = torch.full((4,), 140.0, dtype=torch.float64)

    epsilon = 1e-5
    t_max = evaluator.grid.t_max
    # Interior points only: a one-sided difference at the boundary would not be
    # comparable to a central difference. All three stencil offsets go into one
    # grid so the basis is constructed once.
    query = np.linspace(1.0, t_max - 1.0, 40)
    stencil = np.concatenate([query - epsilon, query, query + epsilon])
    stencil_evaluator = SplineEvaluator(
        SplineGrid(collocation_min=stencil, horizon_min=np.array(HORIZONS, dtype=float)),
        n_basis=evaluator.n_basis,
    )

    values = stencil_evaluator.value(coefficients, G0, at="collocation")
    analytic = stencil_evaluator.derivative(coefficients)

    count = len(query)
    numeric = (values[:, 2 * count :] - values[:, :count]) / (2 * epsilon)
    assert torch.allclose(numeric, analytic[:, count : 2 * count], atol=1e-6)


def test_integrated_derivative_recovers_value(evaluator: SplineEvaluator):
    """Integrating ``dG/dt`` from 0 must reproduce ``G(t) - G(0)``.

    An independent check on the value/derivative pair under the fundamental
    theorem of calculus. Rather than pick an absolute tolerance, this asserts the
    *convergence order*: the only discrepancy should be trapezoid truncation,
    which is O(h^2), so a 10x finer grid must cut the error ~100x. Any
    inconsistency between the two design matrices would instead leave a
    grid-independent error floor and the ratio would collapse to 1.
    """
    torch.manual_seed(2)
    coefficients = (torch.randn(3, evaluator.n_basis) * 20.0).double()
    G0 = torch.full((3,), 130.0, dtype=torch.float64)

    errors = []
    for n_points in (1201, 12001, 120001):
        dense = SplineGrid(
            collocation_min=np.linspace(0.0, 120.0, n_points),
            horizon_min=np.array(HORIZONS, dtype=float),
        )
        fine = SplineEvaluator(dense, n_basis=evaluator.n_basis)
        values = fine.value(coefficients, G0)
        derivative = fine.derivative(coefficients)

        step = float(dense.collocation_min[1] - dense.collocation_min[0])
        integrated = torch.cumulative_trapezoid(derivative, dx=step, dim=-1)
        errors.append(float((integrated - (values[:, 1:] - G0.unsqueeze(-1))).abs().max()))

    assert errors[0] < 1e-2, f"even the coarse grid should agree closely, got {errors[0]:.3e}"
    for coarse, fine_error in zip(errors, errors[1:], strict=False):
        ratio = coarse / fine_error
        assert 50.0 < ratio < 200.0, f"convergence ratio {ratio:.1f} is not second order"


# --------------------------------------------------------------------------- #
# Horizon consistency -- no train/report mismatch
# --------------------------------------------------------------------------- #


def test_horizon_values_match_collocation_at_same_times(evaluator: SplineEvaluator):
    """Reported horizons must equal the constrained function at those times.

    With a 121-point grid over 0..120 min, the horizons land exactly on
    collocation indices 30/60/90/120. If these disagreed, the physics term would
    be constraining something other than what is reported.
    """
    torch.manual_seed(3)
    coefficients = torch.randn(5, evaluator.n_basis) * 25.0
    G0 = torch.full((5,), 160.0)

    collocation = evaluator.value(coefficients, G0, at="collocation")
    horizon = evaluator.value(coefficients, G0, at="horizon")

    for index, minutes in enumerate(HORIZONS):
        position = int(np.argmin(np.abs(evaluator.grid.collocation_min - minutes)))
        assert evaluator.grid.collocation_min[position] == pytest.approx(minutes)
        assert torch.allclose(horizon[:, index], collocation[:, position], atol=1e-4)


def test_gradients_flow_to_coefficients(evaluator: SplineEvaluator):
    """Both value and derivative paths must be differentiable."""
    coefficients = torch.zeros(2, evaluator.n_basis, requires_grad=True)
    G0 = torch.full((2,), 140.0)

    values, derivative, horizons = evaluator(coefficients, G0)
    (values.sum() + derivative.sum() + horizons.sum()).backward()

    assert coefficients.grad is not None
    assert torch.isfinite(coefficients.grad).all()
    assert coefficients.grad.abs().sum() > 0


def test_buffers_are_saved_in_state_dict(evaluator: SplineEvaluator):
    """The grid must travel with the checkpoint.

    A checkpoint that does not record its own evaluation grid cannot be
    reproduced, and silently changing ``n_collocation`` between train and eval
    would go unnoticed.
    """
    keys = set(evaluator.state_dict())
    for expected in (
        "collocation_basis",
        "collocation_derivative",
        "horizon_basis",
        "basis_at_zero",
        "collocation_min",
        "horizon_min",
    ):
        assert expected in keys, f"{expected} missing from state_dict"


def test_expressive_enough_to_fit_a_realistic_curve(evaluator: SplineEvaluator):
    """12 cubic bases must represent a postprandial excursion to within ~1 mg/dL.

    Sanity check on capacity: if the basis cannot represent a plausible glucose
    trajectory, the residual would be fighting the head's own bias rather than
    shaping the forecast.
    """
    times = torch.tensor(evaluator.grid.collocation_min, dtype=torch.float64)
    G0 = 120.0
    # A meal-like rise and partial fall. The head anchors G(0) = G0, so the
    # target must satisfy the same constraint or the test measures the anchor
    # rather than the basis capacity.
    bump = torch.exp(-((times - 55.0) ** 2) / (2 * 35.0**2))
    target = G0 + 90.0 * (bump - bump[0])

    basis = (evaluator.collocation_basis - evaluator.basis_at_zero.unsqueeze(0)).double()
    solution = torch.linalg.lstsq(basis, (target - G0).unsqueeze(-1)).solution
    fitted = G0 + (basis @ solution).squeeze(-1)

    assert (fitted - target).abs().max() < 1.0
