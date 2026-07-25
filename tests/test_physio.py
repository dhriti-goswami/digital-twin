"""Verification of the mechanistic model.

These tests are the reason the physics can be trusted. Each one targets a
specific defect in the legacy implementation:

* ``test_bergman_residual_zero_on_analytic_solution`` -- the old residual set
  ``X = p3 * IOB`` algebraically and never integrated ``dX/dt``, so it was not
  the Bergman model and had no solution that zeroed it.
* ``test_insulin_mass_conserved`` / ``test_carb_mass_conserved`` -- the old COB
  kernel was an unnormalised exponential that jumped instantaneously and never
  decayed inside the window.
* ``test_iob_peaks_at_bolus_not_later`` -- the old IOB convolved with a
  *time-reversed activity* curve, so it was ~0 at the moment of the bolus and
  peaked 145 minutes afterwards.
* ``test_params_stay_within_bounds`` -- parameters must be unable to leave
  physiological ranges regardless of network output.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from twin.physio import (
    BOUNDS,
    ESTIMATED,
    PatientParams,
    basal_steady_state,
    glucose_residual,
    integrate_glucose,
    population_params,
    population_unconstrained,
    simulate_compartments,
    unconstrained_to_params,
)
from twin.physio.compartments import IDX_S2, discretise, system_matrices

DT = 1.0
DTYPE = torch.float64


@pytest.fixture
def params() -> PatientParams:
    return population_params(batch_size=1, dtype=DTYPE)


def _zero_inputs(steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    zeros = torch.zeros(1, steps, dtype=DTYPE)
    return zeros.clone(), zeros.clone()


# --------------------------------------------------------------------------- #
# Bergman glucose equation
# --------------------------------------------------------------------------- #


def test_bergman_residual_zero_on_analytic_solution():
    """With ``p3 = 0`` and no meal, glucose decays exponentially to ``G_b``.

    Then ``X`` stays identically zero and ``dG/dt = -p1 (G - G_b)``, whose
    solution is ``G(t) = G_b + (G_0 - G_b) exp(-p1 t)``. Feeding that solution and
    its exact derivative into the residual must give zero.
    """
    steps = 121
    p1 = 0.02
    G_b, G_0 = 120.0, 250.0

    estimated = {name: torch.tensor([v], dtype=DTYPE) for name, v in _mean_dict().items()}
    estimated["p1"] = torch.tensor([p1], dtype=DTYPE)
    estimated["p3"] = torch.tensor([0.0], dtype=DTYPE)  # X can never leave zero
    params = PatientParams.from_estimated(
        estimated,
        body_weight_kg=torch.tensor([70.0], dtype=DTYPE),
        G_b=torch.tensor([G_b], dtype=DTYPE),
        I_b=torch.tensor([10.0], dtype=DTYPE),
    )

    t = torch.arange(steps, dtype=DTYPE).unsqueeze(0) * DT
    G = G_b + (G_0 - G_b) * torch.exp(-p1 * t)
    dG_dt = -p1 * (G_0 - G_b) * torch.exp(-p1 * t)
    X = torch.zeros_like(G)
    Ra = torch.zeros_like(G)

    residual = glucose_residual(G, dG_dt, X, Ra, params)
    assert residual.abs().max() < 1e-10, f"max |residual| = {residual.abs().max():.3e}"


def test_integrate_glucose_matches_analytic_decay():
    """``integrate_glucose`` must reproduce the closed-form exponential decay."""
    steps = 241
    p1 = 0.02
    G_b, G_0 = 120.0, 250.0

    estimated = {name: torch.tensor([v], dtype=DTYPE) for name, v in _mean_dict().items()}
    estimated["p1"] = torch.tensor([p1], dtype=DTYPE)
    params = PatientParams.from_estimated(
        estimated,
        body_weight_kg=torch.tensor([70.0], dtype=DTYPE),
        G_b=torch.tensor([G_b], dtype=DTYPE),
        I_b=torch.tensor([10.0], dtype=DTYPE),
    )

    X = torch.zeros(1, steps, dtype=DTYPE)
    Ra = torch.zeros(1, steps, dtype=DTYPE)
    numeric = integrate_glucose(torch.tensor([G_0], dtype=DTYPE), X, Ra, params, dt=DT)

    t = torch.arange(steps, dtype=DTYPE) * DT
    analytic = G_b + (G_0 - G_b) * torch.exp(-p1 * t)
    assert torch.allclose(numeric[0], analytic, atol=1e-9)


def test_integrate_glucose_stable_when_p1_zero():
    """``p1 = 0`` must not produce a division by zero.

    Fixing ``p1 = 0`` for type 1 diabetes is a common control-oriented
    simplification (and is a configuration the ablation reaches), so the ``k -> 0``
    branch of the closed-form update has to be exercised explicitly. Note it is
    *not* the population default: Ward et al. 1991 measured non-zero glucose
    effectiveness in IDDM subjects.

    With ``p1 = 0``, ``X = 0`` and constant ``Ra``, glucose rises linearly.
    """
    steps = 61
    ra = 2.0  # mg/dL/min

    estimated = {name: torch.tensor([v], dtype=DTYPE) for name, v in _mean_dict().items()}
    estimated["p1"] = torch.tensor([0.0], dtype=DTYPE)
    params = PatientParams.from_estimated(
        estimated,
        body_weight_kg=torch.tensor([70.0], dtype=DTYPE),
        G_b=torch.tensor([120.0], dtype=DTYPE),
        I_b=torch.tensor([10.0], dtype=DTYPE),
    )
    assert float(params.p1[0]) == 0.0

    X = torch.zeros(1, steps, dtype=DTYPE)
    Ra = torch.full((1, steps), ra, dtype=DTYPE)
    G = integrate_glucose(torch.tensor([100.0], dtype=DTYPE), X, Ra, params, dt=DT)

    assert torch.isfinite(G).all()
    expected = 100.0 + ra * torch.arange(steps, dtype=DTYPE) * DT
    assert torch.allclose(G[0], expected, atol=1e-9)


def test_population_p1_is_nonzero_per_ward_1991():
    """The population default must reflect the measured IDDM value, not the
    ``p1 = 0`` modelling shortcut."""
    params = population_params(batch_size=1, dtype=DTYPE)
    assert float(params.p1[0]) > 0.0
    # Ward et al. 1991 report S_G = 1.0-1.6e-2 /min in IDDM subjects.
    assert 0.010 <= float(params.p1[0]) <= 0.016


def test_population_insulin_sensitivity_matches_ward_1991():
    """``S_I = p3 / p2`` must equal the measured IDDM value by construction."""
    from twin.physio.params import S_I_IDDM_MEAN

    params = population_params(batch_size=1, dtype=DTYPE)
    assert float(params.S_I[0]) == pytest.approx(S_I_IDDM_MEAN, rel=1e-9)


def _meal_scenario_residual(dt: float) -> tuple[torch.Tensor, int]:
    """Residual of an integrated meal-and-bolus trajectory, via central differences.

    Returns the absolute residual over the interior grid (index ``j`` corresponds
    to grid point ``j + 1``) together with the index of the impulse.
    """
    steps = int(360 / dt) + 1
    meal_index = int(60 / dt)
    params = population_params(batch_size=1, dtype=DTYPE)
    u_ins = torch.zeros(1, steps, dtype=DTYPE)
    u_carb = torch.zeros(1, steps, dtype=DTYPE)
    u_ins[0, meal_index] = 6.0 / dt  # 6 U bolus, as a rate
    u_carb[0, meal_index] = 60_000.0 / dt  # 60 g carbohydrate, as a rate
    u_ins += 0.02  # basal, U/min

    trajectory = simulate_compartments(params, u_ins, u_carb, dt=dt)
    X = trajectory.X[:, :steps]
    Ra = trajectory.Ra_mgdl_per_min[:, :steps]
    G = integrate_glucose(torch.tensor([140.0], dtype=DTYPE), X, Ra, params, dt=dt)

    dG_dt = (G[:, 2:] - G[:, :-2]) / (2 * dt)
    residual = glucose_residual(G[:, 1:-1], dG_dt, X[:, 1:-1], Ra[:, 1:-1], params)
    return residual.abs()[0], meal_index


def test_residual_zero_on_integrated_trajectory():
    """Forward solution and residual must be mutually consistent.

    Integrate glucose through a realistic meal-and-bolus scenario, differentiate
    numerically, and confirm the residual vanishes. This catches a sign error or a
    unit mismatch between the two functions, which no analytic special case would
    expose.

    The neighbourhood of the impulse is excluded: an instantaneous meal makes the
    true ``dG/dt`` discontinuous there, so a central difference is first-order at
    best. That is a limitation of differencing in this test, not of the model --
    during training ``dG/dt`` comes analytically from the spline head and is never
    differenced.
    """
    residual, meal_index = _meal_scenario_residual(DT)
    mask = torch.ones_like(residual, dtype=torch.bool)
    mask[max(0, meal_index - 4) : meal_index + 3] = False

    assert residual[mask].max() < 1e-2, f"max |residual| = {residual[mask].max():.3e}"
    assert residual.median() < 1e-4, f"median |residual| = {residual.median():.3e}"


def test_glucose_integrator_is_second_order():
    """Halving ``dt`` must cut the residual by ~4x, confirming O(dt^2) accuracy.

    Sampling ``k`` and ``c`` at the step start would give only O(dt) and leave a
    sustained residual after a meal; the step-averaged (midpoint) coefficients in
    :func:`integrate_glucose` are what buy the second order. This test is what
    keeps that from silently regressing.
    """
    medians = []
    for dt in (1.0, 0.5, 0.25):
        residual, _ = _meal_scenario_residual(dt)
        medians.append(float(residual.median()))

    for coarse, fine in zip(medians, medians[1:], strict=False):
        ratio = coarse / fine
        assert 3.0 < ratio < 5.5, f"convergence ratio {ratio:.2f} is not second order"


# --------------------------------------------------------------------------- #
# Mass conservation
# --------------------------------------------------------------------------- #


def test_insulin_mass_conserved(params: PatientParams):
    """All delivered insulin must eventually appear in plasma, and none extra.

    Total appearance is ``integral of a * S2 dt``; it must equal the delivered
    dose once the compartments have emptied.
    """
    steps = 1200  # 20 h, >> tmax_I
    dose = 5.0
    u_ins, u_carb = _zero_inputs(steps)
    u_ins[0, 0] = dose / DT

    trajectory = simulate_compartments(params, u_ins, u_carb, dt=DT)
    a = 1.0 / params.tmax_I
    appearance = a.unsqueeze(-1) * trajectory.S2
    total = torch.trapezoid(appearance, dx=DT, dim=-1)

    assert trajectory.iob_u[0, -1].abs() < 1e-6, "compartments did not empty"
    assert abs(float(total[0]) - dose) < 1e-4, f"appeared {float(total[0]):.6f} of {dose} U"


def test_carb_mass_conserved(params: PatientParams):
    """Glucose appearance must integrate to ``f * D``, not more and not less."""
    steps = 1500
    carbs_g = 60.0
    u_ins, u_carb = _zero_inputs(steps)
    u_carb[0, 0] = carbs_g * 1000.0 / DT  # mg/min

    trajectory = simulate_compartments(params, u_ins, u_carb, dt=DT)
    total_mg = torch.trapezoid(trajectory.Ra_mg_per_min, dx=DT, dim=-1)
    expected_mg = float(params.f[0]) * carbs_g * 1000.0

    assert trajectory.cob_g[0, -1].abs() < 1e-6, "gut did not empty"
    relative_error = abs(float(total_mg[0]) - expected_mg) / expected_mg
    assert relative_error < 1e-5, f"appeared {float(total_mg[0]):.1f} mg vs {expected_mg:.1f} mg"


def test_cob_starts_at_ingested_amount_and_decays(params: PatientParams):
    """COB must equal the ingested amount right after the meal, then fall.

    The legacy ``_compute_cob`` used an unnormalised ``exp(-k/36)`` kernel, so COB
    neither started at the ingested mass nor reached zero inside the window.
    """
    # With the verified k_abs = 0.0167 /min (Lehmann & Deutsch 1992, 1/h), the gut
    # empties with a ~60 min time constant, so clearing to 1e-6 of the ingested
    # mass needs on the order of 14 time constants.
    steps = 1500
    carbs_g = 45.0
    u_ins, u_carb = _zero_inputs(steps)
    u_carb[0, 0] = carbs_g * 1000.0 / DT

    cob = simulate_compartments(params, u_ins, u_carb, dt=DT).cob_g[0]

    assert cob[0] == pytest.approx(0.0, abs=1e-9), "COB nonzero before ingestion"
    assert cob[1] == pytest.approx(carbs_g, rel=2e-3), f"COB after meal = {float(cob[1]):.3f} g"
    differences = torch.diff(cob[1:])
    assert (differences <= 1e-9).all(), "COB must be non-increasing after ingestion"
    assert cob[-1] / carbs_g < 1e-6, f"COB must reach zero, ended at {float(cob[-1]):.3e} g"


# --------------------------------------------------------------------------- #
# IOB shape -- the legacy kernel was time-reversed
# --------------------------------------------------------------------------- #


def test_iob_peaks_at_bolus_not_later(params: PatientParams):
    """IOB must be maximal immediately after the bolus and decrease thereafter.

    Regression guard for ``ode_features._compute_iob``, which convolved the bolus
    train with a reversed *activity* curve. That put ~0 weight on a bolus at the
    moment it was given and peaked roughly 145 minutes later -- the feature was
    neither insulin-remaining nor insulin-activity.
    """
    # IOB decays as exp(-t/tmax)*(1 + t/tmax), so reaching 1e-6 of the dose needs
    # roughly 19 time constants; 1500 min covers it for tmax_I up to 90 min.
    steps = 1500
    dose = 8.0
    bolus_step = 10
    u_ins, u_carb = _zero_inputs(steps)
    u_ins[0, bolus_step] = dose / DT

    iob = simulate_compartments(params, u_ins, u_carb, dt=DT).iob_u[0]

    peak_index = int(torch.argmax(iob))
    assert peak_index == bolus_step + 1, f"IOB peaks at index {peak_index}, expected {bolus_step + 1}"
    assert iob[bolus_step + 1] == pytest.approx(dose, rel=1e-3)
    assert iob[bolus_step] == pytest.approx(0.0, abs=1e-12), "IOB nonzero before the bolus"
    after_peak = torch.diff(iob[peak_index:])
    assert (after_peak <= 1e-12).all(), "IOB must be monotonically non-increasing after the bolus"
    assert iob[-1] / dose < 1e-6, f"IOB must decay to zero, ended at {float(iob[-1]):.3e} U"


def test_insulin_action_peaks_after_bolus(params: PatientParams):
    """``X`` must lag the bolus -- insulin action is delayed, not instantaneous."""
    steps = 600
    u_ins, u_carb = _zero_inputs(steps)
    u_ins[0, 0] = 8.0 / DT

    X = simulate_compartments(params, u_ins, u_carb, dt=DT).X[0]
    peak_minutes = int(torch.argmax(X)) * DT

    assert X[0] == 0.0, "X must start at zero"
    assert 60.0 < peak_minutes < 200.0, f"X peaks at {peak_minutes} min, outside plausible range"


# --------------------------------------------------------------------------- #
# Steady state and discretisation
# --------------------------------------------------------------------------- #


def test_insulin_action_is_zero_at_a_consistent_basal():
    """``X`` must vanish at basal when ``I_b`` matches the basal rate.

    Bergman drives remote insulin action by insulin *above basal*. If ``X`` is
    driven by total plasma insulin instead, it is positive at basal, which destroys
    the glucose equilibrium (see the next test).
    """
    from twin.physio import basal_insulin_concentration

    basal = torch.tensor([0.02], dtype=DTYPE)
    reference = population_params(batch_size=1, dtype=DTYPE)
    consistent_ib = basal_insulin_concentration(reference, basal)

    estimated = {name: torch.tensor([v], dtype=DTYPE) for name, v in _mean_dict().items()}
    params = PatientParams.from_estimated(
        estimated,
        body_weight_kg=torch.tensor([70.0], dtype=DTYPE),
        G_b=torch.tensor([120.0], dtype=DTYPE),
        I_b=consistent_ib,
    )

    x0 = basal_steady_state(params, basal)
    from twin.physio.compartments import IDX_X

    assert abs(float(x0[0, IDX_X])) < 1e-12


def test_basal_is_a_glucose_equilibrium():
    """Starting at ``G_b`` under basal insulin, glucose must not drift.

    This is the property whose absence made the mechanistic prior collapse by more
    than 200 mg/dL over a two-hour forecast: with ``X > 0`` at basal, the glucose
    equation pulled toward ``p1 G_b / (p1 + X)``, far below basal, with no stimulus
    present at all.
    """
    from twin.physio import basal_insulin_concentration

    steps = 241
    basal = torch.tensor([0.02], dtype=DTYPE)
    reference = population_params(batch_size=1, dtype=DTYPE)
    estimated = {name: torch.tensor([v], dtype=DTYPE) for name, v in _mean_dict().items()}
    params = PatientParams.from_estimated(
        estimated,
        body_weight_kg=torch.tensor([70.0], dtype=DTYPE),
        G_b=torch.tensor([120.0], dtype=DTYPE),
        I_b=basal_insulin_concentration(reference, basal),
    )

    u_ins = basal.unsqueeze(-1).expand(1, steps).clone()
    u_carb = torch.zeros(1, steps, dtype=DTYPE)
    trajectory = simulate_compartments(
        params, u_ins, u_carb, dt=DT, x0=basal_steady_state(params, basal)
    )
    glucose = integrate_glucose(
        params.G_b,
        trajectory.X[:, :steps],
        trajectory.Ra_mgdl_per_min[:, :steps],
        params,
        dt=DT,
    )
    drift = (glucose[0] - params.G_b[0]).abs().max()
    assert drift < 1e-9, f"glucose drifted {float(drift):.4f} mg/dL from basal"


def test_inconsistent_basal_insulin_breaks_the_equilibrium():
    """Documents the failure mode, so the fix cannot be silently reverted."""
    steps = 121
    basal = torch.tensor([0.02], dtype=DTYPE)
    estimated = {name: torch.tensor([v], dtype=DTYPE) for name, v in _mean_dict().items()}
    params = PatientParams.from_estimated(
        estimated,
        body_weight_kg=torch.tensor([70.0], dtype=DTYPE),
        G_b=torch.tensor([120.0], dtype=DTYPE),
        # Deliberately wrong: not the concentration implied by the basal rate.
        I_b=torch.tensor([0.0], dtype=DTYPE),
    )
    u_ins = basal.unsqueeze(-1).expand(1, steps).clone()
    u_carb = torch.zeros(1, steps, dtype=DTYPE)
    trajectory = simulate_compartments(
        params, u_ins, u_carb, dt=DT, x0=basal_steady_state(params, basal)
    )
    glucose = integrate_glucose(
        params.G_b,
        trajectory.X[:, :steps],
        trajectory.Ra_mgdl_per_min[:, :steps],
        params,
        dt=DT,
    )
    assert (glucose[0] - params.G_b[0]).abs().max() > 20.0


def test_basal_steady_state_is_stationary(params: PatientParams):
    """Starting at the analytic basal steady state, nothing may drift."""
    steps = 300
    basal = torch.tensor([0.02], dtype=DTYPE)  # U/min
    x0 = basal_steady_state(params, basal)
    u_ins = basal.unsqueeze(-1).expand(1, steps).clone()
    u_carb = torch.zeros(1, steps, dtype=DTYPE)

    states = simulate_compartments(params, u_ins, u_carb, dt=DT, x0=x0).states
    drift = (states[0, -1] - states[0, 0]).abs().max()
    assert drift < 1e-9, f"steady state drifted by {float(drift):.3e}"


def test_discretisation_is_exact_for_scalar_decay():
    """The matrix-exponential discretisation must be exact, not approximate.

    ``S1`` alone obeys ``dS1/dt = -a S1``, whose exact solution is known. This
    isolates the discretisation from the rest of the cascade.
    """
    params = population_params(batch_size=1, dtype=DTYPE)
    A, B = system_matrices(params)
    Ad, _ = discretise(A, B, dt=5.0)
    a = 1.0 / float(params.tmax_I[0])
    assert Ad[0, 0, 0].item() == pytest.approx(np.exp(-a * 5.0), rel=1e-12)


def test_input_step_size_conserves_dose(params: PatientParams):
    """A dose spread as ``d/dt`` over one step must deliver ``d`` units for any dt.

    Guards the convention that pump inputs are *rates*, which is the only way the
    mass balance holds when the grid resolution changes.
    """
    dose = 4.0
    totals = []
    for dt in (0.5, 1.0, 2.0):
        steps = int(1200 / dt)
        u_ins = torch.zeros(1, steps, dtype=DTYPE)
        u_carb = torch.zeros(1, steps, dtype=DTYPE)
        u_ins[0, 0] = dose / dt
        trajectory = simulate_compartments(params, u_ins, u_carb, dt=dt)
        a = 1.0 / params.tmax_I
        totals.append(float(torch.trapezoid(a.unsqueeze(-1) * trajectory.S2, dx=dt, dim=-1)[0]))

    for total in totals:
        assert total == pytest.approx(dose, rel=1e-3), f"delivered {total:.6f} of {dose} U"


# --------------------------------------------------------------------------- #
# Parameter bounds
# --------------------------------------------------------------------------- #


def test_params_stay_within_bounds():
    """No network output, however extreme, may push a parameter out of range."""
    extreme = torch.tensor(
        [
            [-1e4] * len(ESTIMATED),
            [1e4] * len(ESTIMATED),
            [0.0] * len(ESTIMATED),
        ],
        dtype=DTYPE,
    )
    resolved = unconstrained_to_params(extreme)
    for name, value in resolved.items():
        bound = BOUNDS[name]
        assert (value >= bound.low).all(), f"{name} below {bound.low}"
        assert (value <= bound.high).all(), f"{name} above {bound.high}"


def test_zero_input_maps_to_interval_midpoint():
    """A zero pre-activation must give the midpoint, so init is sensible."""
    resolved = unconstrained_to_params(torch.zeros(1, len(ESTIMATED), dtype=DTYPE))
    for name, value in resolved.items():
        bound = BOUNDS[name]
        assert float(value[0]) == pytest.approx((bound.low + bound.high) / 2, rel=1e-9)


def test_population_unconstrained_round_trips():
    """The population-mean initialiser must decode back to the population means."""
    from twin.physio.params import POPULATION_MEANS

    resolved = unconstrained_to_params(population_unconstrained().unsqueeze(0))
    for name, value in resolved.items():
        # p1's mean sits exactly on its lower bound, so the logit is clamped;
        # a small absolute tolerance is expected there.
        assert float(value[0]) == pytest.approx(
            POPULATION_MEANS[name], rel=1e-3, abs=1e-5 * max(1.0, abs(POPULATION_MEANS[name]))
        ), name


def test_insulin_sensitivity_definition():
    """``S_I`` must be exactly ``p3 / p2``."""
    params = population_params(batch_size=3, dtype=DTYPE)
    assert torch.allclose(params.S_I, params.p3 / params.p2)


def test_provisional_bounds_block_reporting():
    """Reporting must refuse to run while any bound is still PROVISIONAL.

    This is the guard that stops a development placeholder reaching a paper
    table. Once ``docs/CITATIONS.md`` is complete and the bounds carry real
    sources, this test flips to asserting the call succeeds.
    """
    from twin.physio.params import PROVISIONAL, assert_bounds_sourced

    unsourced = [name for name, bound in BOUNDS.items() if not bound.sourced]
    if unsourced:
        with pytest.raises(RuntimeError, match=PROVISIONAL):
            assert_bounds_sourced()
    else:
        assert_bounds_sourced()


# --------------------------------------------------------------------------- #
# Batching and differentiability
# --------------------------------------------------------------------------- #


def test_batched_simulation_matches_individual():
    """A batch must give identical results to simulating each member alone."""
    steps = 200
    batch = 4
    params = population_params(batch_size=batch, dtype=DTYPE)
    generator = torch.Generator().manual_seed(0)
    u_ins = torch.rand(batch, steps, generator=generator, dtype=DTYPE) * 0.05
    u_carb = torch.zeros(batch, steps, dtype=DTYPE)
    u_carb[:, 30] = 40_000.0

    batched = simulate_compartments(params, u_ins, u_carb, dt=DT).states
    for index in range(batch):
        single = simulate_compartments(
            population_params(batch_size=1, dtype=DTYPE),
            u_ins[index : index + 1],
            u_carb[index : index + 1],
            dt=DT,
        ).states
        assert torch.allclose(batched[index], single[0], atol=1e-10)


def test_gradients_flow_to_parameters():
    """The residual must be differentiable w.r.t. the estimated parameters.

    Without this the physics term cannot train the parameter encoder, which is
    the entire mechanism behind the patient-specific insulin-sensitivity claim.
    """
    steps = 120
    raw = population_unconstrained().unsqueeze(0).clone().requires_grad_(True)
    estimated = unconstrained_to_params(raw)
    params = PatientParams.from_estimated(
        estimated,
        body_weight_kg=torch.tensor([70.0]),
        G_b=torch.tensor([120.0]),
        I_b=torch.tensor([10.0]),
    )
    u_ins = torch.zeros(1, steps)
    u_carb = torch.zeros(1, steps)
    u_ins[0, 10] = 5.0
    u_carb[0, 10] = 50_000.0

    trajectory = simulate_compartments(params, u_ins, u_carb, dt=DT)
    G = integrate_glucose(
        torch.tensor([150.0]), trajectory.X[:, :steps], trajectory.Ra_mgdl_per_min[:, :steps], params
    )
    G.sum().backward()

    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert raw.grad.abs().sum() > 0, "no gradient reached the parameters"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _mean_dict() -> dict[str, float]:
    from twin.physio.params import POPULATION_MEANS

    return dict(POPULATION_MEANS)


def test_state_index_constants_match_names():
    """Index constants and ``STATE_NAMES`` must not drift apart."""
    from twin.physio.compartments import STATE_NAMES

    assert STATE_NAMES[IDX_S2] == "S2"
    assert len(STATE_NAMES) == len(set(STATE_NAMES))


def test_chunked_advance_matches_sequential_scan():
    """``advance_compartments`` must be exact, not an approximation.

    The burn-in is advanced in chunks of matrix powers rather than step by step,
    because a 144-step scan of tiny matrix-vector products was 73% of the model's
    forward pass. Speed is only acceptable if the result is identical.
    """
    from twin.physio import advance_compartments, basal_insulin_concentration

    steps = 144
    reference = population_params(batch_size=4, dtype=DTYPE)
    basal = torch.full((4,), 0.02, dtype=DTYPE)
    estimated = {name: getattr(reference, name) for name in
                 ("p1", "p2", "p3", "n", "tmax_I", "k_gri", "k_abs", "f")}
    estimated["V_G_per_kg"] = reference.V_G / 70.0
    estimated["V_I_per_kg"] = reference.V_I / 70.0
    params = PatientParams.from_estimated(
        estimated,
        body_weight_kg=torch.full((4,), 70.0, dtype=DTYPE),
        G_b=torch.full((4,), 120.0, dtype=DTYPE),
        I_b=basal_insulin_concentration(reference, basal),
    )

    generator = torch.Generator().manual_seed(0)
    u_ins = torch.rand(4, steps, generator=generator, dtype=DTYPE) * 0.05
    u_carb = torch.zeros(4, steps, dtype=DTYPE)
    u_carb[:, 30] = 40_000.0
    u_carb[:, 90] = 60_000.0
    x0 = basal_steady_state(params, basal)

    sequential = simulate_compartments(params, u_ins, u_carb, dt=5.0, x0=x0).states[:, -1]
    for chunk_size in (1, 8, 24, 48):
        chunked = advance_compartments(
            params, u_ins, u_carb, dt=5.0, x0=x0, chunk_size=chunk_size
        )
        assert torch.allclose(chunked, sequential, atol=1e-9), f"chunk_size={chunk_size}"
