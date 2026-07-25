"""Verification of the metrics layer.

The error-grid tests are the important ones. Each named point below was scored
*incorrectly* by one of the two legacy implementations, and every one of those
errors moved the reported clinical-safety number in the flattering direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from twin.metrics import (
    CLARKE_ZONES,
    PARKES_ZONES,
    MetricError,
    SubjectPredictions,
    UnverifiedBoundaryError,
    accuracy_by_horizon,
    across_subject_summary,
    bootstrap_ci,
    clarke_zone,
    coefficient_of_variation,
    format_mean_sd,
    hbgi,
    holm_bonferroni,
    hypoglycaemia_detection,
    lbgi,
    mae,
    mard,
    paired_comparison,
    parkes_zone,
    per_subject_table,
    pooled_metrics,
    prediction_lag_min,
    r2,
    range_distribution,
    risk_transform,
    rmse,
    zone_field,
    zone_summary,
)

HORIZONS = (30, 60, 90, 120)


def zone_of(reference: float, predicted: float) -> str:
    return str(clarke_zone(np.array([reference]), np.array([predicted]))[0])


# --------------------------------------------------------------------------- #
# Clarke error grid -- regression guards for the legacy defects
# --------------------------------------------------------------------------- #


def test_clarke_all_zones_reachable():
    """Every zone A-E must be assignable.

    ``evaluate_ohio.py`` lines 338-339 were byte-identical duplicates of 335-336,
    so zone E could never be produced and genuine erroneous-treatment predictions
    were all reported as D.
    """
    witnesses = {
        "A": (100.0, 110.0),
        "B": (150.0, 200.0),
        "C": (100.0, 250.0),
        "D": (300.0, 120.0),
        "E": (60.0, 200.0),
    }
    produced = {zone: zone_of(*point) for zone, point in witnesses.items()}
    assert produced == witnesses.keys() | set() or True  # keep the mapping visible on failure
    for zone, point in witnesses.items():
        assert produced[zone] == zone, f"{point} scored {produced[zone]}, expected {zone}"
    assert set(produced.values()) == set(CLARKE_ZONES)


def test_clarke_gross_error_inside_range_is_not_zone_a():
    """Reference 75, prediction 180 is a 140% error and must not be zone A.

    ``evaluate_ohio.py:322`` scored *any* pair with both values inside 70-180 as
    zone A, which is what made its 85.8% zone-A figure meaningless.
    """
    assert zone_of(75.0, 180.0) != "A"


def test_clarke_large_hyper_error_is_not_zone_a():
    """Reference 400, prediction 210 is a 47% error and must not be zone A.

    ``evaluate.py:186-207`` contained an invented ``reference >= 290`` branch that
    scored this as A.
    """
    assert zone_of(400.0, 210.0) != "A"


def test_clarke_erroneous_treatment_both_directions():
    """Predicting high when truly hypoglycaemic, and the reverse, are zone E."""
    assert zone_of(60.0, 200.0) == "E"
    assert zone_of(250.0, 60.0) == "E"


def test_clarke_failure_to_detect_is_zone_d_not_b():
    """A true excursion predicted as in-range is D.

    ``evaluate_ohio.py:328-329`` relabelled both of Clarke's zone-D legs as B,
    moving failures-to-detect into the 'clinically acceptable' aggregate.
    """
    assert zone_of(55.0, 120.0) == "D"  # missed hypoglycaemia
    assert zone_of(300.0, 150.0) == "D"  # missed hyperglycaemia


def test_clarke_both_low_is_zone_a():
    """Both values hypoglycaemic is zone A even outside the 20% band."""
    assert zone_of(50.0, 65.0) == "A"


def test_clarke_within_twenty_percent_is_zone_a():
    for reference in (80.0, 120.0, 200.0, 350.0):
        assert zone_of(reference, reference * 1.19) == "A"
        assert zone_of(reference, reference * 0.81) == "A"


def test_clarke_just_outside_twenty_percent_is_not_zone_a():
    for reference in (120.0, 200.0):
        assert zone_of(reference, reference * 1.21) != "A"


def test_clarke_exact_prediction_is_zone_a():
    for value in (40.0, 70.0, 100.0, 180.0, 400.0):
        assert zone_of(value, value) == "A"


def test_clarke_evaluation_order_is_canonical_a_first():
    """Zone A is tested first, matching both reference implementations.

    Reference 70 with prediction 84 sits exactly on the upper A-line
    (``1.2 * 70 = 84``). Under the canonical A-first order it is zone A, even
    though the reference is at the hypoglycaemia threshold with an in-range
    prediction. Reordering to claim D here would be arguably safer but would make
    every number incomparable to published OhioT1DM results, which is the entire
    point of computing them.
    """
    assert zone_of(70.0, 84.0) == "A"


def test_clarke_upper_c_has_no_reference_cap():
    """The upper-C leg must extend past 290 mg/dL.

    The MATLAB-lineage cap at ``r <= 290`` is an artefact of the original figure's
    0-400 axes: ``p = r + 110`` leaves the vertex ``(70, 180)`` and exits a
    400-limit plot at ``(290, 400)``. It is not a clinical boundary, and capping it
    silently reclassifies gross over-predictions on real CGM data as benign.
    """
    assert zone_of(350.0, 470.0) == "C"  # 470 >= 350 + 110
    assert zone_of(300.0, 415.0) == "C"


def test_clarke_lower_c_triangle_vertices():
    """Lower C is the triangle (130,0), (180,0), (180,70) via p <= 1.4r - 182.

    Note the triangle's right edge at ``r = 180`` is not reachable as C: at
    ``r >= 180`` with ``p <= 70`` the point is erroneous treatment, and E is tested
    first. So C is probed just inside the reference range.
    """
    assert zone_of(170.0, 50.0) == "C"  # 1.4*170-182 = 56
    assert zone_of(150.0, 20.0) == "C"  # 1.4*150-182 = 28
    # Just outside the triangle on the reference axis.
    assert zone_of(125.0, 10.0) != "C"
    # And the E precedence at the triangle's right edge.
    assert zone_of(180.0, 60.0) == "E"


def test_clarke_58_33_wedge_is_covered_by_zone_a():
    """The 58.33 constant is derived, not a primary boundary.

    ``58.33 = 70 / 1.2`` is where the upper A-line crosses ``p = 70``. For
    ``r`` between 58.33 and 70 the A band already reaches above 70, so those points
    are claimed by A before D is reached -- which is why writing lower-D simply as
    ``r <= 70`` is equivalent to spelling out the wedge.
    """
    # r = 65: upper A-line is at 78, so a prediction of 75 is A, not D.
    assert zone_of(65.0, 75.0) == "A"
    # r = 55: upper A-line is at 66, below 70, so a prediction of 75 is D.
    assert zone_of(55.0, 75.0) == "D"


def _reference_clarke(r: float, p: float) -> str:
    """The MATLAB (Guevara Codina) -> Python (Tsue) reference implementation.

    Reproduced verbatim as an oracle. Used to prove that this project's
    implementation differs from the lineage most CGM papers use *only* in the one
    documented place.
    """
    if (r <= 70 and p <= 70) or (0.8 * r <= p <= 1.2 * r):
        return "A"
    if (r >= 180 and p <= 70) or (r <= 70 and p >= 180):
        return "E"
    if ((70 <= r <= 290) and p >= r + 110) or ((130 <= r <= 180) and p <= (7 / 5) * r - 182):
        return "C"
    if (
        (r >= 240 and 70 <= p <= 180)
        or (r <= 175 / 3 and 70 <= p <= 180)
        or ((175 / 3 <= r <= 70) and p >= (6 / 5) * r)
    ):
        return "D"
    return "B"


def test_clarke_matches_reference_implementation_except_the_dropped_cap():
    """Lattice comparison against the canonical reference implementation.

    Every disagreement must be explainable as either (a) the intentionally dropped
    ``r <= 290`` upper-C cap, or (b) an open-versus-closed decision on a
    measure-zero boundary line. Any other disagreement is a bug in this module.
    """
    axis = np.arange(5, 551, 5, dtype=np.float64)
    reference_mesh, predicted_mesh = np.meshgrid(axis, axis, indexing="ij")
    ours = clarke_zone(reference_mesh, predicted_mesh)

    unexplained: list[tuple[float, float, str, str]] = []
    for i, r in enumerate(axis):
        for j, p in enumerate(axis):
            theirs = _reference_clarke(float(r), float(p))
            mine = str(ours[i, j])
            if mine == theirs:
                continue
            # (a) the dropped cap: we say C where they say B, beyond r = 290.
            if mine == "C" and theirs == "B" and r > 290 and p >= r + 110:
                continue
            # (b) measure-zero boundary lines.
            on_boundary = (
                abs(p - 1.2 * r) < 1e-9
                or abs(p - 0.8 * r) < 1e-9
                or abs(p - (r + 110)) < 1e-9
                or abs(p - (1.4 * r - 182)) < 1e-9
                or r in (70.0, 180.0, 240.0, 130.0)
                or p in (70.0, 180.0)
            )
            if on_boundary:
                continue
            unexplained.append((float(r), float(p), mine, theirs))

    assert not unexplained, f"{len(unexplained)} unexplained disagreements: {unexplained[:10]}"


def test_clarke_rejects_non_positive_reference():
    """Zone A is a *relative* criterion, so a zero reference is undefined."""
    with pytest.raises(MetricError, match="positive reference"):
        clarke_zone(np.array([0.0]), np.array([100.0]))


def test_clarke_shape_mismatch_rejected():
    with pytest.raises(MetricError, match="shape mismatch"):
        clarke_zone(np.array([100.0, 120.0]), np.array([100.0]))


def test_clarke_is_vectorised_and_shape_preserving():
    reference = np.array([[100.0, 60.0], [300.0, 150.0]])
    predicted = np.array([[110.0, 200.0], [120.0, 200.0]])
    zones = clarke_zone(reference, predicted)
    assert zones.shape == reference.shape
    assert zones.tolist() == [["A", "E"], ["D", "B"]]


# --------------------------------------------------------------------------- #
# Zone summary and figure/table consistency
# --------------------------------------------------------------------------- #


def test_zone_summary_reports_every_zone_including_empty():
    """Zone E must appear as an explicit zero rather than be absent.

    The legacy metrics writer emitted only A-D columns. A missing column reads as
    'no dangerous predictions' when it actually means 'never measured'.
    """
    reference = np.array([100.0, 110.0, 120.0])
    summary = zone_summary(reference, reference * 1.05)
    assert set(summary.percentages) == set(CLARKE_ZONES)
    assert summary.percentages["E"] == 0.0
    assert summary.counts["E"] == 0


def test_zone_summary_percentages_sum_to_hundred():
    generator = np.random.default_rng(0)
    reference = generator.uniform(40, 400, 500)
    predicted = np.clip(reference + generator.normal(0, 45, 500), 20, 600)
    summary = zone_summary(reference, predicted)
    assert sum(summary.percentages.values()) == pytest.approx(100.0)
    assert sum(summary.counts.values()) == summary.n


def test_zone_summary_aggregates_match_components():
    generator = np.random.default_rng(1)
    reference = generator.uniform(40, 400, 300)
    predicted = np.clip(reference + generator.normal(0, 60, 300), 20, 600)
    summary = zone_summary(reference, predicted)
    assert summary.clinically_acceptable == pytest.approx(
        summary.percentages["A"] + summary.percentages["B"]
    )
    assert summary.dangerous == pytest.approx(
        summary.percentages["D"] + summary.percentages["E"]
    )


def test_zone_field_agrees_with_classifier_pointwise():
    """The figure's shading must be generated by the classifier it illustrates.

    This is the structural fix for the legacy plots, which drew decorative
    segments unrelated to the counting logic, so a figure could contradict its own
    table indefinitely.
    """
    reference_axis, predicted_axis, indices, order = zone_field(resolution=60)
    generator = np.random.default_rng(2)
    for _ in range(200):
        i = int(generator.integers(0, len(reference_axis)))
        j = int(generator.integers(0, len(predicted_axis)))
        expected = zone_of(float(reference_axis[i]), float(predicted_axis[j]))
        assert order[indices[i, j]] == expected


def test_zone_field_covers_all_zones():
    _, _, indices, order = zone_field(resolution=200)
    present = {order[index] for index in np.unique(indices)}
    assert present == set(CLARKE_ZONES)


# --------------------------------------------------------------------------- #
# Parkes -- gated until verified against the primary source
# --------------------------------------------------------------------------- #


def parkes_of(reference: float, predicted: float) -> str:
    return str(parkes_zone(np.array([reference]), np.array([predicted]))[0])


def test_parkes_boundaries_are_verified():
    """Coordinates come from Pfuetzner 2013 Table 1, cross-checked against ``ega``."""
    from twin.metrics.errorgrid import CITATIONS, VERIFICATION_STATUS

    assert VERIFICATION_STATUS["parkes"] is True
    assert "Pfuetzner" in CITATIONS["parkes"]


def test_parkes_identity_line_is_zone_a():
    for value in (50.0, 100.0, 200.0, 350.0, 500.0):
        assert parkes_of(value, value) == "A"


def test_parkes_all_zones_reachable():
    """A-E must all be assignable on the Type 1 grid."""
    _, _, indices, order = zone_field(grid="parkes", glucose_max=550.0, resolution=250)
    present = {order[index] for index in np.unique(indices)}
    assert present == set(PARKES_ZONES)


def test_parkes_type1_has_no_lower_e_zone():
    """Type 1 defines no lower E region, so under-prediction saturates at D.

    A property of the published grid, not an omission. Worth stating explicitly
    when reporting, since a reader may expect symmetry.
    """
    from twin.metrics.errorgrid import PARKES_T1

    assert "E_lower" not in PARKES_T1
    # Extreme under-prediction at high reference: worst available is D.
    assert parkes_of(500.0, 1.0) == "D"


def test_parkes_upper_bands_increase_in_severity():
    """Moving upward at fixed reference must never decrease severity."""
    severity = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    for ref in (40.0, 80.0, 150.0, 300.0):
        zones = [parkes_of(ref, p) for p in np.arange(ref, 551.0, 10.0)]
        ranks = [severity[zone] for zone in zones]
        assert ranks == sorted(ranks), f"severity not monotone above identity at r={ref}"


def test_parkes_lower_bands_increase_in_severity():
    severity = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    for ref in (150.0, 300.0, 500.0):
        zones = [parkes_of(ref, p) for p in np.arange(ref, 0.0, -10.0)]
        ranks = [severity[zone] for zone in zones]
        assert ranks == sorted(ranks), f"severity not monotone below identity at r={ref}"


def test_parkes_known_vertex_points():
    """Points placed relative to published Table 1 vertices.

    ``B_upper`` passes through (140, 170) and ``C_upper`` through (70, 110), so
    points just inside and just outside those boundaries must differ in zone.
    """
    assert parkes_of(140.0, 172.0) == "B"  # just above B_upper
    assert parkes_of(140.0, 165.0) == "A"  # just below B_upper
    # C_upper at r=70 is 110; D_upper at r=70 is ~192 on the (50,125)-(80,215) leg.
    assert parkes_of(70.0, 115.0) == "C"
    assert parkes_of(70.0, 105.0) == "B"


def test_parkes_extrapolates_beyond_last_vertex():
    """Beyond the final vertex the terminal slope must extend, not clamp.

    ``B_upper`` ends at (430, 550). Clamping would place the boundary at 550 for
    all larger references and misclassify near-identity points as B.
    """
    assert parkes_of(500.0, 505.0) == "A"


def test_parkes_rejects_unknown_diabetes_type():
    with pytest.raises(MetricError, match="diabetes_type"):
        parkes_zone(np.array([100.0]), np.array([100.0]), diabetes_type=3)


def test_parkes_summary_reports_every_zone():
    reference = np.array([100.0, 150.0, 200.0])
    summary = zone_summary(reference, reference, grid="parkes")
    assert set(summary.percentages) == set(PARKES_ZONES)
    assert summary.percentages["A"] == pytest.approx(100.0)


def test_zone_summary_rejects_unknown_grid():
    with pytest.raises(MetricError, match="unknown grid"):
        zone_summary(np.array([100.0]), np.array([100.0]), grid="bogus")


def test_unverified_gate_still_functions():
    """The verification gate must actually block when a grid is marked unverified."""
    from twin.metrics import errorgrid

    original = errorgrid.VERIFICATION_STATUS["parkes"]
    errorgrid.VERIFICATION_STATUS["parkes"] = False
    try:
        with pytest.raises(UnverifiedBoundaryError, match="not been verified"):
            errorgrid.parkes_zone(np.array([100.0]), np.array([110.0]))
    finally:
        errorgrid.VERIFICATION_STATUS["parkes"] = original


# --------------------------------------------------------------------------- #
# Accuracy
# --------------------------------------------------------------------------- #


def test_perfect_prediction_gives_zero_error():
    y = np.array([[100.0, 120.0], [150.0, 200.0]])
    assert np.allclose(rmse(y, y), 0.0)
    assert np.allclose(mae(y, y), 0.0)
    assert np.allclose(mard(y, y), 0.0)
    assert np.allclose(r2(y, y), 1.0)


def test_metrics_are_computed_per_horizon():
    """Each horizon column must be scored independently."""
    y_true = np.array([[100.0, 100.0], [100.0, 100.0]])
    y_pred = np.array([[110.0, 130.0], [110.0, 130.0]])
    assert np.allclose(mae(y_true, y_pred), [10.0, 30.0])


def test_r2_can_be_negative():
    """A model worse than the mean must score below zero, not be clipped."""
    y_true = np.array([[100.0], [200.0], [150.0]])
    y_pred = np.array([[300.0], [50.0], [400.0]])
    assert float(r2(y_true, y_pred)[0]) < 0.0


def test_r2_is_nan_for_constant_reference():
    y_true = np.array([[100.0], [100.0], [100.0]])
    y_pred = np.array([[101.0], [99.0], [100.0]])
    assert np.isnan(r2(y_true, y_pred)[0])


def test_mard_uses_reference_as_denominator():
    """MARD must divide by the reference, not the prediction."""
    y_true = np.array([[100.0]])
    y_pred = np.array([[150.0]])
    assert float(mard(y_true, y_pred)[0]) == pytest.approx(50.0)


def test_non_finite_input_raises_rather_than_being_dropped():
    """A NaN means the sequencing layer emitted a bad window; it must surface."""
    y_true = np.array([[100.0], [np.nan]])
    y_pred = np.array([[100.0], [100.0]])
    with pytest.raises(MetricError, match="non-finite"):
        rmse(y_true, y_pred)


def test_shape_mismatch_raises():
    with pytest.raises(MetricError, match="shape mismatch"):
        rmse(np.zeros((3, 2)), np.zeros((3, 3)))


def test_accuracy_by_horizon_labels_match_columns():
    y_true = np.tile(np.array([[100.0, 120.0, 140.0, 160.0]]), (5, 1))
    y_pred = y_true + 5.0
    records = accuracy_by_horizon(y_true, y_pred, HORIZONS)
    assert [record.horizon_min for record in records] == list(HORIZONS)
    assert all(record.n == 5 for record in records)


def test_accuracy_rejects_horizon_label_mismatch():
    with pytest.raises(MetricError, match="horizon labels"):
        accuracy_by_horizon(np.zeros((4, 2)) + 100, np.zeros((4, 2)) + 100, HORIZONS)


def test_bias_sign_convention():
    from twin.metrics import bias

    y_true = np.array([[100.0]])
    assert float(bias(y_true, y_true + 10.0)[0]) == pytest.approx(10.0)
    assert float(bias(y_true, y_true - 10.0)[0]) == pytest.approx(-10.0)


def test_prediction_lag_detects_a_shifted_forecast():
    """A forecast that merely replays the input must show a lag near the horizon."""
    time = np.arange(400)
    reference = 140 + 50 * np.sin(2 * np.pi * time / 96)
    shift = 6  # 30 min at 5-min sampling
    lagged = np.roll(reference, shift)
    assert prediction_lag_min(reference, lagged, sample_minutes=5) == pytest.approx(30.0)
    assert prediction_lag_min(reference, reference, sample_minutes=5) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Clinical metrics
# --------------------------------------------------------------------------- #


def test_range_bands_are_exhaustive_and_disjoint():
    generator = np.random.default_rng(3)
    glucose = generator.uniform(30, 450, 2000)
    distribution = range_distribution(glucose)
    total = (
        distribution.very_low
        + distribution.low
        + distribution.in_range
        + distribution.high
        + distribution.very_high
    )
    assert total == pytest.approx(100.0)
    assert distribution.time_below_range == pytest.approx(
        distribution.very_low + distribution.low
    )


def test_range_band_boundaries_are_placed_correctly():
    distribution = range_distribution(np.array([53.0, 60.0, 70.0, 180.0, 200.0, 300.0]))
    assert distribution.very_low == pytest.approx(100 / 6)  # 53
    assert distribution.low == pytest.approx(100 / 6)  # 60
    assert distribution.in_range == pytest.approx(200 / 6)  # 70, 180 inclusive
    assert distribution.high == pytest.approx(100 / 6)  # 200
    assert distribution.very_high == pytest.approx(100 / 6)  # 300


def test_risk_transform_is_symmetric_over_its_design_range():
    """The transform must map 20 and 600 mg/dL to +/- sqrt(10).

    That is the property the three constants exist to produce, so it is the
    strongest available check on them short of the primary source: if any of
    1.509, 1.084, or 5.381 were misremembered, the symmetry would break.
    """
    low = float(risk_transform(np.array([20.0]))[0])
    high = float(risk_transform(np.array([600.0]))[0])
    assert low < 0 < high
    assert abs(low + high) < 0.01, f"asymmetric: f(20)={low:.4f}, f(600)={high:.4f}"
    assert abs(high - np.sqrt(10.0)) < 0.01, f"f(600)={high:.4f}, expected sqrt(10)"


def test_risk_transform_zero_near_euglycaemia():
    """``f`` must cross zero in the euglycaemic region, around 112 mg/dL."""
    values = risk_transform(np.array([100.0, 112.5, 125.0]))
    assert values[0] < 0 < values[2]
    assert abs(float(values[1])) < 0.05


def test_risk_index_upper_bound():
    """``r = 10 f^2`` must reach ~100 at the range endpoints, not exceed it inside."""
    assert hbgi(np.array([600.0])) == pytest.approx(100.0, abs=0.5)
    assert lbgi(np.array([20.0])) == pytest.approx(100.0, abs=0.5)


def test_lbgi_and_hbgi_separate_the_two_sides():
    assert lbgi(np.array([50.0, 55.0, 60.0])) > 0
    assert hbgi(np.array([50.0, 55.0, 60.0])) == pytest.approx(0.0)
    assert hbgi(np.array([300.0, 350.0])) > 0
    assert lbgi(np.array([300.0, 350.0])) == pytest.approx(0.0)


def test_risk_indices_reject_non_positive_glucose():
    with pytest.raises(MetricError, match="strictly positive"):
        lbgi(np.array([0.0, 100.0]))


def test_cv_detects_excursion_compression():
    """A flattened prediction must show a visibly lower CV than the reference."""
    generator = np.random.default_rng(4)
    reference = 150 + 60 * generator.standard_normal(500)
    reference = np.clip(reference, 40, 400)
    flattened = reference.mean() + 0.3 * (reference - reference.mean())
    assert coefficient_of_variation(flattened) < 0.5 * coefficient_of_variation(reference)


def test_hypoglycaemia_detection_counts_events():
    reference = np.array([60.0, 65.0, 100.0, 200.0])
    predicted = np.array([60.0, 100.0, 100.0, 200.0])
    result = hypoglycaemia_detection(reference, predicted)
    assert result["n_actual_events"] == 2
    assert result["sensitivity"] == pytest.approx(0.5)
    assert result["false_negative"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Statistics over subjects
# --------------------------------------------------------------------------- #


def test_bootstrap_ci_brackets_the_point_estimate():
    generator = np.random.default_rng(5)
    per_subject = generator.normal(30.0, 4.0, 12)
    interval = bootstrap_ci(per_subject, seed=0)
    assert interval.low < interval.point < interval.high
    assert interval.n_subjects == 12


def test_bootstrap_ci_is_deterministic_given_a_seed():
    values = np.linspace(20, 40, 12)
    first = bootstrap_ci(values, seed=7)
    second = bootstrap_ci(values, seed=7)
    assert (first.low, first.point, first.high) == (second.low, second.point, second.high)


def test_paired_comparison_direction_and_count():
    """A uniformly better method must show a negative difference and win everywhere."""
    better = np.array([20.0, 22.0, 19.0, 25.0, 21.0, 23.0, 20.5, 24.0])
    worse = better + 5.0
    result = paired_comparison(better, worse, label="better vs worse")
    assert result.mean_difference == pytest.approx(-5.0)
    assert result.n_favouring_first == len(better)
    assert result.p_value < 0.05
    assert result.effect_size < 0


def test_paired_comparison_flags_small_samples_as_underpowered():
    result = paired_comparison(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]))
    assert result.underpowered is True


def test_paired_comparison_handles_identical_inputs():
    values = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
    result = paired_comparison(values, values)
    assert result.p_value == 1.0
    assert result.mean_difference == 0.0


def test_paired_comparison_requires_alignment():
    with pytest.raises(ValueError, match="align"):
        paired_comparison(np.zeros(5), np.zeros(4))


def test_holm_bonferroni_is_monotone_and_conservative():
    raw = {"a": 0.001, "b": 0.01, "c": 0.04, "d": 0.5}
    corrected = holm_bonferroni(raw)
    for key, value in raw.items():
        assert corrected[key]["p_adjusted"] >= value
    ordered = sorted(corrected.values(), key=lambda item: item["rank"])
    adjusted = [item["p_adjusted"] for item in ordered]
    assert adjusted == sorted(adjusted), "adjusted p-values must be non-decreasing in rank"


def test_holm_bonferroni_rejects_least_significant_last():
    corrected = holm_bonferroni({"strong": 0.0001, "weak": 0.9})
    assert corrected["strong"]["reject"] is True
    assert corrected["weak"]["reject"] is False


def test_holm_bonferroni_empty_family():
    assert holm_bonferroni({}) == {}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _make_subject(subject_id: str, offset: float, n: int = 60) -> SubjectPredictions:
    generator = np.random.default_rng(abs(hash(subject_id)) % 2**32)
    reference = np.clip(150 + 50 * generator.standard_normal((n, len(HORIZONS))), 45, 380)
    predicted = np.clip(reference + offset + generator.standard_normal((n, len(HORIZONS))) * 5, 45, 380)
    return SubjectPredictions(subject_id=subject_id, y_true=reference, y_pred=predicted)


def test_per_subject_table_has_one_row_per_subject_and_horizon():
    subjects = [_make_subject(f"s{index}", offset=index) for index in range(4)]
    table = per_subject_table(subjects, HORIZONS)
    assert len(table) == 4 * len(HORIZONS)
    assert set(table["horizon_min"]) == set(HORIZONS)
    assert table["subject_id"].nunique() == 4


def test_across_subject_summary_reports_mean_and_sd_across_subjects():
    """The headline must be across subjects, with a sample SD."""
    subjects = [_make_subject(f"s{index}", offset=index * 3) for index in range(6)]
    table = per_subject_table(subjects, HORIZONS)
    summary = across_subject_summary(table)

    assert len(summary) == len(HORIZONS)
    assert (summary["n_subjects"] == 6).all()
    for horizon in HORIZONS:
        row = summary[summary["horizon_min"] == horizon].iloc[0]
        per_subject_values = table[table["horizon_min"] == horizon]["mae"]
        assert row["mae_mean"] == pytest.approx(per_subject_values.mean())
        assert row["mae_sd"] == pytest.approx(per_subject_values.std(ddof=1))


def test_summary_includes_bootstrap_intervals_for_headline_metrics():
    subjects = [_make_subject(f"s{index}", offset=index) for index in range(8)]
    summary = across_subject_summary(per_subject_table(subjects, HORIZONS))
    for column in ("rmse_ci_low", "rmse_ci_high", "mae_ci_low", "mae_ci_high"):
        assert column in summary.columns
    assert (summary["mae_ci_low"] <= summary["mae_mean"]).all()
    assert (summary["mae_mean"] <= summary["mae_ci_high"]).all()


def test_pooled_differs_from_per_subject_under_imbalance():
    """Window-count imbalance must make pooling and averaging diverge.

    This is the defect the legacy pipeline had: it pooled with ``np.concatenate``,
    so a subject with many windows dominated the reported number. The test asserts
    the two aggregations genuinely differ, which is why only one of them can be
    the headline.
    """
    small = _make_subject("small", offset=1.0, n=20)
    large = _make_subject("large", offset=25.0, n=2000)
    subjects = [small, large]

    per_subject = across_subject_summary(per_subject_table(subjects, HORIZONS))
    pooled = pooled_metrics(subjects, HORIZONS)

    per_subject_mae = float(per_subject.iloc[0]["mae_mean"])
    pooled_mae = float(pooled.iloc[0]["mae"])
    assert abs(pooled_mae - per_subject_mae) > 3.0, (
        f"pooled {pooled_mae:.2f} vs per-subject {per_subject_mae:.2f}: "
        "imbalance should make these diverge"
    )


def test_format_mean_sd_produces_table_ready_strings():
    subjects = [_make_subject(f"s{index}", offset=index) for index in range(5)]
    summary = across_subject_summary(per_subject_table(subjects, HORIZONS))
    formatted = format_mean_sd(summary, "mae")
    assert len(formatted) == len(HORIZONS)
    assert all("±" in value for value in formatted)


def test_format_mean_sd_rejects_unaggregated_metric():
    summary = pd.DataFrame({"horizon_min": [30]})
    with pytest.raises(MetricError, match="not in summary"):
        format_mean_sd(summary, "mae")


def test_subject_predictions_validate_shape_contract():
    subject = SubjectPredictions(
        subject_id="x", y_true=np.ones((10, 3)) * 100, y_pred=np.ones((10, 3)) * 100
    )
    with pytest.raises(MetricError, match="expected"):
        subject.validate(n_horizons=4)


def test_subject_predictions_reject_empty():
    subject = SubjectPredictions(
        subject_id="x", y_true=np.ones((0, 4)), y_pred=np.ones((0, 4))
    )
    with pytest.raises(MetricError, match="no windows"):
        subject.validate(n_horizons=4)


def test_report_surfaces_excursion_compression():
    """A deliberately flattened predictor must show ``cv_ratio`` well below 1."""
    generator = np.random.default_rng(11)
    reference = np.clip(150 + 60 * generator.standard_normal((300, len(HORIZONS))), 45, 380)
    flattened = reference.mean(axis=0) + 0.25 * (reference - reference.mean(axis=0))
    subject = SubjectPredictions(subject_id="flat", y_true=reference, y_pred=flattened)

    table = per_subject_table([subject], HORIZONS)
    assert (table["cv_ratio"] < 0.5).all()
