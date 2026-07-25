"""Verification of gap-aware window construction.

The first three tests are the most important in the suite. They guard the defect
that invalidated every Ohio number in the legacy pipeline: windows whose nominal
forecast horizon was not the actual elapsed time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from twin.data.ohio import GriddedSubject, discover_files, load_subject
from twin.data.sequencing import (
    FILLED_COLUMN,
    INTERPOLATED_COLUMN,
    SequencingError,
    build_windows,
    interpolate_bounded,
    persistence_targets,
    window_report_table,
)

OHIO_ROOT = Path("OhioT1DM")
needs_ohio = pytest.mark.skipif(
    not OHIO_ROOT.is_dir(), reason="OhioT1DM data directory not present"
)

HORIZONS = (30, 60, 90, 120)
SEQ_LEN = 24


def make_subject(
    glucose: list[float | None],
    *,
    start: str = "2021-12-07 00:00:00",
    grid_minutes: int = 5,
    split: str = "test",
) -> GriddedSubject:
    """A gridded subject with an explicit glucose pattern; ``None`` means a gap."""
    index = pd.date_range(pd.Timestamp(start), periods=len(glucose), freq=f"{grid_minutes}min")
    index.name = "timestamp"
    values = np.array([np.nan if value is None else float(value) for value in glucose])
    frame = pd.DataFrame(index=index)
    frame["glucose_mg_dl"] = values
    frame["glucose_observed"] = ~np.isnan(values)
    for column in ("bolus_u_per_min", "basal_u_per_min", "carbs_mg_per_min"):
        frame[column] = 0.0
    for column in ("exercise_intensity", "sleeping", "working"):
        frame[column] = 0.0
    return GriddedSubject(
        subject_id="synthetic",
        cohort="2018",
        split=split,
        grid_minutes=grid_minutes,
        frame=frame,
        body_weight_kg=70.0,
        available_sensors=(),
        excluded_warmup_samples=0,
        source_path=Path("synthetic.xml"),
    )


def clean_series(n: int) -> list[float]:
    """A smoothly varying, fully observed glucose series."""
    return [120.0 + 40.0 * np.sin(2 * np.pi * i / 96) for i in range(n)]


# --------------------------------------------------------------------------- #
# Horizon integrity -- the central guarantee
# --------------------------------------------------------------------------- #


def test_targets_are_at_the_true_horizon():
    """Elapsed wall-clock time to each target must equal the nominal horizon.

    The legacy pipeline dropped NaN rows and called ``reset_index(drop=True)``
    before slicing, so a window spanning a gap had targets at the wrong times.
    ``WindowSet.verify`` re-derives this from the frame; here it is asserted
    directly.
    """
    subject = make_subject(clean_series(200))
    windows = build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS)
    assert len(windows) > 0

    times = subject.frame.index
    for column, horizon in enumerate(HORIZONS):
        step = horizon // subject.grid_minutes
        elapsed = times[windows.anchors + step] - times[windows.anchors]
        assert (elapsed == pd.Timedelta(minutes=horizon)).all()


def test_no_window_spans_a_gap_with_a_shifted_horizon():
    """A gap must remove windows, never silently shorten their horizon."""
    series: list[float | None] = clean_series(300)
    # A three-hour outage in the middle.
    for index in range(120, 156):
        series[index] = None

    subject = make_subject(series)
    windows = build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS)
    times = subject.frame.index

    for column, horizon in enumerate(HORIZONS):
        step = horizon // subject.grid_minutes
        elapsed = times[windows.anchors + step] - times[windows.anchors]
        assert (elapsed == pd.Timedelta(minutes=horizon)).all()

    observed = subject.frame["glucose_observed"].to_numpy(dtype=bool)
    for step in windows.horizon_steps:
        assert observed[windows.anchors + step].all()


def test_targets_are_never_interpolated():
    """Every target must be a sensor reading, not a filled value.

    The legacy code forward-filled gaps up to 15 minutes and used the filled values
    as targets, so a "30-minute prediction" could be scored against a copy of a
    reading the model had already seen -- which persistence predicts perfectly.
    """
    series: list[float | None] = clean_series(200)
    series[100] = None  # a single-slot gap, short enough to interpolate
    subject = make_subject(series)

    frame = interpolate_bounded(subject.frame, max_interp_gap=2)
    assert bool(frame[INTERPOLATED_COLUMN].iloc[100]), "fixture gap was not interpolated"

    windows = build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS, max_interp_gap=2)
    interpolated = frame[INTERPOLATED_COLUMN].to_numpy(dtype=bool)
    for step in windows.horizon_steps:
        assert not interpolated[windows.anchors + step].any()


def test_stored_targets_match_the_frame():
    subject = make_subject(clean_series(200))
    windows = build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS)
    glucose = subject.frame["glucose_mg_dl"].to_numpy(dtype=np.float64)
    for column, step in enumerate(windows.horizon_steps):
        assert np.allclose(windows.targets[:, column], glucose[windows.anchors + step])


def test_verify_rejects_tampered_targets():
    """The runtime guarantee must actually fail when violated."""
    subject = make_subject(clean_series(200))
    windows = build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS)
    windows.targets[0, 0] += 25.0
    with pytest.raises(SequencingError, match="disagree with the frame"):
        windows.verify(interpolate_bounded(subject.frame))


# --------------------------------------------------------------------------- #
# Interpolation policy
# --------------------------------------------------------------------------- #


def test_long_gaps_are_never_bridged():
    subject = make_subject(clean_series(120) + [None] * 10 + clean_series(120))
    frame = interpolate_bounded(subject.frame, max_interp_gap=2)
    filled = frame[FILLED_COLUMN].to_numpy()
    assert np.isnan(filled[120:130]).all(), "a 50-minute gap must not be interpolated"


def test_short_gaps_are_interpolated_linearly():
    series: list[float | None] = [100.0, None, None, 130.0]
    subject = make_subject(series)
    frame = interpolate_bounded(subject.frame, max_interp_gap=2)
    filled = frame[FILLED_COLUMN].to_numpy()
    assert filled[1] == pytest.approx(110.0)
    assert filled[2] == pytest.approx(120.0)


def test_interpolation_never_extrapolates_past_the_record():
    subject = make_subject([None, None, 100.0, 110.0, None, None])
    frame = interpolate_bounded(subject.frame, max_interp_gap=3)
    filled = frame[FILLED_COLUMN].to_numpy()
    assert np.isnan(filled[0]) and np.isnan(filled[1]), "must not backfill before the record"
    assert np.isnan(filled[4]) and np.isnan(filled[5]), "must not extrapolate past the record"


def test_zero_interp_gap_disables_filling():
    subject = make_subject([100.0, None, 120.0])
    frame = interpolate_bounded(subject.frame, max_interp_gap=0)
    assert np.isnan(frame[FILLED_COLUMN].to_numpy()[1])
    assert not frame[INTERPOLATED_COLUMN].any()


def test_observed_column_is_never_modified_by_interpolation():
    """Measured and inferred must stay distinguishable."""
    subject = make_subject([100.0, None, 120.0])
    frame = interpolate_bounded(subject.frame, max_interp_gap=2)
    assert list(frame["glucose_observed"]) == [True, False, True]


def test_negative_interp_gap_rejected():
    subject = make_subject(clean_series(50))
    with pytest.raises(SequencingError, match="non-negative"):
        interpolate_bounded(subject.frame, max_interp_gap=-1)


def test_coverage_threshold_rejects_heavily_interpolated_inputs():
    """A window mostly built from interpolated values must be rejected."""
    series: list[float | None] = clean_series(200)
    # Alternate gaps across one window's input span; each is short enough to fill.
    for index in range(40, 64, 2):
        series[index] = None
    subject = make_subject(series)

    permissive = build_windows(
        subject, seq_len=SEQ_LEN, horizons_min=HORIZONS, min_input_coverage=0.5, max_interp_gap=2
    )
    strict = build_windows(
        subject, seq_len=SEQ_LEN, horizons_min=HORIZONS, min_input_coverage=1.0, max_interp_gap=2
    )
    assert len(strict) < len(permissive)
    assert strict.report.rejected_low_coverage > 0


def test_interpolated_fraction_is_recorded_per_window():
    series: list[float | None] = clean_series(200)
    series[50] = None
    subject = make_subject(series)
    windows = build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS, min_input_coverage=0.5)
    assert windows.interpolated_fraction.shape == (len(windows),)
    assert windows.interpolated_fraction.max() > 0.0
    assert windows.interpolated_fraction.min() == 0.0


# --------------------------------------------------------------------------- #
# Window bookkeeping
# --------------------------------------------------------------------------- #


def test_anchor_is_the_last_input_slot():
    """Input span is ``[anchor - seq_len + 1, anchor]``, inclusive."""
    subject = make_subject(clean_series(200))
    windows = build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS)
    assert windows.anchors.min() == SEQ_LEN - 1
    assert np.all(windows.input_starts == windows.anchors - SEQ_LEN + 1)
    assert windows.input_starts.min() == 0


def test_last_anchor_leaves_room_for_the_longest_horizon():
    n = 200
    subject = make_subject(clean_series(n))
    windows = build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS)
    max_step = max(windows.horizon_steps)
    assert windows.anchors.max() + max_step <= n - 1


def test_expected_window_count_on_a_clean_series():
    """With no gaps, every candidate anchor must survive."""
    n = 200
    subject = make_subject(clean_series(n))
    windows = build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS)
    max_step = max(horizon // 5 for horizon in HORIZONS)
    expected = n - max_step - (SEQ_LEN - 1)
    assert len(windows) == expected
    assert windows.report.keep_rate == 1.0


def test_series_too_short_yields_no_windows():
    subject = make_subject(clean_series(10))
    windows = build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS)
    assert len(windows) == 0
    assert windows.targets.shape == (0, len(HORIZONS))
    assert windows.report.n_candidates == 0


def test_horizons_must_divide_the_grid():
    subject = make_subject(clean_series(200))
    with pytest.raises(SequencingError, match="multiples of"):
        build_windows(subject, seq_len=SEQ_LEN, horizons_min=(30, 47))


def test_report_accounts_for_every_candidate():
    """Kept plus rejected must equal candidates: nothing may vanish unexplained."""
    series: list[float | None] = clean_series(400)
    for index in range(150, 175):
        series[index] = None
    series[300] = None
    subject = make_subject(series)
    windows = build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS)
    report = windows.report

    total = (
        report.kept
        + report.rejected_input_missing
        + report.rejected_low_coverage
        + report.rejected_target_missing
    )
    assert total == report.n_candidates


def test_report_table_has_one_row_per_subject():
    subjects = [make_subject(clean_series(200)) for _ in range(3)]
    sets = [build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS) for subject in subjects]
    table = window_report_table(sets)
    assert len(table) == 3
    for column in ("kept", "n_candidates", "keep_rate", "rejected_target_missing"):
        assert column in table.columns


# --------------------------------------------------------------------------- #
# Persistence baseline
# --------------------------------------------------------------------------- #


def test_persistence_repeats_the_anchor_value():
    subject = make_subject(clean_series(200))
    windows = build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS)
    frame = interpolate_bounded(subject.frame)
    baseline = persistence_targets(subject, windows, frame=frame)

    assert baseline.shape == windows.targets.shape
    anchor_glucose = frame[FILLED_COLUMN].to_numpy()[windows.anchors]
    for column in range(len(HORIZONS)):
        assert np.allclose(baseline[:, column], anchor_glucose)


def test_persistence_error_grows_with_horizon():
    """A sanity property: the naive forecast must degrade as the horizon lengthens."""
    from twin.metrics import rmse

    subject = make_subject(clean_series(400))
    windows = build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS)
    baseline = persistence_targets(subject, windows)
    errors = rmse(windows.targets, baseline)
    assert list(errors) == sorted(errors)


# --------------------------------------------------------------------------- #
# Real corpus
# --------------------------------------------------------------------------- #


@needs_ohio
def test_real_subject_horizon_integrity():
    """The guarantee must hold on real data with real gaps."""
    subject = load_subject(OHIO_ROOT / "2020" / "test" / "552-ws-testing.xml")
    windows = build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS)
    assert len(windows) > 0

    times = subject.frame.index
    for horizon in HORIZONS:
        step = horizon // subject.grid_minutes
        elapsed = times[windows.anchors + step] - times[windows.anchors]
        assert (elapsed == pd.Timedelta(minutes=horizon)).all()


@needs_ohio
def test_real_subject_loses_windows_to_gaps():
    """552 has ~60% coverage, so a large fraction of candidates must be rejected.

    If this subject ever kept nearly all its candidates, gaps are not being
    respected and the horizon guarantee is vacuous.
    """
    subject = load_subject(OHIO_ROOT / "2020" / "test" / "552-ws-testing.xml")
    windows = build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS)
    assert windows.report.keep_rate < 0.6
    assert windows.report.rejected_input_missing > 0


@needs_ohio
def test_persistence_reproduces_published_rmse_on_2018_cohort():
    """End-to-end validation against two independent published values.

    Published 2018-challenge persistence: RMSE 22.5 +/- 2.2 at 30 min and
    36.6 +/- 3.0 at 60 min. Reproducing those simultaneously validates parsing,
    grid snapping, sequencing, horizon integrity, the metrics implementation, and
    per-subject aggregation. It is the strongest single check in this project, so a
    regression here means something upstream has broken.
    """
    from twin.metrics import rmse

    per_subject = []
    for path in discover_files(OHIO_ROOT, split="test"):
        subject = load_subject(path)
        if subject.cohort != "2018":
            continue
        windows = build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS)
        frame = interpolate_bounded(subject.frame)
        baseline = persistence_targets(subject, windows, frame=frame)
        per_subject.append(rmse(windows.targets, baseline))

    assert len(per_subject) == 6
    values = np.array(per_subject)
    rmse_30 = values[:, 0].mean()
    rmse_60 = values[:, 1].mean()

    assert rmse_30 == pytest.approx(22.5, abs=1.0), f"30-min persistence RMSE {rmse_30:.2f}"
    assert rmse_60 == pytest.approx(36.6, abs=1.5), f"60-min persistence RMSE {rmse_60:.2f}"
