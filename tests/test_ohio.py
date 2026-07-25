"""Verification of OhioT1DM parsing and gridding.

Synthetic XML fixtures cover the behaviour precisely; the tests marked
``needs_ohio`` additionally check the real corpus. Each test below names the
legacy defect it guards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from twin.data.ohio import (
    COHORT_SUBJECTS,
    NOMINAL_WEIGHT_KG,
    PLACEHOLDER_WEIGHT_KG,
    TEST_WARMUP_EXCLUSION,
    OhioParseError,
    discover_files,
    load_subject,
    parse_ohio_xml,
    to_grid,
)

OHIO_ROOT = Path("OhioT1DM")
needs_ohio = pytest.mark.skipif(
    not OHIO_ROOT.is_dir(), reason="OhioT1DM data directory not present"
)


# --------------------------------------------------------------------------- #
# Synthetic fixtures
# --------------------------------------------------------------------------- #


def _write_xml(
    tmp_path: Path,
    *,
    subject: str = "559",
    cohort: str = "2018",
    split: str = "train",
    glucose: str = "",
    basal: str = "",
    temp_basal: str = "",
    bolus: str = "",
    meal: str = "",
    stressors: str = "",
    extra: str = "",
) -> Path:
    filename = f"{subject}-ws-{'training' if split == 'train' else 'testing'}.xml"
    directory = tmp_path / cohort / split
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(
        f"""<patient id="{subject}" weight="99" insulin_type="Novalog">
<glucose_level>{glucose}</glucose_level>
<finger_stick></finger_stick>
<basal>{basal}</basal>
<temp_basal>{temp_basal}</temp_basal>
<bolus>{bolus}</bolus>
<meal>{meal}</meal>
<sleep></sleep>
<work></work>
<stressors>{stressors}</stressors>
<hypo_event></hypo_event>
<illness></illness>
<exercise></exercise>
{extra}
</patient>
"""
    )
    return path


def _glucose_events(start: str, count: int, step_minutes: int = 5, value: float = 120.0) -> str:
    stamps = pd.date_range(pd.Timestamp(start), periods=count, freq=f"{step_minutes}min")
    return "\n".join(
        f'<event ts="{stamp.strftime("%d-%m-%Y %H:%M:%S")}" value="{value}"/>' for stamp in stamps
    )


# --------------------------------------------------------------------------- #
# Structure and provenance
# --------------------------------------------------------------------------- #


def test_cohort_and_split_inferred_from_path(tmp_path):
    path = _write_xml(tmp_path, glucose=_glucose_events("2021-12-07 01:15:00", 10))
    subject = parse_ohio_xml(path)
    assert subject.subject_id == "559"
    assert subject.cohort == "2018"
    assert subject.split == "train"


def test_subject_in_wrong_cohort_directory_is_rejected(tmp_path):
    """A 2018 subject filed under 2020 must fail rather than be mislabelled."""
    path = _write_xml(
        tmp_path, subject="559", cohort="2020", glucose=_glucose_events("2021-12-07 01:15:00", 5)
    )
    with pytest.raises(OhioParseError, match="not a 2020 cohort member"):
        parse_ohio_xml(path)


def test_cohort_membership_lists_are_disjoint():
    assert not set(COHORT_SUBJECTS["2018"]) & set(COHORT_SUBJECTS["2020"])


def test_body_weight_is_nominal_not_the_placeholder(tmp_path):
    """Every file reports weight=99, a de-identification placeholder.

    Using it would be fabricated precision. Body weight is not identifiable from
    this dataset, so a documented nominal value is used and the unknown true weight
    is absorbed into the per-kilogram distribution volumes.
    """
    path = _write_xml(tmp_path, glucose=_glucose_events("2021-12-07 01:15:00", 20))
    gridded = load_subject(path)
    assert gridded.body_weight_kg == NOMINAL_WEIGHT_KG
    assert gridded.body_weight_kg != PLACEHOLDER_WEIGHT_KG


def test_missing_glucose_channel_is_an_error(tmp_path):
    path = _write_xml(tmp_path, glucose="")
    with pytest.raises(OhioParseError, match="no glucose observations"):
        load_subject(path)


# --------------------------------------------------------------------------- #
# temp_basal -- never read by the legacy parser
# --------------------------------------------------------------------------- #


def test_temp_basal_overrides_scheduled_basal(tmp_path):
    """``<temp_basal>`` must override the scheduled rate for its duration.

    The legacy parser never read this channel, so ``basal_u_h`` was wrong for every
    period a temporary rate was active -- precisely the periods around exercise and
    hypoglycaemia where the insulin signal matters most.
    """
    path = _write_xml(
        tmp_path,
        glucose=_glucose_events("2021-12-07 00:00:00", 25),
        basal='<event ts="07-12-2021 00:00:00" value="1.2"/>',
        temp_basal='<event ts_begin="07-12-2021 00:30:00" ts_end="07-12-2021 01:00:00" value="0.6"/>',
    )
    frame = load_subject(path).frame

    scheduled = frame.loc["2021-12-07 00:00:00":"2021-12-07 00:25:00", "basal_u_per_min"]
    overridden = frame.loc["2021-12-07 00:30:00":"2021-12-07 00:55:00", "basal_u_per_min"]
    restored = frame.loc["2021-12-07 01:00:00":, "basal_u_per_min"]

    assert np.allclose(scheduled, 1.2 / 60.0)
    assert np.allclose(overridden, 0.6 / 60.0)
    assert np.allclose(restored, 1.2 / 60.0)


def test_empty_temp_basal_value_means_suspension(tmp_path):
    """An empty ``value`` is a pump suspension, i.e. zero -- not 'leave unchanged'."""
    path = _write_xml(
        tmp_path,
        glucose=_glucose_events("2021-12-07 00:00:00", 25),
        basal='<event ts="07-12-2021 00:00:00" value="1.2"/>',
        temp_basal='<event ts_begin="07-12-2021 00:30:00" ts_end="07-12-2021 01:00:00" value=""/>',
    )
    frame = load_subject(path).frame
    suspended = frame.loc["2021-12-07 00:30:00":"2021-12-07 00:55:00", "basal_u_per_min"]
    assert np.allclose(suspended, 0.0)


def test_basal_step_function_holds_until_next_event(tmp_path):
    path = _write_xml(
        tmp_path,
        glucose=_glucose_events("2021-12-07 00:00:00", 25),
        basal=(
            '<event ts="07-12-2021 00:00:00" value="1.0"/>'
            '<event ts="07-12-2021 01:00:00" value="2.0"/>'
        ),
    )
    frame = load_subject(path).frame
    assert np.allclose(frame.loc["2021-12-07 00:30:00", "basal_u_per_min"], 1.0 / 60.0)
    assert np.allclose(frame.loc["2021-12-07 01:30:00", "basal_u_per_min"], 2.0 / 60.0)


# --------------------------------------------------------------------------- #
# Extended boluses -- collapsed to an instant by the legacy parser
# --------------------------------------------------------------------------- #


def test_extended_bolus_is_spread_over_its_interval(tmp_path):
    """A square-wave bolus must be distributed, not delivered instantaneously.

    The legacy parser read only ``ts_begin``, so a dose delivered over 30-120
    minutes appeared as a spike -- which then propagated into a wrong IOB.
    """
    path = _write_xml(
        tmp_path,
        glucose=_glucose_events("2021-12-07 00:00:00", 25),
        bolus=(
            '<event ts_begin="07-12-2021 00:30:00" ts_end="07-12-2021 01:00:00" '
            'type="square" dose="6.0" bwz_carb_input="60"/>'
        ),
    )
    frame = load_subject(path).frame
    rate = frame["bolus_u_per_min"]

    active = rate[rate > 0]
    assert len(active) > 1, "extended bolus collapsed to a single slot"
    assert active.index.min() == pd.Timestamp("2021-12-07 00:30:00")
    assert active.index.max() == pd.Timestamp("2021-12-07 01:00:00")
    # Mass conservation: sum(rate * grid_minutes) recovers the dose.
    assert float(rate.sum() * 5) == pytest.approx(6.0)


def test_normal_bolus_lands_in_one_slot(tmp_path):
    path = _write_xml(
        tmp_path,
        glucose=_glucose_events("2021-12-07 00:00:00", 25),
        bolus=(
            '<event ts_begin="07-12-2021 00:30:00" ts_end="07-12-2021 00:30:00" '
            'type="normal" dose="4.5" bwz_carb_input="45"/>'
        ),
    )
    rate = load_subject(path).frame["bolus_u_per_min"]
    assert int((rate > 0).sum()) == 1
    assert float(rate.sum() * 5) == pytest.approx(4.5)


def test_carbs_are_mass_conserving_in_mg(tmp_path):
    path = _write_xml(
        tmp_path,
        glucose=_glucose_events("2021-12-07 00:00:00", 25),
        meal='<event ts="07-12-2021 00:30:00" type="dinner" carbs="55"/>',
    )
    carbs = load_subject(path).frame["carbs_mg_per_min"]
    assert float(carbs.sum() * 5) == pytest.approx(55_000.0)


# --------------------------------------------------------------------------- #
# Gaps preserved -- the fatal legacy defect
# --------------------------------------------------------------------------- #


def test_gaps_are_preserved_as_nan_not_compacted(tmp_path):
    """A CGM gap must appear as NaN on the grid, never be removed.

    ``evaluate_ohio.py:254-256`` dropped NaN rows then called
    ``reset_index(drop=True)`` and built windows over the compacted array, so
    windows straddled multi-hour gaps and the nominal horizons were not the actual
    ones. Preserving the grid is what makes that impossible.
    """
    first = _glucose_events("2021-12-07 00:00:00", 12)
    # Resume three hours later.
    second = _glucose_events("2021-12-07 03:00:00", 12)
    path = _write_xml(tmp_path, glucose=first + "\n" + second)
    frame = load_subject(path).frame

    # 00:00 to 03:55 inclusive at 5 min = 48 slots.
    assert len(frame) == 48
    assert int(frame["glucose_observed"].sum()) == 24
    gap = frame.loc["2021-12-07 01:00:00":"2021-12-07 02:55:00"]
    assert gap["glucose_mg_dl"].isna().all()
    assert not gap["glucose_observed"].any()


def test_grid_is_exactly_uniform(tmp_path):
    path = _write_xml(tmp_path, glucose=_glucose_events("2021-12-07 00:02:00", 30))
    frame = load_subject(path).frame
    deltas = np.diff(frame.index.view("int64")) / 60e9
    assert np.allclose(deltas, 5.0), "grid spacing must be exactly the sampling interval"


def test_off_grid_readings_snap_to_nearest_slot(tmp_path):
    """CGM timestamps are not exactly on the clock and must be snapped, not floored."""
    path = _write_xml(
        tmp_path,
        glucose=(
            '<event ts="07-12-2021 00:01:00" value="100"/>'
            '<event ts="07-12-2021 00:06:00" value="110"/>'
            '<event ts="07-12-2021 00:14:00" value="120"/>'
        ),
    )
    frame = load_subject(path).frame
    observed = frame.loc[frame["glucose_observed"], "glucose_mg_dl"]
    assert list(observed.values) == [100.0, 110.0, 120.0]
    assert [stamp.minute for stamp in observed.index] == [0, 5, 15]


# --------------------------------------------------------------------------- #
# Channel naming and cohort sensor differences
# --------------------------------------------------------------------------- #


def test_stressors_channel_is_read(tmp_path):
    """The tag is ``stressors``, not ``stress``; the legacy code read nothing."""
    path = _write_xml(
        tmp_path,
        glucose=_glucose_events("2021-12-07 00:00:00", 12),
        stressors='<event ts_begin="07-12-2021 00:10:00" ts_end="07-12-2021 00:30:00"/>',
    )
    subject = parse_ohio_xml(path)
    assert len(subject.stressors) == 1


def test_absent_sensor_gets_availability_mask_not_silent_zero(tmp_path):
    """An unavailable channel must be flagged, not filled with a bare 0.0.

    Zero-filling before scaling maps an absent channel to a large negative z-score,
    which a model can exploit as a cohort indicator. The mask makes absence
    explicit and learnable as absence.
    """
    path = _write_xml(tmp_path, glucose=_glucose_events("2021-12-07 00:00:00", 12))
    frame = load_subject(path).frame
    assert "basis_heart_rate_available" in frame.columns
    assert (frame["basis_heart_rate_available"] == 0.0).all()


def test_all_zero_sensor_channel_counts_as_unavailable(tmp_path):
    """2020 files carry the Basis channels but fill them with zeros.

    Presence of the tag is not evidence of data, so a constant-zero channel must be
    reported unavailable.
    """
    zeros = "\n".join(
        f'<event ts="07-12-2021 00:{minute:02d}:00" value="0"/>' for minute in range(0, 55, 5)
    )
    path = _write_xml(
        tmp_path,
        glucose=_glucose_events("2021-12-07 00:00:00", 12),
        extra=f"<basis_heart_rate>{zeros}</basis_heart_rate>",
    )
    subject = parse_ohio_xml(path)
    assert "basis_heart_rate" not in subject.available_sensors
    frame = load_subject(path).frame
    assert (frame["basis_heart_rate_available"] == 0.0).all()


def test_populated_sensor_channel_counts_as_available(tmp_path):
    values = "\n".join(
        f'<event ts="07-12-2021 00:{minute:02d}:00" value="{60 + minute}"/>'
        for minute in range(0, 55, 5)
    )
    path = _write_xml(
        tmp_path,
        glucose=_glucose_events("2021-12-07 00:00:00", 12),
        extra=f"<basis_heart_rate>{values}</basis_heart_rate>",
    )
    subject = parse_ohio_xml(path)
    assert "basis_heart_rate" in subject.available_sensors
    frame = load_subject(path).frame
    assert (frame["basis_heart_rate_available"] == 1.0).all()


# --------------------------------------------------------------------------- #
# Per-cohort test protocol
# --------------------------------------------------------------------------- #


def test_2020_test_file_excludes_first_hour(tmp_path):
    """The 2020 BGLP challenge discards the first 12 test samples."""
    path = _write_xml(
        tmp_path,
        subject="540",
        cohort="2020",
        split="test",
        glucose=_glucose_events("2027-07-04 00:00:00", 60),
    )
    gridded = load_subject(path)
    assert gridded.excluded_warmup_samples == 12
    assert gridded.frame.index.min() == pd.Timestamp("2027-07-04 01:00:00")


def test_2018_test_file_excludes_nothing(tmp_path):
    """The 2018 edition uses the whole test file; applying 2020's rule would be wrong."""
    path = _write_xml(
        tmp_path,
        subject="559",
        cohort="2018",
        split="test",
        glucose=_glucose_events("2022-01-18 00:00:00", 60),
    )
    gridded = load_subject(path)
    assert gridded.excluded_warmup_samples == 0
    assert gridded.frame.index.min() == pd.Timestamp("2022-01-18 00:00:00")


def test_train_files_are_never_truncated(tmp_path):
    path = _write_xml(
        tmp_path,
        subject="540",
        cohort="2020",
        split="train",
        glucose=_glucose_events("2027-05-01 00:00:00", 60),
    )
    assert load_subject(path).excluded_warmup_samples == 0


def test_protocol_exclusion_can_be_disabled_for_diagnostics(tmp_path):
    path = _write_xml(
        tmp_path,
        subject="540",
        cohort="2020",
        split="test",
        glucose=_glucose_events("2027-07-04 00:00:00", 60),
    )
    gridded = load_subject(path, apply_protocol_exclusion=False)
    assert gridded.excluded_warmup_samples == 0
    assert gridded.frame.index.min() == pd.Timestamp("2027-07-04 00:00:00")


def test_exclusion_table_matches_documented_protocol():
    assert TEST_WARMUP_EXCLUSION == {"2018": 0, "2020": 12}


# --------------------------------------------------------------------------- #
# Real corpus
# --------------------------------------------------------------------------- #


@needs_ohio
def test_real_corpus_has_twentyfour_files():
    assert len(discover_files(OHIO_ROOT)) == 24
    assert len(discover_files(OHIO_ROOT, split="train")) == 12
    assert len(discover_files(OHIO_ROOT, split="test")) == 12


@needs_ohio
def test_real_subject_matches_raw_event_count():
    """Observed slots must match the raw XML reading count, up to snap collisions."""
    path = OHIO_ROOT / "2018" / "train" / "559-ws-training.xml"
    subject = parse_ohio_xml(path)
    gridded = load_subject(path)
    assert gridded.n_observed <= subject.glucose.size
    assert gridded.n_observed == subject.glucose.size


@needs_ohio
@pytest.mark.slow
def test_real_corpus_gaps_are_substantial():
    """Coverage is well below 100%, which is why gap-aware sequencing is required.

    If this ever reports ~100% coverage, the grid is being built wrongly and the
    horizon-integrity guarantee is vacuous.
    """
    coverages = [load_subject(path).coverage for path in discover_files(OHIO_ROOT)]
    assert min(coverages) < 0.70, "expected at least one subject with heavy gaps"
    assert max(coverages) < 0.99, "no subject should have near-complete coverage"


@needs_ohio
def test_subject_567_test_period_has_no_meal_records():
    """A documented data limitation that must be disclosed, not silently absorbed.

    Subject 567's test period contains no carbohydrate records at all, so COB is
    identically zero there and a physics-informed model is structurally degraded on
    that subject.
    """
    gridded = load_subject(OHIO_ROOT / "2020" / "test" / "567-ws-testing.xml")
    assert float(gridded.frame["carbs_mg_per_min"].sum()) == 0.0


@needs_ohio
def test_real_temp_basal_changes_the_basal_signal():
    """On real data, reading temp_basal must actually alter the basal series."""
    path = OHIO_ROOT / "2018" / "train" / "559-ws-training.xml"
    subject = parse_ohio_xml(path)
    assert len(subject.temp_basal) > 0, "fixture subject should have temp basals"

    with_override = to_grid(subject)["basal_u_per_min"]
    stripped = parse_ohio_xml(path)
    stripped.temp_basal = stripped.temp_basal.iloc[0:0]
    without_override = to_grid(stripped)["basal_u_per_min"]

    assert not np.allclose(with_override, without_override), (
        "temp_basal had no effect; the channel is being ignored"
    )
