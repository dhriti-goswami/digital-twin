"""Verification of the feature contract and corpus assembly.

``test_feature_names_unique_and_ordered`` and ``test_scaler_single_source`` are the
named guards from the rebuild plan; the rest cover the subtler contamination paths
that the legacy feature builder had.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from twin.config import Config
from twin.data.dataset import (
    FittedScaler,
    build_dataset,
    build_loader,
    collect_predictions,
    fit_scaler,
    load_subject_data,
)
from twin.data.features import (
    FEATURE_NAMES,
    GLUCOSE_DERIVED,
    N_FEATURES,
    FeatureError,
    FeatureMatrix,
    build_features,
    feature_provenance,
)
from twin.data.ohio import load_subject
from twin.data.sequencing import build_windows, interpolate_bounded
from twin.data.splits import official_split, verify_no_leakage

OHIO_ROOT = Path("OhioT1DM")
needs_ohio = pytest.mark.skipif(
    not OHIO_ROOT.is_dir(), reason="OhioT1DM data directory not present"
)


def config(**overrides) -> Config:
    base = {"data": {"root": str(OHIO_ROOT)}, "train": {"num_workers": 0, "batch_size": 8}}
    base.update(overrides)
    return Config.from_dict(base)


# --------------------------------------------------------------------------- #
# The contract itself
# --------------------------------------------------------------------------- #


def test_feature_names_unique_and_ordered():
    """No duplicates, and the count matches.

    The legacy list contained ``time_frac_day``, numerically identical to
    ``day_frac``, so 35 declared features were really 34.
    """
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert len(FEATURE_NAMES) == N_FEATURES


def test_no_two_features_are_numerically_identical():
    """Catches a duplicated feature that a name check would miss.

    Uses a non-monotonic series deliberately: on a strictly increasing ramp the
    1-hour rolling maximum equals the current value, which would flag a false
    duplicate.
    """
    from tests.test_sequencing import clean_series, make_subject

    subject = make_subject(clean_series(400), split="train")
    subject.frame["basal_u_per_min"] = 0.02
    # Bolus and meal deliberately at different times, so ``minutes_since_bolus`` and
    # ``minutes_since_meal`` are distinguishable.
    subject.frame.loc[subject.frame.index[100], "bolus_u_per_min"] = 1.0
    subject.frame.loc[subject.frame.index[160], "carbs_mg_per_min"] = 10_000.0
    subject.frame.loc[subject.frame.index[200:210], "exercise_intensity"] = 5.0
    subject.frame.loc[subject.frame.index[300:320], "sleeping"] = 1.0
    subject.frame.loc[subject.frame.index[50:70], "working"] = 1.0

    matrix = build_features(subject)
    values = matrix.values
    finite = matrix.valid_row_mask()

    # Only non-constant columns are compared. A column that is constant in this
    # fixture (an availability mask for a sensor the fixture has no data for, or the
    # interpolation flag when nothing needed interpolating) carries no signal here,
    # so matching another constant column says nothing about the definitions.
    varying = [
        index
        for index in range(values.shape[1])
        if np.ptp(values[finite, index]) > 1e-12
    ]
    assert len(varying) >= 20, "fixture should exercise most features"

    duplicates = [
        (FEATURE_NAMES[i], FEATURE_NAMES[j])
        for position, i in enumerate(varying)
        for j in varying[position + 1 :]
        if np.allclose(values[finite, i], values[finite, j])
    ]
    assert not duplicates, f"numerically identical features: {duplicates}"


def test_matrix_rejects_wrong_names():
    """The contract must be enforced, not merely documented."""
    values = np.zeros((5, N_FEATURES))
    with pytest.raises(FeatureError, match="deviate from the contract"):
        FeatureMatrix(
            subject_id="x",
            values=values,
            names=tuple(reversed(FEATURE_NAMES)),
            index=__import__("pandas").date_range("2021-01-01", periods=5, freq="5min"),
        )


def test_matrix_rejects_column_count_mismatch():
    import pandas as pd

    with pytest.raises(FeatureError, match="columns for"):
        FeatureMatrix(
            subject_id="x",
            values=np.zeros((5, 3)),
            names=FEATURE_NAMES,
            index=pd.date_range("2021-01-01", periods=5, freq="5min"),
        )


def test_provenance_covers_every_feature():
    table = feature_provenance()
    assert list(table["feature"]) == list(FEATURE_NAMES)
    assert table["unit"].notna().all()


# --------------------------------------------------------------------------- #
# Rates, not differences
# --------------------------------------------------------------------------- #


def load_subject_fixture():
    """A small synthetic subject with a known glucose ramp and one gap."""
    from tests.test_sequencing import make_subject

    series: list[float | None] = [100.0 + 2.0 * index for index in range(120)]
    series[60] = None
    series[61] = None
    series[62] = None
    series[63] = None
    series[64] = None  # a 25-minute gap: too long to bridge
    return make_subject(series, split="train")


def test_roc_is_a_rate_in_mg_dl_per_minute():
    """A 2 mg/dL rise per 5-minute slot is 0.4 mg/dL/min, not 2."""
    from tests.test_sequencing import make_subject

    subject = make_subject([100.0 + 2.0 * index for index in range(60)], split="train")
    matrix = build_features(subject)
    roc = matrix.column("roc_5min")
    finite = np.isfinite(roc)
    assert np.allclose(roc[finite], 0.4)


def test_roc_over_longer_lookback_uses_the_right_denominator():
    from tests.test_sequencing import make_subject

    subject = make_subject([100.0 + 2.0 * index for index in range(60)], split="train")
    matrix = build_features(subject)
    for name, minutes in (("roc_15min", 15), ("roc_30min", 30)):
        values = matrix.column(name)
        finite = np.isfinite(values)
        expected = (2.0 * (minutes // 5)) / minutes
        assert np.allclose(values[finite], expected), name


def test_rates_are_nan_across_an_unbridgeable_gap():
    """No carried value may be differenced into a rate.

    Carrying the last observation forward produced rates above 50 mg/dL/min at gap
    edges -- physiologically impossible, and it would have entered model inputs.
    """
    matrix = build_features(load_subject_fixture())
    roc = matrix.column("roc_5min")
    assert np.isnan(roc[60:65]).all()
    # The look-back extends the invalid region one slot past the gap.
    assert np.isnan(roc[65])


def test_nan_appears_only_in_glucose_derived_columns():
    matrix = build_features(load_subject_fixture())
    for column, name in enumerate(FEATURE_NAMES):
        has_nan = not np.isfinite(matrix.values[:, column]).all()
        if has_nan:
            assert name in GLUCOSE_DERIVED, f"{name} unexpectedly contains NaN"


def test_valid_row_mask_extends_past_the_gap_by_the_lookback():
    """``roc_30min`` reads six slots back, so validity must resume six slots later.

    A glucose-only check would admit a window whose earliest rate was differenced
    against a value on the far side of a gap.
    """
    matrix = build_features(load_subject_fixture())
    mask = matrix.valid_row_mask()
    assert not mask[60:65].any(), "gap slots must be invalid"
    # Six slots of look-back after the gap ends at index 64.
    assert not mask[65:71].any(), "look-back region after the gap must be invalid"
    assert mask[71], "validity must resume once the longest look-back clears"


def test_feature_gate_reduces_window_count():
    subject = load_subject_fixture()
    frame = interpolate_bounded(subject.frame)
    matrix = build_features(subject, frame=frame)
    ungated = build_windows(subject, seq_len=24, frame=frame)
    gated = build_windows(subject, seq_len=24, frame=frame, input_valid=matrix.valid_row_mask())
    assert len(gated) <= len(ungated)


def test_input_valid_shape_is_checked():
    from twin.data.sequencing import SequencingError

    subject = load_subject_fixture()
    with pytest.raises(SequencingError, match="input_valid has shape"):
        build_windows(subject, seq_len=24, input_valid=np.ones(5, dtype=bool))


# --------------------------------------------------------------------------- #
# Mechanistic features
# --------------------------------------------------------------------------- #


def test_iob_feature_rises_at_the_bolus():
    """The IOB feature must respond when insulin is given, not 145 minutes later."""
    from tests.test_sequencing import clean_series, make_subject

    subject = make_subject(clean_series(200), split="train")
    subject.frame.loc[subject.frame.index[50], "bolus_u_per_min"] = 5.0 / 5.0

    matrix = build_features(subject)
    iob = matrix.column("iob_u")
    assert iob[49] == pytest.approx(0.0, abs=1e-9)
    assert iob[51] > 4.0, "IOB must be near the delivered dose right after the bolus"
    assert int(np.argmax(iob)) == 51


def test_cob_feature_responds_to_a_meal():
    from tests.test_sequencing import clean_series, make_subject

    subject = make_subject(clean_series(200), split="train")
    subject.frame.loc[subject.frame.index[50], "carbs_mg_per_min"] = 60_000.0 / 5.0

    matrix = build_features(subject)
    cob = matrix.column("cob_g")
    assert cob[49] == pytest.approx(0.0, abs=1e-9)
    assert cob[51] == pytest.approx(60.0, rel=2e-2)
    assert cob[-1] < cob[51]


def test_mechanistic_features_are_independent_of_glucose_gaps():
    """Insulin and carbohydrate states depend only on always-present inputs."""
    matrix = build_features(load_subject_fixture())
    for name in ("iob_u", "cob_g", "insulin_plasma_uU_mL", "insulin_action_per_min"):
        assert np.isfinite(matrix.column(name)).all(), name


def test_minutes_since_starts_at_the_cap():
    """Before any event the honest encoding is 'no recent event', not 'just now'."""
    from twin.data.features import TIME_SINCE_CAP_MIN
    from tests.test_sequencing import clean_series, make_subject

    subject = make_subject(clean_series(60), split="train")
    matrix = build_features(subject)
    assert matrix.column("minutes_since_bolus")[0] == TIME_SINCE_CAP_MIN


def test_unavailable_sensor_value_is_zeroed_and_flagged():
    matrix = build_features(load_subject_fixture())
    assert (matrix.column("basis_heart_rate_available") == 0.0).all()
    assert (matrix.column("basis_heart_rate") == 0.0).all()


# --------------------------------------------------------------------------- #
# Scaler
# --------------------------------------------------------------------------- #


@needs_ohio
@pytest.mark.slow
def test_scaler_single_source():
    """One scaler, bound to the fold that fitted it.

    ``evaluate.py:629-646`` fed two different scalers into the same reported table:
    one refitted on the fly, one loaded from the checkpoint.
    """
    conf = config()
    corpus = {
        "train": {
            data.subject_id: data
            for data in [
                load_subject_data(str(path), conf)
                for path in [OHIO_ROOT / "2018" / "train" / "559-ws-training.xml"]
            ]
        },
        "test": {
            data.subject_id: data
            for data in [
                load_subject_data(str(path), conf)
                for path in [OHIO_ROOT / "2018" / "test" / "559-ws-testing.xml"]
            ]
        },
    }
    train_sets = {k: v.windows for k, v in corpus["train"].items()}
    test_sets = {k: v.windows for k, v in corpus["test"].items()}
    fold = official_split(list(train_sets.values()), list(test_sets.values()))
    verify_no_leakage(fold, train_sets, test_sets)

    scaler = fit_scaler(fold, corpus)
    assert scaler.fold_name == fold.name
    assert scaler.feature_names == FEATURE_NAMES

    # Round-trips through a checkpoint unchanged.
    restored = FittedScaler.from_state_dict(scaler.state_dict())
    assert np.allclose(restored.mean, scaler.mean)
    assert np.allclose(restored.scale, scaler.scale)
    assert restored.fold_name == fold.name


def test_scaler_rejects_wrong_feature_count():
    scaler = FittedScaler(
        mean=np.zeros(N_FEATURES),
        scale=np.ones(N_FEATURES),
        feature_names=FEATURE_NAMES,
        n_slots_fitted=10,
        fold_name="t",
    )
    with pytest.raises(ValueError, match="expected"):
        scaler.transform(np.zeros((4, 3)))


def test_scaler_never_divides_by_zero():
    """A constant feature within a fold must not produce inf or NaN."""
    scaler = FittedScaler(
        mean=np.zeros(2),
        scale=np.array([1.0, 1.0]),
        feature_names=("a", "b"),
        n_slots_fitted=5,
        fold_name="t",
    )
    out = scaler.transform(np.zeros((3, 2)))
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #


@needs_ohio
@pytest.mark.slow
def test_dataset_batches_have_the_declared_shapes():
    conf = config()
    corpus = {
        "train": {},
        "test": {},
    }
    for split, filename in (
        ("train", OHIO_ROOT / "2018" / "train" / "559-ws-training.xml"),
        ("test", OHIO_ROOT / "2018" / "test" / "559-ws-testing.xml"),
    ):
        data = load_subject_data(str(filename), conf)
        corpus[split][data.subject_id] = data

    train_sets = {k: v.windows for k, v in corpus["train"].items()}
    test_sets = {k: v.windows for k, v in corpus["test"].items()}
    fold = official_split(list(train_sets.values()), list(test_sets.values()))
    scaler = fit_scaler(fold, corpus)

    dataset = build_dataset(fold, "train", corpus, scaler, conf)
    assert len(dataset) > 0
    assert dataset.subject_ids == ("559",)

    batch = next(iter(build_loader(dataset, conf, shuffle=False)))
    assert batch["features"].shape == (conf.train.batch_size, conf.data.seq_len, N_FEATURES)
    assert batch["targets"].shape == (conf.train.batch_size, len(conf.data.horizons_min))
    assert batch["insulin_rate"].shape[1] == conf.data.max_horizon_steps + 1
    assert batch["carb_rate"].shape == batch["insulin_rate"].shape
    assert torch.isfinite(batch["features"]).all()
    assert torch.isfinite(batch["targets"]).all()


@needs_ohio
@pytest.mark.slow
def test_predictions_always_ship_with_a_persistence_baseline():
    """A results table cannot be produced without its naive baseline alongside."""
    conf = config()
    corpus = {"train": {}, "test": {}}
    for split, filename in (
        ("train", OHIO_ROOT / "2018" / "train" / "559-ws-training.xml"),
        ("test", OHIO_ROOT / "2018" / "test" / "559-ws-testing.xml"),
    ):
        data = load_subject_data(str(filename), conf)
        corpus[split][data.subject_id] = data

    train_sets = {k: v.windows for k, v in corpus["train"].items()}
    test_sets = {k: v.windows for k, v in corpus["test"].items()}
    fold = official_split(list(train_sets.values()), list(test_sets.values()))

    predictions = collect_predictions(fold, "test", corpus)
    assert set(predictions) == {"559"}
    targets, baseline = predictions["559"]
    assert targets.shape == baseline.shape
    # Persistence repeats the anchor value across horizons.
    assert np.allclose(baseline[:, 0], baseline[:, -1])


def test_build_dataset_rejects_unknown_part():
    from twin.data.splits import Fold

    scaler = FittedScaler(
        mean=np.zeros(N_FEATURES),
        scale=np.ones(N_FEATURES),
        feature_names=FEATURE_NAMES,
        n_slots_fitted=1,
        fold_name="t",
    )
    with pytest.raises(ValueError, match="train/val/test"):
        build_dataset(Fold(name="t", protocol="official"), "bogus", {}, scaler, config())
