"""Verification of the split protocols.

``test_no_window_overlap_across_splits`` and ``test_purge_gap_enforced`` guard the
leak that invalidated the legacy fine-tuning validation set: ``randperm`` over a
pooled window set whose members share 23 of 24 input timesteps, so validation was a
near-duplicate of training and drove both early stopping and the headline metric.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from twin.data.ohio import discover_files, load_subject
from twin.data.sequencing import WindowSet, build_windows
from twin.data.splits import (
    SplitError,
    fold_table,
    loso_splits,
    official_split,
    verify_no_leakage,
)
from twin.data.splits import _occupied_timesteps as occupied_timesteps

from tests.test_sequencing import clean_series, make_subject

OHIO_ROOT = Path("OhioT1DM")
needs_ohio = pytest.mark.skipif(
    not OHIO_ROOT.is_dir(), reason="OhioT1DM data directory not present"
)

HORIZONS = (30, 60, 90, 120)
SEQ_LEN = 24


def synthetic_sets(
    n_subjects: int = 4, n_slots: int = 600
) -> tuple[dict[str, WindowSet], dict[str, WindowSet]]:
    """Window sets for ``n_subjects`` synthetic subjects, train and test."""
    train: dict[str, WindowSet] = {}
    test: dict[str, WindowSet] = {}
    for index in range(n_subjects):
        subject_id = f"s{index}"
        for split, store, slots in (("train", train, n_slots), ("test", test, 250)):
            subject = make_subject(clean_series(slots), split=split)
            subject.subject_id = subject_id
            window_set = build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS)
            window_set.subject_id = subject_id
            window_set.split = split
            store[subject_id] = window_set
    return train, test


# --------------------------------------------------------------------------- #
# Purging and overlap
# --------------------------------------------------------------------------- #


def test_purge_gap_enforced():
    """Training and validation windows must not share a single grid slot."""
    train, test = synthetic_sets()
    fold = official_split(list(train.values()), list(test.values()))
    verify_no_leakage(fold, train, test)

    for selection in fold.val:
        window_set = train[selection.subject_id]
        train_selection = next(s for s in fold.train if s.subject_id == selection.subject_id)
        train_slots = occupied_timesteps(window_set, train_selection.indices)
        val_slots = occupied_timesteps(window_set, selection.indices)
        assert not (train_slots & val_slots)


def test_no_window_overlap_across_splits():
    """The legacy random split must be impossible to reproduce here.

    A too-small purge has to be detected rather than silently tolerated, so this
    checks that ``verify_no_leakage`` actually rejects it.
    """
    train, test = synthetic_sets()
    leaky = official_split(list(train.values()), list(test.values()), purge_steps=0)
    with pytest.raises(SplitError, match="shares .* grid slots"):
        verify_no_leakage(leaky, train, test)


def test_default_purge_is_seq_len_plus_max_horizon():
    train, test = synthetic_sets()
    fold = official_split(list(train.values()), list(test.values()))
    assert fold.purge_steps == SEQ_LEN + max(horizon // 5 for horizon in HORIZONS)
    assert fold.purge_steps == 48


def test_purge_smaller_than_window_span_is_detected():
    """Even a moderate purge is insufficient if smaller than the window span."""
    train, test = synthetic_sets()
    fold = official_split(list(train.values()), list(test.values()), purge_steps=10)
    with pytest.raises(SplitError, match="too small"):
        verify_no_leakage(fold, train, test)


def test_negative_purge_rejected():
    train, test = synthetic_sets()
    with pytest.raises(SplitError, match="non-negative"):
        official_split(list(train.values()), list(test.values()), purge_steps=-1)


# --------------------------------------------------------------------------- #
# Validation is a time-ordered tail, not a random sample
# --------------------------------------------------------------------------- #


def test_validation_is_the_temporal_tail():
    """Validation must be the most recent data, contiguous and after all training."""
    train, test = synthetic_sets()
    fold = official_split(list(train.values()), list(test.values()))

    for selection in fold.val:
        window_set = train[selection.subject_id]
        train_selection = next(s for s in fold.train if s.subject_id == selection.subject_id)

        val_anchors = window_set.anchors[selection.indices]
        train_anchors = window_set.anchors[train_selection.indices]

        assert val_anchors.min() > train_anchors.max(), "validation must follow training in time"
        assert np.all(np.diff(selection.indices) == 1), "validation must be contiguous"
        assert selection.indices.max() == len(window_set) - 1, "validation must be the tail"


def test_validation_fraction_is_respected():
    train, test = synthetic_sets()
    fold = official_split(list(train.values()), list(test.values()), val_fraction=0.2)
    for selection in fold.val:
        total = len(train[selection.subject_id])
        assert len(selection) == int(np.floor(total * 0.2))


def test_too_few_windows_for_a_purged_tail_is_an_error():
    """Refuse to build a split that cannot honour the purge rather than shrink it."""
    subject = make_subject(clean_series(80), split="train")
    window_set = build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS)
    test_subject = make_subject(clean_series(250), split="test")
    test_set = build_windows(test_subject, seq_len=SEQ_LEN, horizons_min=HORIZONS)
    with pytest.raises(SplitError, match="too few for"):
        official_split([window_set], [test_set], val_fraction=0.15, purge_steps=200)


def test_non_monotonic_anchors_rejected():
    """A tail split is only a time split if anchors are ordered."""
    train, test = synthetic_sets(n_subjects=1)
    window_set = next(iter(train.values()))
    window_set.anchors = window_set.anchors[::-1].copy()
    with pytest.raises(SplitError, match="not strictly increasing"):
        official_split([window_set], list(test.values()))


# --------------------------------------------------------------------------- #
# LOSO
# --------------------------------------------------------------------------- #


def test_loso_produces_one_fold_per_subject():
    train, test = synthetic_sets(n_subjects=5)
    folds = loso_splits(list(train.values()), list(test.values()))
    assert len(folds) == 5
    assert {fold.held_out_subject for fold in folds} == set(train)


def test_loso_subject_disjoint():
    """The held-out subject must appear nowhere in training or validation."""
    train, test = synthetic_sets(n_subjects=5)
    for fold in loso_splits(list(train.values()), list(test.values())):
        verify_no_leakage(fold, train, test)
        held_out = fold.held_out_subject
        assert held_out not in fold.subjects("train")
        assert held_out not in fold.subjects("val")
        assert fold.subjects("test") == (held_out,)
        assert len(fold.subjects("train")) == 4


def test_loso_detects_a_contaminated_fold():
    """The disjointness check must actually fire when violated."""
    train, test = synthetic_sets(n_subjects=4)
    folds = loso_splits(list(train.values()), list(test.values()))
    fold = folds[0]
    held_out = fold.held_out_subject
    fold.train.append(
        type(fold.train[0])(held_out, np.arange(5, dtype=np.int64))
    )
    with pytest.raises(SplitError, match="appears in training"):
        verify_no_leakage(fold, train, test)


def test_loso_requires_test_windows_for_every_subject():
    train, test = synthetic_sets(n_subjects=3)
    test.pop(next(iter(test)))
    with pytest.raises(SplitError, match="no test windows"):
        loso_splits(list(train.values()), list(test.values()))


def test_loso_and_official_score_identical_test_windows():
    """The two protocols must differ only in training composition.

    Holding the test set fixed is what makes the official-versus-LOSO difference
    interpretable as the value of subject-specific history, rather than a mixture of
    that and a different evaluation set.
    """
    train, test = synthetic_sets(n_subjects=5)
    official = official_split(list(train.values()), list(test.values()))
    folds = loso_splits(list(train.values()), list(test.values()))

    official_total = official.counts()["test"]
    loso_total = sum(fold.counts()["test"] for fold in folds)
    assert official_total == loso_total

    official_by_subject = {s.subject_id: set(s.indices.tolist()) for s in official.test}
    for fold in folds:
        selection = fold.test[0]
        assert set(selection.indices.tolist()) == official_by_subject[selection.subject_id]


# --------------------------------------------------------------------------- #
# Bookkeeping
# --------------------------------------------------------------------------- #


def test_selection_indices_validated():
    train, test = synthetic_sets(n_subjects=2)
    fold = official_split(list(train.values()), list(test.values()))
    fold.train[0].indices[0] = 10**9
    with pytest.raises(SplitError, match="out-of-range"):
        verify_no_leakage(fold, train, test)


def test_duplicate_indices_rejected():
    train, test = synthetic_sets(n_subjects=2)
    fold = official_split(list(train.values()), list(test.values()))
    selection = fold.train[0]
    selection.indices[1] = selection.indices[0]
    with pytest.raises(SplitError, match="duplicate indices"):
        verify_no_leakage(fold, train, test)


def test_fold_table_reports_counts():
    train, test = synthetic_sets(n_subjects=3)
    table = fold_table(loso_splits(list(train.values()), list(test.values())))
    assert len(table) == 3
    for column in ("n_train", "n_val", "n_test", "held_out_subject", "purge_steps"):
        assert column in table.columns
    assert (table["test_subjects"] == 1).all()
    assert (table["train_subjects"] == 2).all()


# --------------------------------------------------------------------------- #
# Real corpus
# --------------------------------------------------------------------------- #


@needs_ohio
@pytest.mark.slow
def test_real_splits_pass_leakage_verification():
    train, test = {}, {}
    for path in discover_files(OHIO_ROOT, split="train"):
        subject = load_subject(path)
        train[subject.subject_id] = build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS)
    for path in discover_files(OHIO_ROOT, split="test"):
        subject = load_subject(path)
        test[subject.subject_id] = build_windows(subject, seq_len=SEQ_LEN, horizons_min=HORIZONS)

    official = official_split(list(train.values()), list(test.values()))
    verify_no_leakage(official, train, test)
    assert len(official.subjects("train")) == 12

    folds = loso_splits(list(train.values()), list(test.values()))
    assert len(folds) == 12
    for fold in folds:
        verify_no_leakage(fold, train, test)

    assert official.counts()["test"] == sum(fold.counts()["test"] for fold in folds)
