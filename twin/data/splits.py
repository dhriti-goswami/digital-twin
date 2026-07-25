"""Split protocols: official temporal holdout and leave-one-subject-out.

Two protocols are implemented, and they are evaluated on the **identical test
windows**. The only difference is whether the test subject's own earlier data was
available during training:

``official``
    The OhioT1DM protocol. Train on all 12 subjects' training files, test on their
    test files (the next contiguous ~10 days). Measures *personalised* forecasting;
    every published Ohio number uses this. It is not cross-subject generalisation
    and must not be described as such.

``loso``
    Leave-one-subject-out, 12 folds. Train on the other 11 subjects' training
    files, test on the held-out subject's test file. Genuinely subject-disjoint.

Because both score the same test windows, the difference between them isolates the
value of subject-specific history cleanly. That contrast is more informative than
either number alone.

Leakage control
---------------
Inner validation is the **time-ordered tail** of each training subject's record,
never a random sample. The legacy pipeline shuffled the pooled window set and took
15% as validation (``finetune_ohio.py:232-239``); since consecutive windows share
23 of 24 input timesteps, validation was a near-duplicate of training, and it drove
early stopping and the reported headline metric.

A **purge gap** of at least ``seq_len + max_horizon_steps`` windows is dropped at
the train/validation boundary, which is the minimum that guarantees no training
window shares a timestep with any validation window.

No purge is applied at the train/test boundary. Windows are only ever built within
a single file's frame, so a training window's targets always lie inside the
training period -- there is no overlap to purge, and inserting one would deviate
from the published protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from twin.data.sequencing import WindowSet


class SplitError(ValueError):
    """Raised for an inconsistent or leaky split request."""


@dataclass(frozen=True)
class Selection:
    """Which windows of which subject belong to a split part."""

    subject_id: str
    #: Positional indices into that subject's :class:`WindowSet`.
    indices: NDArray[np.int64]

    def __len__(self) -> int:
        return int(self.indices.size)


@dataclass
class Fold:
    """One train/validation/test assignment."""

    name: str
    protocol: str
    train: list[Selection] = field(default_factory=list)
    val: list[Selection] = field(default_factory=list)
    test: list[Selection] = field(default_factory=list)
    held_out_subject: str | None = None
    purge_steps: int = 0

    def counts(self) -> dict[str, int]:
        return {
            "train": sum(len(selection) for selection in self.train),
            "val": sum(len(selection) for selection in self.val),
            "test": sum(len(selection) for selection in self.test),
        }

    def subjects(self, part: str) -> tuple[str, ...]:
        return tuple(sorted({selection.subject_id for selection in getattr(self, part)}))

    def summary(self) -> dict[str, object]:
        counts = self.counts()
        return {
            "fold": self.name,
            "protocol": self.protocol,
            "held_out_subject": self.held_out_subject or "",
            "n_train": counts["train"],
            "n_val": counts["val"],
            "n_test": counts["test"],
            "train_subjects": len(self.subjects("train")),
            "test_subjects": len(self.subjects("test")),
            "purge_steps": self.purge_steps,
        }


def _time_ordered_tail_split(
    window_set: WindowSet, *, val_fraction: float, purge_steps: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Split one subject's windows into (train, val) by time, with a purge gap.

    Windows are already in ascending anchor order. The final ``val_fraction`` become
    validation; the ``purge_steps`` windows immediately before them are discarded so
    no training window can share a timestep with a validation window.
    """
    total = len(window_set)
    if total == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    # Anchors must be ascending for a tail split to be a time split at all.
    if not np.all(np.diff(window_set.anchors) > 0):
        raise SplitError(f"{window_set.subject_id}: window anchors are not strictly increasing")

    n_val = int(np.floor(total * val_fraction))
    if n_val == 0:
        return np.arange(total, dtype=np.int64), np.empty(0, dtype=np.int64)

    val_start = total - n_val
    train_end = val_start - purge_steps
    if train_end <= 0:
        raise SplitError(
            f"{window_set.subject_id}: only {total} windows, too few for a "
            f"{val_fraction:.0%} validation tail plus a {purge_steps}-window purge"
        )

    return (
        np.arange(train_end, dtype=np.int64),
        np.arange(val_start, total, dtype=np.int64),
    )


def official_split(
    train_sets: list[WindowSet],
    test_sets: list[WindowSet],
    *,
    val_fraction: float = 0.15,
    purge_steps: int | None = None,
) -> Fold:
    """The OhioT1DM temporal holdout, with a purged time-ordered inner validation."""
    if not train_sets:
        raise SplitError("no training window sets supplied")
    purge = _resolve_purge(train_sets, purge_steps)

    fold = Fold(name="official", protocol="official", purge_steps=purge)
    for window_set in sorted(train_sets, key=lambda item: item.subject_id):
        train_indices, val_indices = _time_ordered_tail_split(
            window_set, val_fraction=val_fraction, purge_steps=purge
        )
        fold.train.append(Selection(window_set.subject_id, train_indices))
        if val_indices.size:
            fold.val.append(Selection(window_set.subject_id, val_indices))
    for window_set in sorted(test_sets, key=lambda item: item.subject_id):
        fold.test.append(Selection(window_set.subject_id, np.arange(len(window_set), dtype=np.int64)))
    return fold


def loso_splits(
    train_sets: list[WindowSet],
    test_sets: list[WindowSet],
    *,
    val_fraction: float = 0.15,
    purge_steps: int | None = None,
) -> list[Fold]:
    """Leave-one-subject-out folds, one per subject.

    Each fold tests on the held-out subject's **test file**, the same windows the
    official protocol scores, so the two protocols differ only in whether that
    subject's earlier data was seen during training.
    """
    if not train_sets:
        raise SplitError("no training window sets supplied")
    purge = _resolve_purge(train_sets, purge_steps)

    by_subject_test = {window_set.subject_id: window_set for window_set in test_sets}
    subjects = sorted({window_set.subject_id for window_set in train_sets})
    missing = [subject for subject in subjects if subject not in by_subject_test]
    if missing:
        raise SplitError(f"no test windows for held-out subject(s): {missing}")

    folds: list[Fold] = []
    for held_out in subjects:
        fold = Fold(
            name=f"loso-{held_out}",
            protocol="loso",
            held_out_subject=held_out,
            purge_steps=purge,
        )
        for window_set in sorted(train_sets, key=lambda item: item.subject_id):
            if window_set.subject_id == held_out:
                continue
            train_indices, val_indices = _time_ordered_tail_split(
                window_set, val_fraction=val_fraction, purge_steps=purge
            )
            fold.train.append(Selection(window_set.subject_id, train_indices))
            if val_indices.size:
                fold.val.append(Selection(window_set.subject_id, val_indices))

        test_set = by_subject_test[held_out]
        fold.test.append(Selection(held_out, np.arange(len(test_set), dtype=np.int64)))
        folds.append(fold)
    return folds


def _resolve_purge(window_sets: list[WindowSet], purge_steps: int | None) -> int:
    """Default the purge to the minimum safe value: ``seq_len + max_horizon_steps``."""
    if purge_steps is not None:
        if purge_steps < 0:
            raise SplitError("purge_steps must be non-negative")
        return int(purge_steps)
    reference = window_sets[0]
    return reference.seq_len + max(reference.horizon_steps)


# --------------------------------------------------------------------------- #
# Leakage verification
# --------------------------------------------------------------------------- #


def _occupied_timesteps(window_set: WindowSet, indices: NDArray[np.int64]) -> set[int]:
    """Every grid slot a set of windows touches, inputs and targets alike.

    A window reads ``[anchor - seq_len + 1, anchor]`` and is scored at
    ``anchor + step`` for each horizon. Two windows leak into each other if these
    sets intersect at all.
    """
    occupied: set[int] = set()
    anchors = window_set.anchors[indices]
    for anchor in anchors.tolist():
        occupied.update(range(anchor - window_set.seq_len + 1, anchor + 1))
        for step in window_set.horizon_steps:
            occupied.add(anchor + step)
    return occupied


def verify_no_leakage(
    fold: Fold,
    train_sets: dict[str, WindowSet],
    test_sets: dict[str, WindowSet],
) -> None:
    """Assert the fold is free of the leaks the legacy pipeline had.

    Checks, in order:

    1. No training window shares a grid slot with a validation window, for any
       subject. This is what the purge gap exists to guarantee.
    2. Under ``loso``, the held-out subject appears in neither train nor validation.
    3. No subject appears in both the training and test *selections* of a LOSO fold.
    4. Selections index within range and contain no duplicates.
    """
    for part, source in (("train", train_sets), ("val", train_sets), ("test", test_sets)):
        for selection in getattr(fold, part):
            window_set = source.get(selection.subject_id)
            if window_set is None:
                raise SplitError(f"{fold.name}: no window set for {selection.subject_id} in {part}")
            if selection.indices.size != np.unique(selection.indices).size:
                raise SplitError(f"{fold.name}: duplicate indices in {part}/{selection.subject_id}")
            if selection.indices.size and (
                selection.indices.min() < 0 or selection.indices.max() >= len(window_set)
            ):
                raise SplitError(
                    f"{fold.name}: out-of-range indices in {part}/{selection.subject_id}"
                )

    train_by_subject = {selection.subject_id: selection for selection in fold.train}
    val_by_subject = {selection.subject_id: selection for selection in fold.val}

    for subject_id, val_selection in val_by_subject.items():
        train_selection = train_by_subject.get(subject_id)
        if train_selection is None:
            continue
        window_set = train_sets[subject_id]
        train_slots = _occupied_timesteps(window_set, train_selection.indices)
        val_slots = _occupied_timesteps(window_set, val_selection.indices)
        shared = train_slots & val_slots
        if shared:
            raise SplitError(
                f"{fold.name}: subject {subject_id} shares {len(shared)} grid slots "
                f"between train and validation; purge_steps={fold.purge_steps} is too small"
            )

    if fold.protocol == "loso":
        held_out = fold.held_out_subject
        if held_out is None:
            raise SplitError(f"{fold.name}: loso fold without a held-out subject")
        if held_out in train_by_subject or held_out in val_by_subject:
            raise SplitError(f"{fold.name}: held-out subject {held_out} appears in training")
        test_subjects = set(fold.subjects("test"))
        overlap = test_subjects & set(train_by_subject)
        if overlap:
            raise SplitError(f"{fold.name}: subjects in both train and test: {sorted(overlap)}")


def fold_table(folds: list[Fold]) -> pd.DataFrame:
    """Per-fold window and subject counts, for the paper's methods section."""
    return pd.DataFrame([fold.summary() for fold in folds])


__all__ = [
    "Fold",
    "Selection",
    "SplitError",
    "fold_table",
    "loso_splits",
    "official_split",
    "verify_no_leakage",
]
