"""Gap-aware window construction.

This module exists because of the single most damaging defect in the legacy
pipeline. ``scripts/evaluate_ohio.py:254-256`` dropped rows with missing CGM and
then called ``reset_index(drop=True)`` before slicing windows out of the compacted
array. The consequence: for any window spanning a gap, the value labelled
"+30 minutes" was not 30 minutes ahead. Every Ohio number in the repository was
computed against mislabelled targets.

The contract enforced here
--------------------------
A window is emitted only when **all** of the following hold:

1. Its input span is ``seq_len`` consecutive slots on the exact time grid --
   guaranteed structurally, because :mod:`twin.data.ohio` never removes a row.
2. Every input slot has a glucose value, after interpolating runs of at most
   ``max_interp_gap`` missing slots. Longer runs are never bridged.
3. The interpolated fraction of the input does not exceed
   ``1 - min_input_coverage``.
4. **Every target is a real observation at exactly the nominal horizon.** Targets
   are never interpolated and never forward-filled.

Point 4 is not merely a quality filter. The legacy code forward-filled gaps of up
to 15 minutes and then used the filled values *as targets*, so a "30-minute
prediction" could be scored against a copy of a reading the model had already
seen -- which persistence predicts perfectly. That inflates short-horizon accuracy
by construction.

Windows will be lost relative to the legacy counts. That is the point, and
:class:`WindowReport` records exactly how many and why so the paper can state it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from twin.data.ohio import GriddedSubject

GLUCOSE_COLUMN = "glucose_mg_dl"
OBSERVED_COLUMN = "glucose_observed"
FILLED_COLUMN = "glucose_filled"
INTERPOLATED_COLUMN = "glucose_interpolated"


class SequencingError(ValueError):
    """Raised for an inconsistent sequencing request."""


def interpolate_bounded(
    frame: pd.DataFrame, *, max_interp_gap: int = 2
) -> pd.DataFrame:
    """Add bounded-interpolation columns to a gridded frame.

    A run of consecutive missing slots is filled linearly **only if the entire run
    is at most ``max_interp_gap`` long** and it is interior to the record.

    That is stricter than ``Series.interpolate(limit=n)``, which fills the *first*
    ``n`` values of every run: with ``limit=2``, a three-hour outage would receive
    two fabricated values grafted onto its leading edge, and those values could then
    enter an input window. Run length is therefore tested explicitly.

    Adds:

    ``glucose_filled``
        Glucose with short interior gaps filled; ``NaN`` elsewhere.
    ``glucose_interpolated``
        ``True`` where the value came from interpolation rather than the sensor.

    The original ``glucose_mg_dl`` and ``glucose_observed`` are left untouched, so
    the distinction between measured and inferred is never lost.
    """
    if max_interp_gap < 0:
        raise SequencingError("max_interp_gap must be non-negative")

    out = frame.copy()
    glucose = out[GLUCOSE_COLUMN]
    missing = glucose.isna().to_numpy()

    if max_interp_gap == 0 or not missing.any() or missing.all():
        out[FILLED_COLUMN] = glucose.copy()
        out[INTERPOLATED_COLUMN] = False
        return out

    # Label maximal runs of equal missingness, then measure each run's length.
    boundaries = np.concatenate([[True], missing[1:] != missing[:-1]])
    run_id = np.cumsum(boundaries) - 1
    run_length = np.bincount(run_id)[run_id]

    # Interior only: never extrapolate before the first or after the last reading.
    present = np.flatnonzero(~missing)
    first, last = present[0], present[-1]
    position = np.arange(missing.size)

    fillable = missing & (run_length <= max_interp_gap) & (position > first) & (position < last)

    dense = glucose.interpolate(method="linear", limit_area="inside")
    filled = glucose.copy()
    filled.iloc[fillable] = dense.iloc[fillable]

    out[FILLED_COLUMN] = filled
    out[INTERPOLATED_COLUMN] = pd.Series(fillable, index=out.index)
    return out


@dataclass
class WindowReport:
    """Why windows were kept or rejected, for disclosure in the paper."""

    subject_id: str
    cohort: str
    split: str
    n_slots: int
    n_candidates: int
    kept: int
    rejected_input_missing: int
    rejected_low_coverage: int
    rejected_target_missing: int
    #: Per-horizon count of rejections attributable to that horizon alone.
    rejected_by_horizon: dict[int, int] = field(default_factory=dict)

    @property
    def keep_rate(self) -> float:
        return self.kept / self.n_candidates if self.n_candidates else 0.0

    def as_dict(self) -> dict[str, object]:
        out = {
            "subject_id": self.subject_id,
            "cohort": self.cohort,
            "split": self.split,
            "n_slots": self.n_slots,
            "n_candidates": self.n_candidates,
            "kept": self.kept,
            "keep_rate": round(self.keep_rate, 4),
            "rejected_input_missing": self.rejected_input_missing,
            "rejected_low_coverage": self.rejected_low_coverage,
            "rejected_target_missing": self.rejected_target_missing,
        }
        out.update(
            {f"rejected_h{horizon}": count for horizon, count in sorted(self.rejected_by_horizon.items())}
        )
        return out


@dataclass
class WindowSet:
    """Valid windows for one subject.

    ``anchors`` holds the positional index of each window's **last input slot**, so
    the input span is ``[anchor - seq_len + 1, anchor]`` and target ``h`` sits at
    ``anchor + h`` steps. Storing the anchor rather than the start makes the horizon
    arithmetic unambiguous, which is where the legacy code went wrong.
    """

    subject_id: str
    cohort: str
    split: str
    seq_len: int
    horizon_steps: tuple[int, ...]
    horizons_min: tuple[int, ...]
    grid_minutes: int
    anchors: NDArray[np.int64]
    targets: NDArray[np.float64]
    anchor_times: pd.DatetimeIndex
    #: Fraction of each window's inputs that were interpolated rather than measured.
    interpolated_fraction: NDArray[np.float64]
    report: WindowReport

    def __len__(self) -> int:
        return int(self.anchors.size)

    @property
    def input_starts(self) -> NDArray[np.int64]:
        return self.anchors - self.seq_len + 1

    def verify(self, frame: pd.DataFrame) -> None:
        """Re-derive the horizon guarantee from the frame and assert it holds.

        Cheap, and it converts the central correctness claim of this module from a
        comment into a runtime check. Called by the dataset builder.
        """
        observed = frame[OBSERVED_COLUMN].to_numpy(dtype=bool)
        glucose = frame[GLUCOSE_COLUMN].to_numpy(dtype=np.float64)
        times = frame.index

        for column, step in enumerate(self.horizon_steps):
            positions = self.anchors + step
            if not observed[positions].all():
                raise SequencingError(
                    f"{self.subject_id}: horizon {self.horizons_min[column]} min has "
                    "targets that are not real observations"
                )
            if not np.allclose(glucose[positions], self.targets[:, column], equal_nan=False):
                raise SequencingError(
                    f"{self.subject_id}: stored targets disagree with the frame at "
                    f"horizon {self.horizons_min[column]} min"
                )
            # The elapsed wall-clock time must equal the nominal horizon exactly.
            elapsed = (
                times[positions].to_numpy() - times[self.anchors].to_numpy()
            ).astype("timedelta64[m]").astype(np.int64)
            expected = self.horizons_min[column]
            if not np.all(elapsed == expected):
                offenders = int(np.count_nonzero(elapsed != expected))
                raise SequencingError(
                    f"{self.subject_id}: {offenders} windows have an actual elapsed "
                    f"time different from the nominal {expected} min horizon"
                )


def build_windows(
    subject: GriddedSubject,
    *,
    seq_len: int = 24,
    horizons_min: tuple[int, ...] = (30, 60, 90, 120),
    min_input_coverage: float = 0.9,
    max_interp_gap: int = 2,
    frame: pd.DataFrame | None = None,
    input_valid: NDArray[np.bool_] | None = None,
) -> WindowSet:
    """Emit every valid window for one subject.

    ``frame`` may be supplied to reuse an already-interpolated frame; otherwise
    :func:`interpolate_bounded` is applied here.

    ``input_valid`` is an optional per-slot mask that every input slot must satisfy,
    supplied by :meth:`twin.data.features.FeatureMatrix.valid_row_mask`. It extends
    the exclusion zone around a gap to cover each feature's look-back: ``roc_30min``
    reads six slots back, so a glucose-only check would admit a rate differenced
    against a value from the far side of a gap. Passing the mask is required for any
    window set used to train or evaluate a model; it is optional here only so the
    module can be tested without building features.
    """
    grid = subject.grid_minutes
    if any(horizon % grid for horizon in horizons_min):
        raise SequencingError(
            f"horizons {horizons_min} must be multiples of the {grid}-minute grid"
        )
    horizon_steps = tuple(horizon // grid for horizon in horizons_min)
    max_step = max(horizon_steps)

    work = interpolate_bounded(subject.frame, max_interp_gap=max_interp_gap) if frame is None else frame
    n_slots = len(work)

    filled = work[FILLED_COLUMN].to_numpy(dtype=np.float64)
    interpolated = work[INTERPOLATED_COLUMN].to_numpy(dtype=bool)
    observed = work[OBSERVED_COLUMN].to_numpy(dtype=bool)
    glucose = work[GLUCOSE_COLUMN].to_numpy(dtype=np.float64)

    # Candidate anchors: enough history behind and enough room ahead.
    first_anchor = seq_len - 1
    last_anchor = n_slots - max_step - 1
    if last_anchor < first_anchor:
        empty_report = WindowReport(
            subject_id=subject.subject_id,
            cohort=subject.cohort,
            split=subject.split,
            n_slots=n_slots,
            n_candidates=0,
            kept=0,
            rejected_input_missing=0,
            rejected_low_coverage=0,
            rejected_target_missing=0,
        )
        return WindowSet(
            subject_id=subject.subject_id,
            cohort=subject.cohort,
            split=subject.split,
            seq_len=seq_len,
            horizon_steps=horizon_steps,
            horizons_min=horizons_min,
            grid_minutes=grid,
            anchors=np.empty(0, dtype=np.int64),
            targets=np.empty((0, len(horizons_min)), dtype=np.float64),
            anchor_times=work.index[:0],
            interpolated_fraction=np.empty(0, dtype=np.float64),
            report=empty_report,
        )

    candidates = np.arange(first_anchor, last_anchor + 1, dtype=np.int64)

    # Rolling counts over the input span via prefix sums.
    has_value = np.isfinite(filled)
    if input_valid is not None:
        if input_valid.shape != has_value.shape:
            raise SequencingError(
                f"input_valid has shape {input_valid.shape}, expected {has_value.shape}"
            )
        has_value = has_value & input_valid
    value_prefix = np.concatenate([[0], np.cumsum(has_value)])
    interp_prefix = np.concatenate([[0], np.cumsum(interpolated)])

    starts = candidates - seq_len + 1
    n_values = value_prefix[candidates + 1] - value_prefix[starts]
    n_interp = interp_prefix[candidates + 1] - interp_prefix[starts]

    input_complete = n_values == seq_len
    coverage_ok = (seq_len - n_interp) >= min_input_coverage * seq_len

    # Targets must be genuine observations at exactly the nominal offsets.
    target_ok = np.ones(candidates.size, dtype=bool)
    rejected_by_horizon: dict[int, int] = {}
    for horizon, step in zip(horizons_min, horizon_steps, strict=True):
        this_horizon = observed[candidates + step]
        rejected_by_horizon[horizon] = int(np.count_nonzero(~this_horizon))
        target_ok &= this_horizon

    valid = input_complete & coverage_ok & target_ok
    anchors = candidates[valid]

    targets = np.stack(
        [glucose[anchors + step] for step in horizon_steps], axis=-1
    ) if anchors.size else np.empty((0, len(horizons_min)), dtype=np.float64)

    interpolated_fraction = (
        n_interp[valid].astype(np.float64) / seq_len
        if anchors.size
        else np.empty(0, dtype=np.float64)
    )

    report = WindowReport(
        subject_id=subject.subject_id,
        cohort=subject.cohort,
        split=subject.split,
        n_slots=n_slots,
        n_candidates=int(candidates.size),
        kept=int(anchors.size),
        rejected_input_missing=int(np.count_nonzero(~input_complete)),
        rejected_low_coverage=int(np.count_nonzero(input_complete & ~coverage_ok)),
        rejected_target_missing=int(np.count_nonzero(input_complete & coverage_ok & ~target_ok)),
        rejected_by_horizon=rejected_by_horizon,
    )

    window_set = WindowSet(
        subject_id=subject.subject_id,
        cohort=subject.cohort,
        split=subject.split,
        seq_len=seq_len,
        horizon_steps=horizon_steps,
        horizons_min=horizons_min,
        grid_minutes=grid,
        anchors=anchors,
        targets=targets,
        anchor_times=work.index[anchors],
        interpolated_fraction=interpolated_fraction,
        report=report,
    )
    window_set.verify(work)
    return window_set


def window_report_table(window_sets: list[WindowSet]) -> pd.DataFrame:
    """Per-subject window accounting, for the paper's data section.

    Reporting this is not optional: the counts here are substantially lower than the
    legacy ones, and a reader must be able to see that the reduction comes from
    enforcing horizon integrity rather than from losing data carelessly.
    """
    return pd.DataFrame([window_set.report.as_dict() for window_set in window_sets])


def persistence_targets(
    subject: GriddedSubject, window_set: WindowSet, *, frame: pd.DataFrame | None = None
) -> NDArray[np.float64]:
    """The naive "glucose does not change" forecast for each window.

    Returns ``(n_windows, n_horizons)``, every column equal to the glucose value at
    the window's anchor. This is the baseline the legacy pipeline never computed and
    which both of its headline numbers lose to.
    """
    work = interpolate_bounded(subject.frame) if frame is None else frame
    anchor_glucose = work[FILLED_COLUMN].to_numpy(dtype=np.float64)[window_set.anchors]
    return np.repeat(anchor_glucose[:, None], len(window_set.horizons_min), axis=1)


__all__ = [
    "FILLED_COLUMN",
    "GLUCOSE_COLUMN",
    "INTERPOLATED_COLUMN",
    "OBSERVED_COLUMN",
    "SequencingError",
    "WindowReport",
    "WindowSet",
    "build_windows",
    "interpolate_bounded",
    "persistence_targets",
    "window_report_table",
]
