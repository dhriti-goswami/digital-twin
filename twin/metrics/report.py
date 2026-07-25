"""Per-subject metric computation and across-subject aggregation.

The reporting rule for this project, applied without exception:

**Compute per subject, then aggregate across subjects.** The headline number is
``mean +/- SD across subjects``; pooled-over-all-windows figures appear only as a
clearly labelled secondary column.

Why it matters here specifically: OhioT1DM subjects contribute wildly different
window counts, and the legacy pipeline pooled everything with ``np.concatenate``
before computing metrics. One subject with 2137 windows then carried six times
the weight of a subject with 356, so the reported number described the
best-instrumented subjects rather than the cohort. Per-subject-then-aggregate is
also the standard OhioT1DM reporting format, without which the numbers cannot be
compared to published work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from twin.metrics.accuracy import MetricError, accuracy_by_horizon
from twin.metrics.clinical import (
    coefficient_of_variation,
    hypoglycaemia_detection,
    range_agreement,
    risk_indices,
)
from twin.metrics.errorgrid import zone_summary
from twin.metrics.stats import bootstrap_ci

Array = NDArray[np.floating]


@dataclass
class SubjectPredictions:
    """One subject's aligned reference and prediction arrays.

    ``y_true`` and ``y_pred`` are ``(n_windows, n_horizons)``. Both must be real
    observations at exactly the nominal horizons -- the sequencing layer
    guarantees that, and :meth:`validate` re-checks the shape contract here so a
    silent misalignment cannot reach a table.
    """

    subject_id: str
    y_true: Array
    y_pred: Array
    cohort: str = ""
    #: Optional contiguous series for lag analysis, at the first horizon.
    series_true: Array | None = None
    series_pred: Array | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(self, n_horizons: int) -> None:
        if self.y_true.shape != self.y_pred.shape:
            raise MetricError(
                f"subject {self.subject_id}: shape mismatch "
                f"{self.y_true.shape} vs {self.y_pred.shape}"
            )
        if self.y_true.ndim != 2 or self.y_true.shape[1] != n_horizons:
            raise MetricError(
                f"subject {self.subject_id}: expected (n, {n_horizons}), got {self.y_true.shape}"
            )
        if self.y_true.shape[0] == 0:
            raise MetricError(f"subject {self.subject_id}: no windows")

    @property
    def n_windows(self) -> int:
        return int(self.y_true.shape[0])


def subject_metrics(
    subject: SubjectPredictions,
    horizons_min: tuple[int, ...],
    *,
    include_error_grid: bool = True,
) -> pd.DataFrame:
    """All metrics for one subject, one row per horizon.

    ``include_error_grid`` exists because the error-grid boundaries are gated on
    primary-source verification; development runs can proceed without them while
    reported runs cannot.
    """
    subject.validate(len(horizons_min))
    rows: list[dict[str, object]] = []

    accuracy = accuracy_by_horizon(subject.y_true, subject.y_pred, horizons_min)
    for index, record in enumerate(accuracy):
        reference = subject.y_true[:, index]
        prediction = subject.y_pred[:, index]

        row: dict[str, object] = {
            "subject_id": subject.subject_id,
            "cohort": subject.cohort,
            **record.as_dict(),
        }

        agreement = range_agreement(reference, prediction)
        row.update(agreement.as_dict())
        row["actual_cv"] = coefficient_of_variation(reference)
        row["predicted_cv"] = coefficient_of_variation(prediction)
        # Excursion compression: a model that shrinks variability toward the mean
        # gets a flattering RMSE, and this is where it becomes visible.
        row["cv_ratio"] = row["predicted_cv"] / row["actual_cv"]

        actual_risk = risk_indices(reference)
        predicted_risk = risk_indices(prediction)
        row["actual_lbgi"] = actual_risk.lbgi
        row["actual_hbgi"] = actual_risk.hbgi
        row["predicted_lbgi"] = predicted_risk.lbgi
        row["predicted_hbgi"] = predicted_risk.hbgi

        detection = hypoglycaemia_detection(reference, prediction)
        row["hypo_sensitivity"] = detection["sensitivity"]
        row["hypo_specificity"] = detection["specificity"]
        row["hypo_n_events"] = detection["n_actual_events"]

        if include_error_grid:
            row.update(zone_summary(reference, prediction, grid="clarke").as_dict())

        rows.append(row)

    return pd.DataFrame(rows)


def per_subject_table(
    subjects: list[SubjectPredictions],
    horizons_min: tuple[int, ...],
    *,
    include_error_grid: bool = True,
) -> pd.DataFrame:
    """Concatenated per-subject, per-horizon metrics."""
    if not subjects:
        raise MetricError("no subjects supplied")
    frames = [
        subject_metrics(subject, horizons_min, include_error_grid=include_error_grid)
        for subject in subjects
    ]
    return pd.concat(frames, ignore_index=True)


#: Columns aggregated as mean +/- SD across subjects. Counts are summed instead.
_SUMMED_COLUMNS = frozenset({"n", "hypo_n_events"})
_IDENTIFIER_COLUMNS = frozenset({"subject_id", "cohort", "horizon_min"})


def across_subject_summary(
    per_subject: pd.DataFrame,
    *,
    ci_columns: tuple[str, ...] = ("rmse", "mae"),
    seed: int = 42,
) -> pd.DataFrame:
    """Aggregate a per-subject table into the headline across-subject summary.

    For every metric this reports the mean, SD, median, min and max **across
    subjects**, plus the number of subjects contributing. Bootstrap confidence
    intervals (resampling subjects) are added for the columns named in
    ``ci_columns``.

    ``n`` is summed rather than averaged, and reported so a reader can see the
    window count behind each row.
    """
    if per_subject.empty:
        raise MetricError("empty per-subject table")

    numeric = [
        column
        for column in per_subject.columns
        if column not in _IDENTIFIER_COLUMNS
        and pd.api.types.is_numeric_dtype(per_subject[column])
    ]

    rows: list[dict[str, object]] = []
    for horizon, group in per_subject.groupby("horizon_min", sort=True):
        row: dict[str, object] = {
            "horizon_min": int(horizon),
            "n_subjects": int(group["subject_id"].nunique()),
            "n_windows_total": int(group["n"].sum()) if "n" in group else np.nan,
        }
        for column in numeric:
            values = group[column].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            if column in _SUMMED_COLUMNS:
                row[f"{column}_total"] = float(np.nansum(values))
                continue
            if finite.size == 0:
                row[f"{column}_mean"] = np.nan
                row[f"{column}_sd"] = np.nan
                continue
            row[f"{column}_mean"] = float(finite.mean())
            # Sample SD: these are 12 subjects drawn from a population, not the
            # population itself.
            row[f"{column}_sd"] = float(finite.std(ddof=1)) if finite.size > 1 else np.nan
            row[f"{column}_median"] = float(np.median(finite))
            row[f"{column}_min"] = float(finite.min())
            row[f"{column}_max"] = float(finite.max())

            if column in ci_columns:
                interval = bootstrap_ci(finite, seed=seed)
                row[f"{column}_ci_low"] = interval.low
                row[f"{column}_ci_high"] = interval.high

        rows.append(row)

    return pd.DataFrame(rows)


def format_mean_sd(summary: pd.DataFrame, metric: str, *, decimals: int = 2) -> pd.Series:
    """``mean +/- SD`` strings for a metric, for direct use in a paper table."""
    mean_column, sd_column = f"{metric}_mean", f"{metric}_sd"
    for column in (mean_column, sd_column):
        if column not in summary:
            raise MetricError(f"{column} not in summary; was {metric!r} aggregated?")
    return summary.apply(
        lambda row: f"{row[mean_column]:.{decimals}f} ± {row[sd_column]:.{decimals}f}",
        axis=1,
    )


def pooled_metrics(
    subjects: list[SubjectPredictions], horizons_min: tuple[int, ...]
) -> pd.DataFrame:
    """Metrics over all windows pooled across subjects -- **secondary** only.

    Provided so the difference from the per-subject aggregate can be shown
    explicitly. Where they diverge substantially, the divergence is itself a
    finding about cohort imbalance and belongs in the paper.
    """
    if not subjects:
        raise MetricError("no subjects supplied")
    y_true = np.concatenate([subject.y_true for subject in subjects], axis=0)
    y_pred = np.concatenate([subject.y_pred for subject in subjects], axis=0)
    records = accuracy_by_horizon(y_true, y_pred, horizons_min)
    frame = pd.DataFrame([record.as_dict() for record in records])
    frame.insert(0, "aggregation", "pooled")
    return frame


__all__ = [
    "SubjectPredictions",
    "across_subject_summary",
    "format_mean_sd",
    "per_subject_table",
    "pooled_metrics",
    "subject_metrics",
]
