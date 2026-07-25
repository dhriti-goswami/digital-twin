"""Point-accuracy metrics, per horizon.

Conventions fixed here and used everywhere downstream:

* Arrays are ``(n_samples, n_horizons)``; horizon is always the last axis.
* ``y_true`` is the *reference* and comes first in every signature. The legacy
  code had ``mard(a, b)`` taking the reference first while ``rmse``/``mae`` were
  symmetric, which is a trap that survives only until someone reorders a call.
* Metrics are computed **per subject** and aggregated across subjects afterwards
  (see :mod:`twin.metrics.report`). Pooling first lets a subject with 2137
  windows outweigh one with 356 -- the legacy tables pooled, and per-subject
  mean +/- SD is the standard OhioT1DM reporting format.
* Nothing here silently drops NaN. A NaN means the sequencing layer emitted a
  window it should not have, and it must surface as an error rather than a
  quietly smaller denominator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.floating]


class MetricError(ValueError):
    """Raised for malformed inputs -- shape mismatch, NaN, or empty arrays."""


def _validate(y_true: Array, y_pred: Array) -> tuple[Array, Array]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise MetricError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise MetricError("empty input")
    if not np.isfinite(y_true).all():
        raise MetricError("y_true contains non-finite values")
    if not np.isfinite(y_pred).all():
        raise MetricError("y_pred contains non-finite values")
    if y_true.ndim == 1:
        y_true = y_true[:, None]
        y_pred = y_pred[:, None]
    return y_true, y_pred


@dataclass(frozen=True)
class HorizonAccuracy:
    """Accuracy at one forecast horizon."""

    horizon_min: int
    n: int
    rmse: float
    mae: float
    r2: float
    mard: float
    bias: float
    #: 95th percentile of absolute error -- a tail statistic that RMSE hides.
    p95_abs_error: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def rmse(y_true: Array, y_pred: Array) -> Array:
    """Root mean squared error per horizon [mg/dL]."""
    y_true, y_pred = _validate(y_true, y_pred)
    return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))


def mae(y_true: Array, y_pred: Array) -> Array:
    """Mean absolute error per horizon [mg/dL]."""
    y_true, y_pred = _validate(y_true, y_pred)
    return np.mean(np.abs(y_pred - y_true), axis=0)


def bias(y_true: Array, y_pred: Array) -> Array:
    """Mean signed error per horizon [mg/dL]. Positive means over-prediction."""
    y_true, y_pred = _validate(y_true, y_pred)
    return np.mean(y_pred - y_true, axis=0)


def r2(y_true: Array, y_pred: Array) -> Array:
    """Coefficient of determination per horizon.

    ``1 - SSE / SST`` with ``SST`` about the mean of ``y_true``. Note this can go
    negative, and does: a model worse than predicting the subject's mean scores
    below zero. The legacy zero-shot evaluation reported R2 of -2.4 at 120 min,
    which is a correct and highly informative number.
    """
    y_true, y_pred = _validate(y_true, y_pred)
    sse = np.sum((y_true - y_pred) ** 2, axis=0)
    sst = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 - sse / sst
    # A constant reference series has no variance to explain; R2 is undefined.
    return np.where(sst > 0, out, np.nan)


def mard(y_true: Array, y_pred: Array) -> Array:
    """Mean absolute relative difference per horizon [%].

    The *reference* is the denominator, which is the standard definition and the
    only one that makes MARD comparable to sensor-accuracy literature.
    """
    y_true, y_pred = _validate(y_true, y_pred)
    if np.any(y_true <= 0):
        raise MetricError("MARD requires strictly positive reference glucose")
    return 100.0 * np.mean(np.abs(y_pred - y_true) / y_true, axis=0)


def p95_absolute_error(y_true: Array, y_pred: Array) -> Array:
    """95th percentile of absolute error per horizon [mg/dL]."""
    y_true, y_pred = _validate(y_true, y_pred)
    return np.percentile(np.abs(y_pred - y_true), 95, axis=0)


def accuracy_by_horizon(
    y_true: Array, y_pred: Array, horizons_min: tuple[int, ...]
) -> list[HorizonAccuracy]:
    """All point-accuracy metrics, one record per horizon."""
    y_true, y_pred = _validate(y_true, y_pred)
    if y_true.shape[1] != len(horizons_min):
        raise MetricError(
            f"{y_true.shape[1]} horizon columns but {len(horizons_min)} horizon labels"
        )

    values = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "mard": mard(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "p95": p95_absolute_error(y_true, y_pred),
    }
    return [
        HorizonAccuracy(
            horizon_min=horizon,
            n=int(y_true.shape[0]),
            rmse=float(values["rmse"][index]),
            mae=float(values["mae"][index]),
            r2=float(values["r2"][index]),
            mard=float(values["mard"][index]),
            bias=float(values["bias"][index]),
            p95_abs_error=float(values["p95"][index]),
        )
        for index, horizon in enumerate(horizons_min)
    ]


# --------------------------------------------------------------------------- #
# Prediction lag
# --------------------------------------------------------------------------- #


def prediction_lag_min(
    y_true_series: Array,
    y_pred_series: Array,
    *,
    sample_minutes: int = 5,
    max_lag_steps: int = 24,
) -> float:
    """Delay [min] at which the prediction best matches the reference.

    A forecaster that has learned little beyond persistence produces a trace that
    looks like the reference shifted forward in time. Cross-correlating the two
    and reporting the best-matching shift exposes that directly: a model whose
    optimal alignment is at the full horizon is reproducing the input, not
    anticipating the future.

    Reported alongside RMSE because two models with identical RMSE can differ
    entirely on this, and it is the difference that matters clinically.

    Returns a positive value when the prediction *lags* the reference.
    """
    reference = np.asarray(y_true_series, dtype=np.float64).ravel()
    prediction = np.asarray(y_pred_series, dtype=np.float64).ravel()
    if reference.shape != prediction.shape:
        raise MetricError("series must have equal length")
    if reference.size < 2 * max_lag_steps + 1:
        raise MetricError("series too short for the requested maximum lag")

    reference = reference - reference.mean()
    prediction = prediction - prediction.mean()

    best_lag, best_score = 0, -np.inf
    for lag in range(-max_lag_steps, max_lag_steps + 1):
        if lag < 0:
            a, b = reference[-lag:], prediction[: len(prediction) + lag]
        elif lag > 0:
            a, b = reference[: len(reference) - lag], prediction[lag:]
        else:
            a, b = reference, prediction
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator == 0:
            continue
        score = float(np.dot(a, b) / denominator)
        if score > best_score:
            best_score, best_lag = score, lag
    # ``best_lag > 0`` means the prediction had to be advanced to align with the
    # reference, i.e. the prediction lags. Reported positive by convention.
    return float(best_lag * sample_minutes)


__all__ = [
    "HorizonAccuracy",
    "MetricError",
    "accuracy_by_horizon",
    "bias",
    "mae",
    "mard",
    "p95_absolute_error",
    "prediction_lag_min",
    "r2",
    "rmse",
]
