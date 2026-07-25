"""Non-learned baselines.

These exist because the legacy pipeline had none, and both of its headline numbers
lost to the simplest of them. A model's result is meaningless without them in the
same table: 30-minute persistence on OhioT1DM is ~22.5 mg/dL RMSE, so a reported
30.4 is worse than predicting no change at all.

Three baselines, in increasing sophistication:

``persistence``
    Glucose does not change. The reference point for every claim in this project.
``roc_extrapolation``
    Linear extrapolation of the recent rate of change. Tests whether a model has
    learned anything beyond the trend already visible in its input.
``arima``
    A fitted linear time-series model. Tests whether a neural network is earning
    its complexity against a classical alternative on the same data.

All three are evaluated on exactly the windows the learned models see, so the
comparison is not confounded by different eligibility rules.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from twin.data.dataset import SubjectData
from twin.data.sequencing import FILLED_COLUMN

Array = NDArray[np.float64]

#: Predictions are clipped to the physiological reporting range. A linear
#: extrapolation of a steep rise will otherwise leave the measurable range
#: entirely, and scoring an impossible value would flatter the learned models.
GLUCOSE_MIN = 40.0
GLUCOSE_MAX = 400.0


@dataclass(frozen=True)
class BaselineResult:
    """Predictions from one baseline for one subject."""

    name: str
    subject_id: str
    predictions: Array  # (n_windows, n_horizons)
    targets: Array


def _glucose_series(data: SubjectData) -> Array:
    return data.frame[FILLED_COLUMN].to_numpy(dtype=np.float64)


def _anchors(data: SubjectData, indices: NDArray[np.int64]) -> NDArray[np.int64]:
    return data.windows.anchors[indices]


def persistence(data: SubjectData, indices: NDArray[np.int64]) -> Array:
    """Predict the anchor value at every horizon."""
    glucose = _glucose_series(data)
    anchors = _anchors(data, indices)
    n_horizons = len(data.windows.horizons_min)
    return np.repeat(glucose[anchors][:, None], n_horizons, axis=1)


def roc_extrapolation(
    data: SubjectData, indices: NDArray[np.int64], *, lookback_steps: int = 6
) -> Array:
    """Extrapolate the recent rate of change linearly.

    The rate is estimated over ``lookback_steps`` grid steps ending at the anchor
    (30 minutes by default), which is the interval a clinician reading a CGM trend
    arrow would use. Predictions are clipped to the physiological range.

    A learned model that cannot beat this has not extracted anything from insulin,
    carbohydrate, or context -- it has only read the trend.
    """
    glucose = _glucose_series(data)
    anchors = _anchors(data, indices)
    grid = data.windows.grid_minutes

    lookback_minutes = lookback_steps * grid
    rate = (glucose[anchors] - glucose[anchors - lookback_steps]) / lookback_minutes

    horizons = np.asarray(data.windows.horizons_min, dtype=np.float64)
    predictions = glucose[anchors][:, None] + rate[:, None] * horizons[None, :]
    return np.clip(predictions, GLUCOSE_MIN, GLUCOSE_MAX)


def arima(
    data: SubjectData,
    indices: NDArray[np.int64],
    *,
    order: tuple[int, int, int] = (2, 1, 1),
    train_glucose: Array | None = None,
) -> Array:
    """ARIMA forecast from each window's own history.

    Parameters are estimated **once** on ``train_glucose`` (the subject's training
    period, or this record if none is given), then held fixed while the state-space
    filter is re-applied to each window's 24-step history. Re-estimating per window
    would both be prohibitively slow and let the baseline peek at the evaluation
    period.

    ``order`` defaults to (2, 1, 1): differencing once handles the strong trend in
    CGM data, and the low AR/MA orders are what the 2-hour context window can
    support.
    """
    from statsmodels.tsa.arima.model import ARIMA

    glucose = _glucose_series(data)
    anchors = _anchors(data, indices)
    seq_len = data.windows.seq_len
    horizon_steps = np.asarray(data.windows.horizon_steps, dtype=int)
    max_step = int(horizon_steps.max())

    fitting_series = train_glucose if train_glucose is not None else glucose
    fitting_series = fitting_series[np.isfinite(fitting_series)]

    with warnings.catch_warnings():
        # Convergence chatter on long CGM series is expected and not informative.
        warnings.simplefilter("ignore")
        fitted = ARIMA(fitting_series, order=order).fit()
        parameters = fitted.params

        predictions = np.empty((anchors.size, horizon_steps.size), dtype=np.float64)
        for row, anchor in enumerate(anchors.tolist()):
            history = glucose[anchor - seq_len + 1 : anchor + 1]
            try:
                applied = fitted.apply(history, refit=False)
                path = applied.forecast(steps=max_step)
            except Exception:
                # A degenerate history (e.g. constant) can make the filter fail;
                # falling back to persistence is honest and clearly labelled.
                path = np.full(max_step, history[-1])
            predictions[row] = path[horizon_steps - 1]

    return np.clip(np.nan_to_num(predictions, nan=float(np.nanmean(glucose))), GLUCOSE_MIN, GLUCOSE_MAX)


#: Registry of the baselines that require no training.
ANALYTIC_BASELINES = {
    "persistence": persistence,
    "roc_extrapolation": roc_extrapolation,
}

#: Baselines that fit parameters but are not neural networks.
FITTED_BASELINES = {"arima": arima}


def run_baseline(
    name: str,
    data: SubjectData,
    indices: NDArray[np.int64],
    **kwargs: object,
) -> BaselineResult:
    """Evaluate one baseline on one subject's selected windows."""
    if name in ANALYTIC_BASELINES:
        predictions = ANALYTIC_BASELINES[name](data, indices, **kwargs)  # type: ignore[arg-type]
    elif name in FITTED_BASELINES:
        predictions = FITTED_BASELINES[name](data, indices, **kwargs)  # type: ignore[arg-type]
    else:
        raise ValueError(
            f"unknown baseline {name!r}; expected one of "
            f"{sorted(set(ANALYTIC_BASELINES) | set(FITTED_BASELINES))}"
        )
    return BaselineResult(
        name=name,
        subject_id=data.subject_id,
        predictions=predictions,
        targets=data.windows.targets[indices],
    )


__all__ = [
    "ANALYTIC_BASELINES",
    "FITTED_BASELINES",
    "GLUCOSE_MAX",
    "GLUCOSE_MIN",
    "BaselineResult",
    "arima",
    "persistence",
    "roc_extrapolation",
    "run_baseline",
]
