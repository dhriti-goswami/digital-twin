"""Clinical glycaemic metrics: time-in-range bands and Kovatchev risk indices.

These answer a different question from RMSE. A model can have good RMSE while
systematically compressing excursions toward the mean, which looks safe on paper
and is useless clinically. Reporting the *predicted* range distribution against
the *actual* one exposes that.

.. warning::

   The legacy ``evaluate.py`` computed TIR/TAR/TBR on predictions only, with no
   actual-versus-predicted comparison, so compression was invisible. Every
   function here returns both or takes an explicit reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from twin.metrics.accuracy import MetricError

Array = NDArray[np.floating]

#: Consensus glucose bands [mg/dL]. Level-2 events are the clinically urgent
#: ones and are reported separately rather than folded into level 1.
#: PROVENANCE PENDING: to be confirmed against the international consensus on
#: CGM-derived metrics (see docs/CITATIONS.md) before any reported table.
VERY_LOW_MAX = 54.0
LOW_MAX = 70.0
HIGH_MIN = 180.0
VERY_HIGH_MIN = 250.0

#: Kovatchev symmetrising risk transform constants.
#: PROVENANCE PENDING verification in docs/CITATIONS.md; these are the widely
#: reproduced values and are unit-tested for the symmetry property they must have.
RISK_SCALE = 1.509
RISK_EXPONENT = 1.084
RISK_OFFSET = 5.381


@dataclass(frozen=True)
class RangeDistribution:
    """Fraction of readings in each consensus band [%]."""

    very_low: float
    low: float
    in_range: float
    high: float
    very_high: float
    n: int

    @property
    def time_below_range(self) -> float:
        """TBR: everything under 70 mg/dL, both levels combined."""
        return self.very_low + self.low

    @property
    def time_above_range(self) -> float:
        """TAR: everything over 180 mg/dL, both levels combined."""
        return self.high + self.very_high

    def as_dict(self) -> dict[str, float]:
        out = asdict(self)
        out["time_below_range"] = self.time_below_range
        out["time_above_range"] = self.time_above_range
        return out


def range_distribution(glucose: Array) -> RangeDistribution:
    """Consensus band occupancy for a glucose series.

    Bands are half-open and exhaustive, so the five fractions sum to exactly 100%
    with no reading counted twice and none dropped.
    """
    values = np.asarray(glucose, dtype=np.float64).ravel()
    if values.size == 0:
        raise MetricError("empty glucose series")
    if not np.isfinite(values).all():
        raise MetricError("glucose series contains non-finite values")

    total = values.size
    very_low = np.count_nonzero(values < VERY_LOW_MAX)
    low = np.count_nonzero((values >= VERY_LOW_MAX) & (values < LOW_MAX))
    in_range = np.count_nonzero((values >= LOW_MAX) & (values <= HIGH_MIN))
    high = np.count_nonzero((values > HIGH_MIN) & (values <= VERY_HIGH_MIN))
    very_high = np.count_nonzero(values > VERY_HIGH_MIN)

    return RangeDistribution(
        very_low=100.0 * very_low / total,
        low=100.0 * low / total,
        in_range=100.0 * in_range / total,
        high=100.0 * high / total,
        very_high=100.0 * very_high / total,
        n=int(total),
    )


@dataclass(frozen=True)
class RangeAgreement:
    """Predicted versus actual band occupancy, with the signed gap."""

    actual: RangeDistribution
    predicted: RangeDistribution

    @property
    def in_range_delta(self) -> float:
        return self.predicted.in_range - self.actual.in_range

    @property
    def below_range_delta(self) -> float:
        return self.predicted.time_below_range - self.actual.time_below_range

    @property
    def above_range_delta(self) -> float:
        return self.predicted.time_above_range - self.actual.time_above_range

    def as_dict(self) -> dict[str, float]:
        return {
            **{f"actual_{k}": v for k, v in self.actual.as_dict().items()},
            **{f"predicted_{k}": v for k, v in self.predicted.as_dict().items()},
            "in_range_delta": self.in_range_delta,
            "below_range_delta": self.below_range_delta,
            "above_range_delta": self.above_range_delta,
        }


def range_agreement(y_true: Array, y_pred: Array) -> RangeAgreement:
    """Band occupancy of reference and prediction side by side.

    A model that under-predicts hypoglycaemia shows up as ``below_range_delta``
    being strongly negative even when RMSE looks acceptable.
    """
    return RangeAgreement(
        actual=range_distribution(y_true), predicted=range_distribution(y_pred)
    )


# --------------------------------------------------------------------------- #
# Kovatchev risk indices
# --------------------------------------------------------------------------- #


def risk_transform(glucose: Array) -> Array:
    """Kovatchev symmetrising transform of the glucose scale.

    ``f(G) = 1.509 * (ln(G)^1.084 - 5.381)`` for ``G`` in mg/dL.

    The glucose scale is asymmetric: euglycaemia sits at ~112 mg/dL, far from the
    midpoint of the clinically meaningful range. This transform maps that range
    symmetrically about zero so hypo- and hyperglycaemic deviations become
    comparable, which is what makes a single risk number meaningful.

    The symmetric endpoints are **20 and 600 mg/dL**, not 40 and 400: numerically
    ``f(20) = -3.1634`` and ``f(600) = +3.1629``, i.e. the transform maps
    ``[20, 600]`` onto ``[-sqrt(10), +sqrt(10)]`` and hence ``r = 10 f^2`` onto
    ``[0, 100]``. Getting this range wrong is an easy way to misread the indices.
    """
    values = np.asarray(glucose, dtype=np.float64)
    if np.any(values <= 0):
        raise MetricError("risk transform requires strictly positive glucose")
    return RISK_SCALE * (np.log(values) ** RISK_EXPONENT - RISK_OFFSET)


def _risk(glucose: Array) -> Array:
    return 10.0 * risk_transform(glucose) ** 2


def lbgi(glucose: Array) -> float:
    """Low blood glucose index.

    Mean of the risk function over the *whole* series, counting only readings on
    the hypoglycaemic side. Dividing by the full ``n`` rather than the count of
    low readings is what makes LBGI comparable between subjects who spend
    different amounts of time low.
    """
    values = np.asarray(glucose, dtype=np.float64).ravel()
    transformed = risk_transform(values)
    contributions = np.where(transformed < 0, 10.0 * transformed**2, 0.0)
    return float(np.mean(contributions))


def hbgi(glucose: Array) -> float:
    """High blood glucose index. Mirror of :func:`lbgi`."""
    values = np.asarray(glucose, dtype=np.float64).ravel()
    transformed = risk_transform(values)
    contributions = np.where(transformed > 0, 10.0 * transformed**2, 0.0)
    return float(np.mean(contributions))


@dataclass(frozen=True)
class RiskIndices:
    lbgi: float
    hbgi: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def risk_indices(glucose: Array) -> RiskIndices:
    return RiskIndices(lbgi=lbgi(glucose), hbgi=hbgi(glucose))


# --------------------------------------------------------------------------- #
# Variability
# --------------------------------------------------------------------------- #


def coefficient_of_variation(glucose: Array) -> float:
    """CV [%] -- the consensus variability measure.

    Included because a model that flattens its output toward the mean gets good
    RMSE and a visibly wrong CV. Comparing predicted CV to actual CV is the
    cheapest available test for excursion compression.
    """
    values = np.asarray(glucose, dtype=np.float64).ravel()
    mean = values.mean()
    if mean <= 0:
        raise MetricError("CV requires positive mean glucose")
    return float(100.0 * values.std(ddof=1) / mean)


def hypoglycaemia_detection(
    y_true: Array, y_pred: Array, *, threshold: float = LOW_MAX
) -> dict[str, float]:
    """Event-level detection performance for hypoglycaemia.

    Sensitivity here is the number that matters clinically and is invisible in
    RMSE: missing 40% of hypoglycaemic events is compatible with a respectable
    error metric when such events are rare.

    Reported instead of building an asymmetric penalty into the training loss --
    measuring the property is honest, optimising the metric into the objective
    inflates the error grid by construction.
    """
    reference = np.asarray(y_true, dtype=np.float64).ravel()
    prediction = np.asarray(y_pred, dtype=np.float64).ravel()
    if reference.shape != prediction.shape:
        raise MetricError("shape mismatch")

    actual = reference < threshold
    predicted = prediction < threshold

    true_positive = int(np.count_nonzero(actual & predicted))
    false_negative = int(np.count_nonzero(actual & ~predicted))
    false_positive = int(np.count_nonzero(~actual & predicted))
    true_negative = int(np.count_nonzero(~actual & ~predicted))

    def ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    return {
        "threshold": float(threshold),
        "n_actual_events": true_positive + false_negative,
        "sensitivity": ratio(true_positive, true_positive + false_negative),
        "specificity": ratio(true_negative, true_negative + false_positive),
        "precision": ratio(true_positive, true_positive + false_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
    }


__all__ = [
    "HIGH_MIN",
    "LOW_MAX",
    "RISK_EXPONENT",
    "RISK_OFFSET",
    "RISK_SCALE",
    "VERY_HIGH_MIN",
    "VERY_LOW_MAX",
    "RangeAgreement",
    "RangeDistribution",
    "RiskIndices",
    "coefficient_of_variation",
    "hbgi",
    "hypoglycaemia_detection",
    "lbgi",
    "range_agreement",
    "range_distribution",
    "risk_indices",
    "risk_transform",
]
