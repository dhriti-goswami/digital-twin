"""Metrics, clinical indices, error grids, and statistical reporting.

Design rules enforced across this subpackage:

* **Reference first.** Every signature is ``(y_true, y_pred)``.
* **Per subject, then across subjects.** ``mean +/- SD`` across subjects is the
  headline; pooled numbers are secondary and labelled as such.
* **Nothing silently dropped.** A NaN or a non-finite value raises, because it
  means the sequencing layer emitted a window it should not have.
* **Verification gates.** Error-grid boundaries cannot be used for reportable
  output until they have been checked against the primary source.
"""

from twin.metrics.accuracy import (
    HorizonAccuracy,
    MetricError,
    accuracy_by_horizon,
    bias,
    mae,
    mard,
    p95_absolute_error,
    prediction_lag_min,
    r2,
    rmse,
)
from twin.metrics.clinical import (
    RangeAgreement,
    RangeDistribution,
    RiskIndices,
    coefficient_of_variation,
    hbgi,
    hypoglycaemia_detection,
    lbgi,
    range_agreement,
    range_distribution,
    risk_indices,
    risk_transform,
)
from twin.metrics.errorgrid import (
    CLARKE_ZONES,
    PARKES_ZONES,
    VERIFICATION_STATUS,
    UnverifiedBoundaryError,
    ZoneSummary,
    assert_verified,
    clarke_zone,
    parkes_zone,
    zone_field,
    zone_summary,
)
from twin.metrics.report import (
    SubjectPredictions,
    across_subject_summary,
    format_mean_sd,
    per_subject_table,
    pooled_metrics,
    subject_metrics,
)
from twin.metrics.stats import (
    BootstrapCI,
    PairedComparison,
    bootstrap_ci,
    describe_comparison,
    holm_bonferroni,
    paired_comparison,
)

__all__ = [
    "CLARKE_ZONES",
    "PARKES_ZONES",
    "VERIFICATION_STATUS",
    "BootstrapCI",
    "HorizonAccuracy",
    "MetricError",
    "PairedComparison",
    "RangeAgreement",
    "RangeDistribution",
    "RiskIndices",
    "SubjectPredictions",
    "UnverifiedBoundaryError",
    "ZoneSummary",
    "accuracy_by_horizon",
    "across_subject_summary",
    "assert_verified",
    "bias",
    "bootstrap_ci",
    "clarke_zone",
    "coefficient_of_variation",
    "describe_comparison",
    "format_mean_sd",
    "hbgi",
    "holm_bonferroni",
    "hypoglycaemia_detection",
    "lbgi",
    "mae",
    "mard",
    "p95_absolute_error",
    "paired_comparison",
    "parkes_zone",
    "per_subject_table",
    "pooled_metrics",
    "prediction_lag_min",
    "r2",
    "range_agreement",
    "range_distribution",
    "risk_indices",
    "risk_transform",
    "rmse",
    "subject_metrics",
    "zone_field",
    "zone_summary",
]
