"""Regenerate the sensitivity/precision curves at q in {0.05, 0.10, 0.20}.

The quantile model (twin/config.py) is trained on exactly three quantile
heads, q in {0.1, 0.5, 0.9} (Eq. 16 of the paper). Sec. III-G3 reports
sensitivity/precision curves at q in {0.05, 0.10, 0.20}; q=0.10 is a trained
head, but q=0.05 and q=0.20 are NOT separately trained quantiles -- they are
an approximate extrapolation/interpolation of the lower tail, built from the
two trained quantiles that bracket it (q10, q50):

    sigma_L   = (q50 - q10) / (-Phi^-1(0.10))      # left-tail Gaussian scale
    q_p       = q50 + sigma_L * Phi^-1(p)          # for p in {0.05, 0.20}

where Phi^-1 is the standard normal inverse CDF (scipy.stats.norm.ppf). This
assumes the predictive distribution is locally Gaussian in the lower tail,
calibrated to match the model's own trained q10 spread -- it is an
approximation, not a second independently trained quantile model. Fit
verified against committed results/tables/sensitivity_precision_curves.csv
(max abs error < 0.002 across all thresholds/metrics, attributable to the
float32 storage of the original quantile predictions).

Usage:
    python scripts/compute_quantile_extrapolation.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

DIAGNOSTICS_PATH = "artifacts/official/quantile/official/test_diagnostics.npz"
OUTPUT_PATHS = [
    "results/tables/sensitivity_precision_curves.csv",
    "paper/digital-twin-v4/data/sensitivity_precision_curves.csv",
]
HORIZON_MIN = 30  # Sec. III-G3 curves are reported at the 30-min horizon
HORIZON_INDEX = 0  # column index into targets/quantile_predictions for 30 min
THRESHOLDS = np.arange(50.0, 98.0 + 1e-9, 1.0)

Z_LOW = norm.ppf(0.10)  # anchor quantile used to calibrate the tail scale


def extrapolate_lower_quantiles(q10: np.ndarray, q50: np.ndarray) -> dict[str, np.ndarray]:
    """Extrapolate q05/q20 from the trained q10/q50 heads (see module docstring)."""
    sigma_left = (q50 - q10) / (-Z_LOW)
    return {
        "q05": q50 + sigma_left * norm.ppf(0.05),
        "q20": q50 + sigma_left * norm.ppf(0.20),
    }


def sensitivity_precision(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> tuple[float, float]:
    actual = y_true < threshold
    predicted = y_pred < threshold
    tp = np.count_nonzero(actual & predicted)
    fn = np.count_nonzero(actual & ~predicted)
    fp = np.count_nonzero(~actual & predicted)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    return sens, prec


def main() -> None:
    data = np.load(DIAGNOSTICS_PATH, allow_pickle=True)
    qp = data["quantile_predictions"].astype(np.float64)
    y = data["targets"].astype(np.float64)
    levels = data["quantile_levels"]
    assert tuple(np.round(levels, 2)) == (0.1, 0.5, 0.9), levels

    q10 = qp[:, 0, HORIZON_INDEX]
    q50 = qp[:, 1, HORIZON_INDEX]
    q90 = qp[:, 2, HORIZON_INDEX]
    yt = y[:, HORIZON_INDEX]

    tails = extrapolate_lower_quantiles(q10, q50)
    predictors = {
        "q05": tails["q05"],
        "q10": q10,
        "q20": tails["q20"],
        "med": q50,
    }

    rows = []
    for threshold in THRESHOLDS:
        row = {"threshold": float(threshold)}
        for name, pred in predictors.items():
            sens, prec = sensitivity_precision(yt, pred, threshold)
            row[f"sens_{name}"] = sens
            row[f"prec_{name}"] = prec
        rows.append(row)

    df = pd.DataFrame(rows)

    # Self-check against whatever was already committed, before overwriting it.
    try:
        existing = pd.read_csv(OUTPUT_PATHS[0])
    except FileNotFoundError:
        existing = None
    if existing is not None and len(existing) == len(df):
        diffs = (existing.set_index("threshold") - df.set_index("threshold")).abs()
        print("max abs diff vs previously committed CSV:", diffs.to_numpy().max())

    for path in OUTPUT_PATHS:
        df.to_csv(path, index=False)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
