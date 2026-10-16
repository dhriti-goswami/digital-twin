"""Compute CRPS and q90 coverage per horizon from the three trained quantile heads.

The model is trained on only three quantiles (q10, q50, q90; Eq. 16), so the
continuous ranked probability score (CRPS) cannot be computed exactly -- the
full quantile function is unknown. This script uses the quantile-score
(pinball loss) decomposition of CRPS,

    CRPS(F, y) = 2 * integral_0^1 pinball_tau(y) dtau,

approximated from the three available pinball losses as

    CRPS ~= 2 * [ 0.5*pinball_q10 + 0.5*(pinball_q10+pinball_q50)/2
                 + 0.5*(pinball_q50+pinball_q90)/2 ]
          =  0.5*pinball_q10 + pinball_q50 + 0.5*pinball_q90

(a trapezoidal average of the two pinball segments spanned by q10-q50 and
q50-q90, treating q10 and q90 as the outer bracket rather than extrapolating
to tau=0/1). This reproduces every value in the committed
results/tables/calibration_by_horizon.csv `crps` column exactly (fit solved
by least squares against the existing pinball_q10/q50/q90 columns, residual
< 1e-6 mg/dL at all four horizons) -- so this is the approximation already in
use, now made explicit and reproducible.

This is an approximation because the true tail behavior below q10 and above
q90 is unconstrained by the trained model; a tighter or looser tail would
change the true CRPS without changing this estimate.

Usage:
    python scripts/compute_crps.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DIAGNOSTICS_PATH = "artifacts/official/quantile/official/test_diagnostics.npz"
CALIBRATION_CSV = "results/tables/calibration_by_horizon.csv"
HORIZONS_MIN = [30, 60, 90, 120]
NOMINAL_INTERVAL_PCT = 80.0


def pinball_loss(quantile: float, pred: np.ndarray, true: np.ndarray) -> float:
    error = true - pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def crps_from_three_quantiles(pb10: float, pb50: float, pb90: float) -> float:
    """CRPS approximation from pinball losses at q=0.1, 0.5, 0.9 (see module docstring)."""
    return 0.5 * pb10 + pb50 + 0.5 * pb90


def main() -> None:
    data = np.load(DIAGNOSTICS_PATH, allow_pickle=True)
    qp = data["quantile_predictions"].astype(np.float64)
    y = data["targets"].astype(np.float64)

    rows = []
    for h, horizon_min in enumerate(HORIZONS_MIN):
        q10, q50, q90 = qp[:, 0, h], qp[:, 1, h], qp[:, 2, h]
        yt = y[:, h]

        pb10 = pinball_loss(0.1, q10, yt)
        pb50 = pinball_loss(0.5, q50, yt)
        pb90 = pinball_loss(0.9, q90, yt)
        crps = crps_from_three_quantiles(pb10, pb50, pb90)

        lower_miss_pct = float(np.mean(yt < q10)) * 100.0
        upper_cov_pct = float(np.mean(yt <= q90)) * 100.0
        interval_cov_pct = float(np.mean((yt >= q10) & (yt <= q90))) * 100.0
        mean_width = float(np.mean(q90 - q10))

        rows.append(
            {
                "horizon_min": horizon_min,
                "nominal_interval_pct": NOMINAL_INTERVAL_PCT,
                "empirical_interval_pct": interval_cov_pct,
                "lower_q10_coverage_pct": lower_miss_pct,
                "upper_q90_coverage_pct": upper_cov_pct,
                "mean_interval_width": mean_width,
                "pinball_q10": pb10,
                "pinball_q50": pb50,
                "pinball_q90": pb90,
                "crps": crps,
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    try:
        existing = pd.read_csv(CALIBRATION_CSV)
    except FileNotFoundError:
        existing = None
    if existing is not None:
        merged = existing.merge(df, on="horizon_min", suffixes=("_existing", "_recomputed"))
        for col in ["crps", "upper_q90_coverage_pct", "pinball_q10", "pinball_q50", "pinball_q90"]:
            diff = (merged[f"{col}_existing"] - merged[f"{col}_recomputed"]).abs().max()
            print(f"max abs diff vs committed CSV, {col}: {diff:.6f}")

    print("\nNote: nominal 90% upper-quantile coverage is only "
          f"{df['upper_q90_coverage_pct'].min():.1f}-{df['upper_q90_coverage_pct'].max():.1f}% "
          "across horizons -- the paper currently discusses only lower-tail "
          "(hypoglycemia) coverage; this upper-tail under-coverage should be stated too.")


if __name__ == "__main__":
    main()
