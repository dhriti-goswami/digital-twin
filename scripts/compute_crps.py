"""Compute CRPS and q90 coverage per horizon from the three trained quantile heads.

The model is trained on only three quantiles (q10, q50, q90; Eq. 16), so the
continuous ranked probability score (CRPS) cannot be computed exactly -- the
full quantile function is unknown. This script uses the quantile-score
(pinball loss) decomposition of CRPS:

    CRPS(F, y) = 2 * integral_0^1 pinball_tau(y) dtau.

(CORRECTED 2026-10-17: a previous version of this script fit an ad hoc
coefficient set by least squares against a `crps` column whose original
generating code is not in this repository, and the docstring's own stated
derivation did not even algebraically reduce to the formula actually used
-- 2*[0.5*p10 + 0.5*(p10+p50)/2 + 0.5*(p50+p90)/2] simplifies to
1.5*p10 + p50 + 0.5*p90, not the 0.5*p10 + p50 + 0.5*p90 that was coded.
Since the true source of those old numbers is unknown and unreproducible,
they are replaced here with a properly derived trapezoidal-rule estimate.)

Derivation: pinball_tau(y) -> 0 as tau -> 0 and as tau -> 1, because the
optimal quantile at those limits approaches the support boundary and the
expected weighted absolute error vanishes there. Treating (tau, pinball_tau)
as a function sampled at five nodes -- tau = 0, 0.1, 0.5, 0.9, 1, with
pinball = 0, pb10, pb50, pb90, 0 -- gives the trapezoidal-rule integral

    integral_0^1 pinball_tau dtau
      ~= 0.1*(0+pb10)/2 + 0.4*(pb10+pb50)/2 + 0.4*(pb50+pb90)/2 + 0.1*(pb90+0)/2
      =  0.05*pb10 + 0.2*(pb10+pb50) + 0.2*(pb50+pb90) + 0.05*pb90
      =  0.25*pb10 + 0.4*pb50 + 0.25*pb90

so

    CRPS ~= 2 * (0.25*pb10 + 0.4*pb50 + 0.25*pb90)
          =  0.5*pb10 + 0.8*pb50 + 0.5*pb90.

This does NOT exactly reproduce the previously committed crps column (see
the printed diff below) -- that is expected, since the old column's
generating method could not be located or verified. This is an approximation
because the true tail behavior below q10 and above q90, and the true shape
between nodes, is unconstrained by the trained model; it assumes the
pinball-vs-tau curve is well approximated by linear interpolation between
the five nodes above, including the boundary-zero assumption at tau=0/1.

Usage:
    python scripts/compute_crps.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DIAGNOSTICS_PATH = "artifacts/official/quantile/official/test_diagnostics.npz"
CALIBRATION_CSVS = [
    "results/tables/calibration_by_horizon.csv",
    "paper/digital-twin-v5/data/calibration_by_horizon.csv",
]
HORIZONS_MIN = [30, 60, 90, 120]
NOMINAL_INTERVAL_PCT = 80.0


def pinball_loss(quantile: float, pred: np.ndarray, true: np.ndarray) -> float:
    error = true - pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def crps_from_three_quantiles(pb10: float, pb50: float, pb90: float) -> float:
    """CRPS approximation from pinball losses at q=0.1, 0.5, 0.9 (see module docstring).

    Trapezoidal-rule integral of pinball_tau over tau in [0, 1], with
    boundary nodes pinball_0 = pinball_1 = 0:
        CRPS ~= 2 * (0.25*pb10 + 0.4*pb50 + 0.25*pb90)
              =  0.5*pb10 + 0.8*pb50 + 0.5*pb90
    """
    return 0.5 * pb10 + 0.8 * pb50 + 0.5 * pb90


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

    for csv_path in CALIBRATION_CSVS:
        try:
            existing = pd.read_csv(csv_path)
        except FileNotFoundError:
            print(f"(skip, not found: {csv_path})")
            continue
        merged = existing.merge(
            df, on="horizon_min", suffixes=("_existing", "_recomputed")
        )
        print(f"\n--- {csv_path} ---")
        for col in ["crps", "upper_q90_coverage_pct", "pinball_q10", "pinball_q50", "pinball_q90"]:
            diff = (merged[f"{col}_existing"] - merged[f"{col}_recomputed"]).abs().max()
            print(f"max abs diff vs committed CSV, {col}: {diff:.6f}")

        # Overwrite only columns this script owns; keep any other existing columns as-is.
        updated = existing.copy()
        for col in [
            "empirical_interval_pct",
            "lower_q10_coverage_pct",
            "upper_q90_coverage_pct",
            "mean_interval_width",
            "pinball_q10",
            "pinball_q50",
            "pinball_q90",
            "crps",
        ]:
            if col in updated.columns:
                updated = updated.merge(
                    df[["horizon_min", col]].rename(columns={col: f"{col}__new"}),
                    on="horizon_min",
                    how="left",
                )
                updated[col] = updated[f"{col}__new"]
                updated = updated.drop(columns=[f"{col}__new"])
        updated.to_csv(csv_path, index=False)
        print(f"wrote corrected crps column to {csv_path}")

    print("\nNote: nominal 90% upper-quantile coverage is only "
          f"{df['upper_q90_coverage_pct'].min():.1f}-{df['upper_q90_coverage_pct'].max():.1f}% "
          "across horizons -- the paper currently discusses only lower-tail "
          "(hypoglycemia) coverage; this upper-tail under-coverage should be stated too.")


if __name__ == "__main__":
    main()
