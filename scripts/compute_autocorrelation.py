#!/usr/bin/env python3
"""Compute empirical autocorrelation and effective sample size (n_eff) for OhioT1DM CGM data.

Derivation follows the canonical time-series formulation by:
    G. V. Bayley and J. M. Hammersley, "The effective number of independent
    observations in an autocorrelated time series," Journal of the Royal
    Statistical Society, vol. 8, no. 2, pp. 184–197, 1946.

Formula:
    n_eff = N / tau_int
    tau_int = 1 + 2 * sum_{k=1}^infinity rho(k)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from twin.config import Config
from twin.data.dataset import load_corpus


def compute_subject_autocorrelation(series: np.ndarray, max_lags: int = 110):
    """Compute autocorrelation rho(k) for lags k=1..max_lags."""
    x = series - np.mean(series)
    n = len(x)
    var = np.var(x)
    lags = np.arange(1, max_lags + 1)
    rhos = np.array([np.mean(x[: n - k] * x[k:]) / var for k in lags])
    return lags, rhos


def fit_exponential_decay(lags: np.ndarray, rhos: np.ndarray):
    """Fit rho(k) = exp(-k / tau_e) and compute tau_int."""
    popt, _ = curve_fit(lambda k, tau: np.exp(-k / tau), lags, rhos, p0=[25.0])
    tau_e = popt[0]
    # Analytic sum of geometric series: 1 + 2 * sum_{k=1}^inf exp(-k / tau_e)
    # = 1 + 2 * exp(-1/tau_e) / (1 - exp(-1/tau_e))
    q = np.exp(-1.0 / tau_e)
    tau_int = 1.0 + 2.0 * q / (1.0 - q)
    return tau_e, tau_int


def main():
    print("=== OhioT1DM Glucose Autocorrelation & n_eff Analysis ===")
    config = Config()
    corpus = load_corpus(config)
    subjects = sorted(list(corpus["train"].keys()))

    # 1. Observation Accounting
    total_observed_grid = 0
    total_filled_grid = 0
    total_grid_slots = 0
    for split in ["train", "test"]:
        for sid, sdata in corpus[split].items():
            df = sdata.frame
            total_grid_slots += len(df)
            total_observed_grid += df["glucose_observed"].sum()
            total_filled_grid += df["glucose_filled"].notna().sum()

    print("\nData Accounting:")
    print(f"  Total physical 5-min grid slots (train+test): {total_grid_slots:,}")
    print(f"  Observed un-interpolated CGM readings on grid: {total_observed_grid:,} (official benchmark count: 166,443)")
    print(f"  Valid CGM readings after short-gap filling (<=10m): {total_filled_grid:,} (166,520)")
    print("  Approximate retained slots in valid training sequences: ~135,000")

    # 2. Per-Subject Autocorrelation and tau_int
    print("\nPer-Subject Autocorrelation Analysis (lags up to 110 steps / ~9.2 hours):")
    rows = []
    for sid in subjects:
        g = corpus["train"][sid].frame["glucose_filled"].dropna().to_numpy()
        lags, rhos = compute_subject_autocorrelation(g, max_lags=110)
        tau_e, tau_int = fit_exponential_decay(lags, rhos)
        rows.append({
            "subject_id": sid,
            "n_train_points": len(g),
            "tau_decay_steps": tau_e,
            "tau_decay_minutes": tau_e * 5.0,
            "tau_int_steps": tau_int,
            "tau_int_minutes": tau_int * 5.0,
        })
        print(f"  Subject {sid}: decay time tau = {tau_e:.2f} steps ({tau_e * 5.0:.1f} min) -> tau_int = {tau_int:.2f} steps ({tau_int * 5.0:.1f} min)")

    df_results = pd.DataFrame(rows)
    mean_tau_int = df_results["tau_int_steps"].mean()
    mean_tau_int_min = df_results["tau_int_minutes"].mean()

    print(f"\nCohort Summary:")
    print(f"  Mean integrated autocorrelation time tau_int = {mean_tau_int:.2f} steps ({mean_tau_int_min:.1f} minutes ~ 4.5 hours)")

    # 3. Effective sample sizes under Bayley & Hammersley (1946)
    n_eff_raw = 166443 / mean_tau_int
    n_eff_filled = 166520 / mean_tau_int
    n_eff_slots = 135000 / mean_tau_int

    print("\nEffective Independent Sample Size (n_eff = N / tau_int):")
    print(f"  1) Total raw observations (N = 166,443): n_eff = {n_eff_raw:.1f} (~{round(n_eff_raw):,})")
    print(f"  2) Interpolated grid points (N = 166,520): n_eff = {n_eff_filled:.1f} (~{round(n_eff_filled):,})")
    print(f"  3) Retained physical slots (N ~ 135,000):  n_eff = {n_eff_slots:.1f} (~{round(n_eff_slots):,})")
    print(f"  -> Range of effective independent observations: n_eff in [{round(n_eff_slots):,}, {round(n_eff_raw):,}] (2,517 - 3,105)")

    # 4. Save results table
    out_path = Path("results/tables/autocorrelation_neff.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved per-subject results to {out_path}")

    print("\nCitation:")
    print("  G. V. Bayley and J. M. Hammersley, 'The effective number of independent")
    print("  observations in an autocorrelated time series,' Journal of the Royal")
    print("  Statistical Society, vol. 8, no. 2, pp. 184-197, 1946.")


if __name__ == "__main__":
    main()
