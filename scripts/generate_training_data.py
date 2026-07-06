#!/usr/bin/env python3
"""
Generate synthetic T1D training data using the UVA/Padova ODE simulator.

Simulates all 30 virtual patients (10 adolescents, 10 adults, 10 children)
for N days each, saving one CSV per patient to data/raw/simulated/.

Usage:
    python scripts/generate_training_data.py
    python scripts/generate_training_data.py --days 14 --patients 10
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.t1d_ode import T1DPatient, list_patient_names, simulate_patient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Generate ODE-based T1D training data")
    p.add_argument("--days", type=int, default=14, help="Days to simulate per patient")
    p.add_argument("--patients", type=int, default=30, help="Number of patients (max 30)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="data/raw/simulated")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list_patient_names()[: args.patients]
    master_rng = np.random.RandomState(args.seed)

    logger.info("Simulating %d patients × %d days", len(names), args.days)
    logger.info("Output → %s", out_dir)
    print()

    for i, name in enumerate(names):
        seed_i = int(master_rng.randint(0, 9999))
        rng = np.random.RandomState(seed_i)

        logger.info("[%2d/%2d] %s …", i + 1, len(names), name)
        try:
            patient = T1DPatient.from_name(name)
            df = simulate_patient(patient, n_days=args.days, rng=rng)
            df["patient_name"] = name
            df["patient_idx"] = i

            fname = out_dir / f"{name.replace('#', '_')}.csv"
            df.to_csv(fname, index=False)
            cgm_min = df["cgm_mg_dl"].min()
            cgm_max = df["cgm_mg_dl"].max()
            cgm_mean = df["cgm_mg_dl"].mean()
            tir = ((df["cgm_mg_dl"] >= 70) & (df["cgm_mg_dl"] <= 180)).mean() * 100
            logger.info(
                "    CGM: %.0f–%.0f mg/dL  mean=%.0f  TIR=%.1f%%  rows=%d",
                cgm_min, cgm_max, cgm_mean, tir, len(df),
            )
        except Exception as e:
            logger.error("    FAILED: %s", e)

    print()
    logger.info("Done — %d patient files in %s", len(names), out_dir)


if __name__ == "__main__":
    main()
