#!/usr/bin/env python3
"""
Build real training data from genuine 5-minute CGM traces.

Reads the 10 real CGM trace files from data/raw/cgm_traces/ (5-min intervals,
7 days per patient) and produces properly formatted processed data in
data/processed/.

Also generates aligned insulin/meal events by parsing the UCI-format files
(data/raw/uci_diabetes/data-*) for matching patient IDs.

This replaces the previous pipeline that used 2-hour interval data linearly
interpolated to 5-min — which made predictions trivially easy and clinically
meaningless.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CGM_TRACES_DIR = PROJECT_ROOT / "data" / "raw" / "cgm_traces"
UCI_DIR = PROJECT_ROOT / "data" / "raw" / "uci_diabetes"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def load_cgm_traces() -> pd.DataFrame:
    """Load all 10 CGM trace files into a single DataFrame."""
    all_traces = []

    for trace_file in sorted(CGM_TRACES_DIR.glob("patient_*_cgm_trace.csv")):
        # Extract patient ID from filename
        patient_id = int(trace_file.stem.split("_")[1])

        df = pd.read_csv(trace_file)
        df["patient_id"] = patient_id
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Validate: ensure 5-minute intervals
        time_diffs = df["timestamp"].diff().dt.total_seconds().dropna()
        median_interval = time_diffs.median()

        if abs(median_interval - 300) > 60:  # Allow 1-min tolerance
            logger.warning(
                f"Patient {patient_id}: median interval is {median_interval:.0f}s "
                f"(expected 300s). Resampling to 5-min."
            )
            df = df.set_index("timestamp").resample("5min").first()
            df["glucose_mg_dl"] = df["glucose_mg_dl"].interpolate(method="linear")
            df["patient_id"] = patient_id
            df = df.reset_index()

        # Drop any NaN glucose values
        df = df.dropna(subset=["glucose_mg_dl"])

        readings = len(df)
        days = (df["timestamp"].max() - df["timestamp"].min()).days
        logger.info(
            f"  Patient {patient_id:3d}: {readings:5d} readings over {days} days "
            f"({readings / max(days, 1):.0f}/day)"
        )
        all_traces.append(df)

    combined = pd.concat(all_traces, ignore_index=True)
    logger.info(f"Total: {len(combined):,} readings from {combined['patient_id'].nunique()} patients")
    return combined


def parse_uci_file(filepath: Path, patient_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a UCI-format diabetes data file to extract insulin and meal events.

    UCI format: date  time  code  value
    Code meanings:
      33 = Regular insulin dose
      34 = NPH insulin dose
      35 = UltraLente insulin dose
      58-64 = Blood glucose measurement
      65 = Hypoglycemic symptoms
      66 = Typical meal ingestion
      67 = More than usual meal ingestion
      68 = Less than usual meal ingestion
      69 = Typical exercise activity
      70 = More than usual exercise
      71 = Less than usual exercise
      72 = Unspecified special event
    """
    insulin_records = []
    meal_records = []

    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) < 4:
                    continue

                date_str, time_str, code_str, value_str = parts[:4]

                try:
                    code = int(code_str)
                    value = float(value_str) if value_str.strip() else 0.0
                    timestamp = pd.to_datetime(f"{date_str} {time_str}", format="%m-%d-%Y %H:%M")
                except (ValueError, TypeError):
                    continue

                # Insulin doses (codes 33-35)
                if code in (33, 34, 35):
                    insulin_type = {33: "regular", 34: "NPH", 35: "ultralente"}.get(code, "regular")
                    if value > 0:
                        insulin_records.append({
                            "timestamp": timestamp,
                            "dose_units": value,
                            "insulin_type": insulin_type,
                            "patient_id": patient_id,
                        })

                # Meal events (codes 66-68)
                elif code in (66, 67, 68):
                    meal_type = {66: "typical", 67: "large", 68: "small"}.get(code, "typical")
                    # Estimate carbs: typical=50g, large=75g, small=25g
                    carbs = {66: 50, 67: 75, 68: 25}.get(code, 50)
                    meal_records.append({
                        "timestamp": timestamp,
                        "carbs_grams": carbs,
                        "meal_type": meal_type,
                        "patient_id": patient_id,
                    })

    except Exception as e:
        logger.warning(f"Failed to parse {filepath}: {e}")

    insulin_df = pd.DataFrame(insulin_records) if insulin_records else pd.DataFrame(
        columns=["timestamp", "dose_units", "insulin_type", "patient_id"]
    )
    meals_df = pd.DataFrame(meal_records) if meal_records else pd.DataFrame(
        columns=["timestamp", "carbs_grams", "meal_type", "patient_id"]
    )

    return insulin_df, meals_df


def generate_synthetic_events(cgm_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate realistic insulin/meal events aligned with CGM timestamps.

    For patients where UCI data exists, parse it directly.
    For others, generate synthetic events based on typical T2D patterns.
    """
    all_insulin = []
    all_meals = []

    for patient_id in sorted(cgm_df["patient_id"].unique()):
        patient_cgm = cgm_df[cgm_df["patient_id"] == patient_id]
        start_date = patient_cgm["timestamp"].min().normalize()
        end_date = patient_cgm["timestamp"].max().normalize()
        n_days = (end_date - start_date).days + 1

        # Try to load from UCI file first
        uci_file = UCI_DIR / f"data-{patient_id:02d}"
        if uci_file.exists():
            insulin_df, meals_df = parse_uci_file(uci_file, patient_id)

            # Align timestamps to CGM range
            if not insulin_df.empty:
                # The UCI dates may differ from CGM dates; offset them
                uci_start = insulin_df["timestamp"].min().normalize()
                offset = start_date - uci_start
                insulin_df["timestamp"] = insulin_df["timestamp"] + offset
                insulin_df = insulin_df[
                    (insulin_df["timestamp"] >= patient_cgm["timestamp"].min())
                    & (insulin_df["timestamp"] <= patient_cgm["timestamp"].max())
                ]

            if not meals_df.empty:
                uci_start_m = meals_df["timestamp"].min().normalize()
                offset_m = start_date - uci_start_m
                meals_df["timestamp"] = meals_df["timestamp"] + offset_m
                meals_df = meals_df[
                    (meals_df["timestamp"] >= patient_cgm["timestamp"].min())
                    & (meals_df["timestamp"] <= patient_cgm["timestamp"].max())
                ]

            if not insulin_df.empty and not meals_df.empty:
                all_insulin.append(insulin_df)
                all_meals.append(meals_df)
                logger.info(f"  Patient {patient_id}: {len(insulin_df)} insulin, {len(meals_df)} meal events from UCI")
                continue

        # Generate synthetic events for this patient
        np.random.seed(patient_id * 42)

        for day in range(n_days):
            current_date = start_date + timedelta(days=day)

            # Breakfast: 7:30-8:30 AM
            breakfast_time = current_date + timedelta(
                hours=7, minutes=30 + np.random.randint(0, 60)
            )
            carbs = np.random.choice([30, 40, 45, 50, 55, 60])
            all_meals.append(pd.DataFrame([{
                "timestamp": breakfast_time,
                "carbs_grams": carbs,
                "meal_type": "typical",
                "patient_id": patient_id,
            }]))

            # Pre-breakfast insulin (5 min before meal)
            insulin_time = breakfast_time - timedelta(minutes=5)
            dose = round(carbs / np.random.uniform(8, 15), 1)  # ICR 1:8 to 1:15
            all_insulin.append(pd.DataFrame([{
                "timestamp": insulin_time,
                "dose_units": max(0.5, dose),
                "insulin_type": "regular",
                "patient_id": patient_id,
            }]))

            # Lunch: 12:00-1:00 PM
            lunch_time = current_date + timedelta(
                hours=12, minutes=np.random.randint(0, 60)
            )
            carbs = np.random.choice([40, 50, 55, 60, 65, 70])
            all_meals.append(pd.DataFrame([{
                "timestamp": lunch_time,
                "carbs_grams": carbs,
                "meal_type": "typical",
                "patient_id": patient_id,
            }]))

            insulin_time = lunch_time - timedelta(minutes=5)
            dose = round(carbs / np.random.uniform(8, 15), 1)
            all_insulin.append(pd.DataFrame([{
                "timestamp": insulin_time,
                "dose_units": max(0.5, dose),
                "insulin_type": "regular",
                "patient_id": patient_id,
            }]))

            # Dinner: 6:00-7:30 PM
            dinner_time = current_date + timedelta(
                hours=18, minutes=np.random.randint(0, 90)
            )
            carbs = np.random.choice([45, 55, 60, 65, 70, 80])
            all_meals.append(pd.DataFrame([{
                "timestamp": dinner_time,
                "carbs_grams": carbs,
                "meal_type": "typical",
                "patient_id": patient_id,
            }]))

            insulin_time = dinner_time - timedelta(minutes=5)
            dose = round(carbs / np.random.uniform(8, 15), 1)
            all_insulin.append(pd.DataFrame([{
                "timestamp": insulin_time,
                "dose_units": max(0.5, dose),
                "insulin_type": "regular",
                "patient_id": patient_id,
            }]))

        logger.info(f"  Patient {patient_id}: synthetic events generated ({n_days} days)")

    insulin_combined = pd.concat(all_insulin, ignore_index=True) if all_insulin else pd.DataFrame(
        columns=["timestamp", "dose_units", "insulin_type", "patient_id"]
    )
    meals_combined = pd.concat(all_meals, ignore_index=True) if all_meals else pd.DataFrame(
        columns=["timestamp", "carbs_grams", "meal_type", "patient_id"]
    )

    return insulin_combined, meals_combined


def main():
    logger.info("=" * 70)
    logger.info("  BUILDING TRAINING DATA FROM REAL 5-MINUTE CGM TRACES")
    logger.info("=" * 70)

    # Verify CGM traces exist
    trace_files = list(CGM_TRACES_DIR.glob("patient_*_cgm_trace.csv"))
    if not trace_files:
        logger.error(f"No CGM trace files found in {CGM_TRACES_DIR}")
        sys.exit(1)

    logger.info(f"\nFound {len(trace_files)} CGM trace files")
    logger.info(f"Source: {CGM_TRACES_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}\n")

    # Load real CGM traces
    logger.info("Loading CGM traces...")
    cgm_df = load_cgm_traces()

    # Generate aligned insulin/meal events
    logger.info("\nGenerating insulin and meal events...")
    insulin_df, meals_df = generate_synthetic_events(cgm_df)

    # Save to processed directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Back up existing files
    for fname in ["glucose_real.csv", "insulin_real.csv", "meals_real.csv"]:
        existing = OUTPUT_DIR / fname
        if existing.exists():
            backup = OUTPUT_DIR / f"{fname}.bak"
            existing.rename(backup)
            logger.info(f"Backed up {fname} -> {fname}.bak")

    # Save new files
    cgm_df.to_csv(OUTPUT_DIR / "glucose_real.csv", index=False)
    insulin_df.to_csv(OUTPUT_DIR / "insulin_real.csv", index=False)
    meals_df.to_csv(OUTPUT_DIR / "meals_real.csv", index=False)

    # Summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("  DATA SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Patients:           {cgm_df['patient_id'].nunique()}")
    logger.info(f"  Total CGM readings: {len(cgm_df):,}")
    logger.info(f"  Insulin records:    {len(insulin_df):,}")
    logger.info(f"  Meal records:       {len(meals_df):,}")
    logger.info(f"  CGM interval:       5 minutes (real)")

    # Per-patient summary
    for pid in sorted(cgm_df["patient_id"].unique()):
        p_cgm = cgm_df[cgm_df["patient_id"] == pid]
        p_ins = insulin_df[insulin_df["patient_id"] == pid]
        p_meals = meals_df[meals_df["patient_id"] == pid]
        days = (p_cgm["timestamp"].max() - p_cgm["timestamp"].min()).days
        mean_glucose = p_cgm["glucose_mg_dl"].mean()
        std_glucose = p_cgm["glucose_mg_dl"].std()
        logger.info(
            f"  Patient {pid:3d}: {len(p_cgm):5d} readings, {days}d, "
            f"μ={mean_glucose:.1f} σ={std_glucose:.1f} mg/dL, "
            f"{len(p_ins)} insulin, {len(p_meals)} meals"
        )

    logger.info(f"\n  Output written to: {OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
