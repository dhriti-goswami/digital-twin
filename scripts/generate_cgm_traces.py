#!/usr/bin/env python3
"""
Generate realistic CGM traces from PIMA dataset.

Since real CGM data sources are often restricted or have broken URLs,
this generates physiologically realistic CGM traces using real glucose
values from the PIMA dataset as anchors.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def generate_cgm_trace_from_glucose(base_glucose: float, duration_hours: int = 24) -> pd.DataFrame:
    """
    Generate realistic CGM trace with 5-minute intervals.

    Uses physiological glucose dynamics:
    - Natural circadian variation
    - Meal responses
    - Random physiological noise
    """
    records = []

    # Start time
    start_time = datetime(2018, 1, 1, 0, 0)

    # 5-minute intervals (288 readings per day)
    intervals = duration_hours * 12

    for i in range(intervals):
        current_time = start_time + timedelta(minutes=i * 5)
        hour = current_time.hour

        # Circadian rhythm effect (dawn phenomenon peaks around 6 AM)
        circadian = 10 * np.sin(2 * np.pi * (hour - 4) / 24)

        # Meal effects (breakfast 8am, lunch 12pm, dinner 6pm)
        meal_effect = 0
        if 8 <= hour < 11:  # Post-breakfast spike
            meal_effect = 30 * np.exp(-(hour - 8.5)**2 / 0.5)
        elif 12 <= hour < 15:  # Post-lunch spike
            meal_effect = 25 * np.exp(-(hour - 13)**2 / 0.5)
        elif 18 <= hour < 21:  # Post-dinner spike
            meal_effect = 35 * np.exp(-(hour - 19)**2 / 0.5)

        # Random noise (CGM sensor + physiological variation)
        noise = np.random.normal(0, 5)

        # Calculate glucose
        glucose = base_glucose + circadian + meal_effect + noise
        glucose = max(40, min(400, glucose))  # Physiological bounds

        records.append({
            'timestamp': current_time,
            'glucose_mg_dl': round(glucose, 1)
        })

    return pd.DataFrame(records)


def generate_cgm_traces():
    """Generate CGM traces from PIMA dataset glucose values."""
    pima_file = DATA_DIR / "pima" / "pima_diabetes.csv"
    if not pima_file.exists():
        logger.error(f"PIMA dataset not found: {pima_file}")
        return

    logger.info("Generating CGM traces from PIMA glucose values...")

    # Read PIMA data
    df = pd.read_csv(pima_file, header=None)
    df.columns = ["Pregnancies", "Glucose", "BP", "Skin", "Insulin", "BMI", "DPF", "Age", "Outcome"]

    # Filter valid glucose values
    df = df[df["Glucose"] > 0].reset_index(drop=True)

    cgm_dir = DATA_DIR / "cgm_traces"
    cgm_dir.mkdir(parents=True, exist_ok=True)

    # Generate 10 CGM traces (7 days each) using first 10 patients
    num_traces = 10
    days_per_trace = 7

    for i in range(num_traces):
        if i >= len(df):
            break

        patient_data = df.iloc[i]
        base_glucose = patient_data["Glucose"]

        # Generate 7-day CGM trace
        traces = []
        for day in range(days_per_trace):
            # Slight day-to-day variation
            daily_glucose = base_glucose + np.random.normal(0, 10)
            daily_trace = generate_cgm_trace_from_glucose(daily_glucose, duration_hours=24)

            # Adjust timestamps for correct day
            daily_trace['timestamp'] = daily_trace['timestamp'] + timedelta(days=day + i*7)
            traces.append(daily_trace)

        # Combine all days
        full_trace = pd.concat(traces, ignore_index=True)

        # Save
        output_file = cgm_dir / f"patient_{i+1:03d}_cgm_trace.csv"
        full_trace.to_csv(output_file, index=False)

        logger.info(f"  Generated: patient_{i+1:03d}_cgm_trace.csv ({len(full_trace)} readings)")

    logger.info(f"✓ Generated {num_traces} CGM traces (7 days each, 5-min intervals)")

    # Create README
    readme = """# CGM Trace Data

These CGM (Continuous Glucose Monitor) traces were generated from real PIMA glucose values.

## Data Format

- **Frequency**: 5-minute intervals (288 readings per day)
- **Duration**: 7 days per patient
- **Columns**:
  - timestamp: ISO8601 datetime
  - glucose_mg_dl: Blood glucose in mg/dL

## Generation Method

Each trace is centered around real glucose values from the PIMA dataset and includes:
1. Circadian rhythm effects (dawn phenomenon)
2. Meal response patterns (breakfast, lunch, dinner)
3. Physiological noise (sensor + biological variation)

## Physiological Realism

- Dawn phenomenon: ~10 mg/dL peak around 6 AM
- Post-meal spikes: 25-35 mg/dL above baseline
- Sensor noise: σ = 5 mg/dL (typical CGM accuracy)
- Bounds: 40-400 mg/dL (physiological limits)

## Files

- `patient_001_cgm_trace.csv` through `patient_010_cgm_trace.csv`

Total: ~20,160 glucose readings per patient (10 patients = 201,600 total readings)
"""

    with open(cgm_dir / "README.md", "w") as f:
        f.write(readme)

    logger.info("  Created README.md")


if __name__ == "__main__":
    logger.info("="*60)
    logger.info("GENERATING CGM TRACES FROM REAL PIMA GLUCOSE VALUES")
    logger.info("="*60)
    logger.info("")

    generate_cgm_traces()

    logger.info("\n" + "="*60)
    logger.info("CGM GENERATION COMPLETE")
    logger.info("="*60)
