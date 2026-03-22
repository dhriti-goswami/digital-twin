"""
Parser for REAL diabetes datasets.

This module parses actual patient data from publicly available sources:
1. UCI Diabetes Dataset - 70 patients with blood glucose, insulin, meals
2. PIMA Indians Diabetes - 768 patients with clinical measurements
3. Diabetes 130-Hospitals - 100k+ EHR encounters

All data is REAL, anonymized patient data - NOT simulated.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import re

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class UCIDiabetesParser:
    """
    Parser for UCI Diabetes Dataset.

    This dataset contains REAL data from 70 Type 1 diabetes patients.
    Each file contains timestamped records of:
    - Blood glucose measurements
    - Insulin doses (Regular, NPH, UltraLente)
    - Meals
    - Exercise
    - Special events

    Data format:
    MM-DD-YYYY    HH:MM    Code    Value

    Code meanings:
    33 = Regular insulin dose
    34 = NPH insulin dose
    35 = UltraLente insulin dose
    48, 57 = Unspecified blood glucose measurement
    58 = Pre-breakfast blood glucose
    59 = Post-breakfast blood glucose
    60 = Pre-lunch blood glucose
    61 = Post-lunch blood glucose
    62 = Pre-supper blood glucose
    63 = Post-supper blood glucose
    64 = Pre-snack blood glucose
    65 = Hypoglycemic symptoms
    66 = Typical meal ingestion
    67 = More-than-usual meal ingestion
    68 = Less-than-usual meal ingestion
    69 = Typical exercise activity
    70 = More-than-usual exercise activity
    71 = Less-than-usual exercise activity
    72 = Unspecified special event
    """

    # Code mappings
    GLUCOSE_CODES = {48, 57, 58, 59, 60, 61, 62, 63, 64}
    INSULIN_CODES = {33: "regular", 34: "NPH", 35: "ultralente"}
    MEAL_CODES = {66: "typical", 67: "large", 68: "small"}
    EXERCISE_CODES = {69: "typical", 70: "more", 71: "less"}
    GLUCOSE_TIMING = {
        58: "pre_breakfast",
        59: "post_breakfast",
        60: "pre_lunch",
        61: "post_lunch",
        62: "pre_supper",
        63: "post_supper",
        64: "pre_snack",
    }

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def parse_patient_file(self, filepath: Path) -> dict[str, pd.DataFrame]:
        """Parse a single UCI patient file."""
        records = []

        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 4:
                        continue

                    try:
                        date_str = parts[0]
                        time_str = parts[1]
                        code = int(parts[2])
                        value = float(parts[3]) if parts[3] else None

                        # Parse datetime
                        dt_str = f"{date_str} {time_str}"
                        try:
                            timestamp = datetime.strptime(dt_str, "%m-%d-%Y %H:%M")
                        except ValueError:
                            continue

                        records.append({
                            "timestamp": timestamp,
                            "code": code,
                            "value": value,
                        })
                    except (ValueError, IndexError):
                        continue

        except Exception as e:
            logger.warning(f"Error parsing {filepath}: {e}")
            return {}

        if not records:
            return {}

        df = pd.DataFrame(records)

        # Separate into different data types
        result = {}

        # Blood glucose readings
        glucose_mask = df["code"].isin(self.GLUCOSE_CODES)
        if glucose_mask.any():
            glucose_df = df[glucose_mask].copy()
            glucose_df["glucose_mg_dl"] = glucose_df["value"]
            glucose_df["timing"] = glucose_df["code"].map(self.GLUCOSE_TIMING)
            result["glucose"] = glucose_df[["timestamp", "glucose_mg_dl", "timing"]].dropna(subset=["glucose_mg_dl"])

        # Insulin doses
        insulin_mask = df["code"].isin(self.INSULIN_CODES.keys())
        if insulin_mask.any():
            insulin_df = df[insulin_mask].copy()
            insulin_df["dose_units"] = insulin_df["value"]
            insulin_df["insulin_type"] = insulin_df["code"].map(self.INSULIN_CODES)
            result["insulin"] = insulin_df[["timestamp", "dose_units", "insulin_type"]].dropna(subset=["dose_units"])

        # Meals
        meal_mask = df["code"].isin(self.MEAL_CODES.keys())
        if meal_mask.any():
            meal_df = df[meal_mask].copy()
            meal_df["meal_size"] = meal_df["code"].map(self.MEAL_CODES)
            # Estimate carbs based on meal size
            meal_df["carbs_grams"] = meal_df["meal_size"].map({
                "typical": 50,
                "large": 75,
                "small": 25,
            })
            result["meals"] = meal_df[["timestamp", "carbs_grams", "meal_size"]]

        # Exercise
        exercise_mask = df["code"].isin(self.EXERCISE_CODES.keys())
        if exercise_mask.any():
            exercise_df = df[exercise_mask].copy()
            exercise_df["intensity"] = exercise_df["code"].map(self.EXERCISE_CODES)
            result["exercise"] = exercise_df[["timestamp", "intensity"]]

        return result

    def parse_all_patients(self) -> dict[str, dict[str, pd.DataFrame]]:
        """Parse all patient files in the directory."""
        all_data = {}

        for filepath in sorted(self.data_dir.glob("data-*")):
            patient_id = filepath.name  # e.g., "data-01"
            data = self.parse_patient_file(filepath)
            if data:
                all_data[patient_id] = data
                logger.info(f"Parsed {patient_id}: {sum(len(df) for df in data.values())} records")

        logger.info(f"Total patients parsed: {len(all_data)}")
        return all_data

    def to_unified_format(self) -> dict[str, pd.DataFrame]:
        """Convert all patient data to unified format for the system."""
        all_patients = self.parse_all_patients()

        all_glucose = []
        all_insulin = []
        all_meals = []
        all_activities = []

        for patient_id, data in all_patients.items():
            # Extract patient number
            patient_num = int(patient_id.replace("data-", ""))

            if "glucose" in data:
                df = data["glucose"].copy()
                df["patient_id"] = patient_num
                df["source"] = "UCI"
                all_glucose.append(df)

            if "insulin" in data:
                df = data["insulin"].copy()
                df["patient_id"] = patient_num
                all_insulin.append(df)

            if "meals" in data:
                df = data["meals"].copy()
                df["patient_id"] = patient_num
                all_meals.append(df)

            if "exercise" in data:
                df = data["exercise"].copy()
                df["patient_id"] = patient_num
                all_activities.append(df)

        return {
            "glucose": pd.concat(all_glucose, ignore_index=True) if all_glucose else pd.DataFrame(),
            "insulin": pd.concat(all_insulin, ignore_index=True) if all_insulin else pd.DataFrame(),
            "meals": pd.concat(all_meals, ignore_index=True) if all_meals else pd.DataFrame(),
            "activities": pd.concat(all_activities, ignore_index=True) if all_activities else pd.DataFrame(),
        }


class PIMAParserdataset:
    """
    Parser for PIMA Indians Diabetes Dataset.

    REAL clinical data from 768 Pima Indian women including:
    - Glucose: Plasma glucose concentration (2 hours after OGTT)
    - BloodPressure: Diastolic blood pressure (mm Hg)
    - SkinThickness: Triceps skin fold thickness (mm)
    - Insulin: 2-Hour serum insulin (mu U/ml)
    - BMI: Body mass index
    - DiabetesPedigreeFunction: Genetic diabetes risk
    - Age: Years
    - Outcome: Diabetes diagnosis (0 or 1)
    """

    COLUMNS = [
        "pregnancies",
        "glucose",  # This is oral glucose tolerance test result
        "blood_pressure",
        "skin_thickness",
        "insulin",  # 2-hour serum insulin
        "bmi",
        "diabetes_pedigree",
        "age",
        "outcome",
    ]

    def __init__(self, filepath: Path):
        self.filepath = Path(filepath)

    def parse(self) -> pd.DataFrame:
        """Parse the PIMA dataset."""
        df = pd.read_csv(self.filepath, names=self.COLUMNS)

        # Replace 0 values with NaN for physiologically impossible values
        for col in ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]:
            df.loc[df[col] == 0, col] = np.nan

        # Add patient IDs
        df["patient_id"] = range(1, len(df) + 1)

        # Calculate estimated HbA1c from glucose (using ADAG formula)
        # HbA1c = (glucose + 46.7) / 28.7
        df["estimated_hba1c"] = (df["glucose"] + 46.7) / 28.7

        logger.info(f"Parsed PIMA dataset: {len(df)} patients")
        return df

    def get_clinical_profiles(self) -> pd.DataFrame:
        """Get patient profiles for the digital twin system."""
        df = self.parse()

        profiles = df[[
            "patient_id",
            "age",
            "bmi",
            "glucose",
            "insulin",
            "blood_pressure",
            "estimated_hba1c",
            "outcome",
        ]].copy()

        profiles = profiles.rename(columns={
            "outcome": "diabetes_status",
            "bmi": "bmi",
        })

        profiles["diabetes_type"] = profiles["diabetes_status"].map({0: "none", 1: "type2"})
        profiles["weight_kg"] = profiles["bmi"] * 1.7 ** 2  # Estimate for avg height 1.7m

        return profiles


class Diabetes130HospitalsParser:
    """
    Parser for Diabetes 130-US Hospitals Dataset.

    REAL EHR data from 130 US hospitals (1999-2008):
    - 101,766 patient encounters
    - Diagnoses, procedures, medications
    - Lab results including HbA1c
    - Hospital readmission outcomes
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def parse(self) -> pd.DataFrame:
        """Parse the 130 hospitals dataset."""
        csv_path = self.data_dir / "diabetic_data.csv"

        if not csv_path.exists():
            logger.warning(f"Dataset not found: {csv_path}")
            return pd.DataFrame()

        df = pd.read_csv(csv_path, low_memory=False)

        # Clean up the data
        df = df.replace("?", np.nan)

        # Convert numeric columns
        numeric_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                        "num_medications", "number_outpatient", "number_emergency",
                        "number_inpatient", "number_diagnoses"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(f"Parsed 130 Hospitals dataset: {len(df)} encounters")
        return df

    def get_hba1c_records(self) -> pd.DataFrame:
        """Extract HbA1c measurements from the dataset."""
        df = self.parse()

        if df.empty or "A1Cresult" not in df.columns:
            return pd.DataFrame()

        # Filter to records with HbA1c results
        hba1c_df = df[df["A1Cresult"].notna()].copy()

        # Map HbA1c categories to estimated values
        hba1c_map = {
            ">8": 9.0,  # Average for >8%
            ">7": 7.5,  # Average for 7-8%
            "Norm": 5.5,  # Normal range
        }
        hba1c_df["hba1c_value"] = hba1c_df["A1Cresult"].map(hba1c_map)

        return hba1c_df[[
            "patient_nbr",
            "age",
            "gender",
            "hba1c_value",
            "time_in_hospital",
            "num_medications",
            "diabetesMed",
        ]]


class CGMTraceParser:
    """
    Parser for CGM trace data from research datasets.

    Real continuous glucose monitoring data with timestamps and values.
    """

    def __init__(self, filepath: Path):
        self.filepath = Path(filepath)

    def parse(self) -> pd.DataFrame:
        """Parse CGM trace file."""
        try:
            df = pd.read_csv(self.filepath)

            # Standardize column names
            df.columns = df.columns.str.lower().str.replace(" ", "_")

            # Look for glucose column
            glucose_col = None
            for col in df.columns:
                if "glucose" in col or "bg" in col or "cgm" in col:
                    glucose_col = col
                    break

            if glucose_col:
                df = df.rename(columns={glucose_col: "glucose_mg_dl"})

            # Look for time column
            time_col = None
            for col in df.columns:
                if "time" in col or "date" in col or "timestamp" in col:
                    time_col = col
                    break

            if time_col:
                df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")

            logger.info(f"Parsed CGM trace: {len(df)} readings")
            return df

        except Exception as e:
            logger.error(f"Error parsing CGM trace: {e}")
            return pd.DataFrame()


def load_all_real_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load all available real datasets and return unified data.

    Returns dict with:
    - patients: Patient profiles
    - glucose: All glucose readings
    - insulin: All insulin doses
    - meals: All meal records
    - clinical: Clinical measurements (HbA1c, etc.)
    """
    data_dir = Path(data_dir)
    all_data = {}

    # UCI Diabetes Dataset
    uci_dir = data_dir / "uci_diabetes"
    if uci_dir.exists():
        logger.info("Loading UCI Diabetes Dataset...")
        parser = UCIDiabetesParser(uci_dir)
        uci_data = parser.to_unified_format()
        for key, df in uci_data.items():
            if not df.empty:
                all_data[key] = df
                logger.info(f"  UCI {key}: {len(df)} records")

    # PIMA Dataset
    pima_file = data_dir / "pima" / "pima_diabetes.csv"
    if pima_file.exists():
        logger.info("Loading PIMA Indians Dataset...")
        parser = PIMAParserdataset(pima_file)
        pima_profiles = parser.get_clinical_profiles()
        all_data["pima_profiles"] = pima_profiles
        logger.info(f"  PIMA profiles: {len(pima_profiles)} patients")

    # 130 Hospitals Dataset
    hospitals_dir = data_dir / "diabetes_130_hospitals"
    if hospitals_dir.exists():
        logger.info("Loading 130-Hospitals Dataset...")
        parser = Diabetes130HospitalsParser(hospitals_dir)
        hba1c_records = parser.get_hba1c_records()
        if not hba1c_records.empty:
            all_data["clinical_hba1c"] = hba1c_records
            logger.info(f"  130-Hospitals HbA1c: {len(hba1c_records)} records")

    # CGM Traces
    cgm_dir = data_dir / "cgm_traces"
    if cgm_dir.exists():
        for cgm_file in cgm_dir.glob("*.csv"):
            logger.info(f"Loading CGM trace: {cgm_file.name}...")
            parser = CGMTraceParser(cgm_file)
            cgm_df = parser.parse()
            if not cgm_df.empty and "glucose_mg_dl" in cgm_df.columns:
                key = f"cgm_{cgm_file.stem}"
                all_data[key] = cgm_df
                logger.info(f"  {cgm_file.name}: {len(cgm_df)} readings")

    return all_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test parsing
    data_dir = Path(__file__).parent.parent / "data" / "raw"

    if data_dir.exists():
        data = load_all_real_data(data_dir)
        print(f"\nLoaded {len(data)} datasets:")
        for key, df in data.items():
            print(f"  {key}: {len(df)} records")
    else:
        print(f"Data directory not found: {data_dir}")
        print("Run: python scripts/download_real_data.py first")
