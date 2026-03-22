"""Data ingestion module for CGM, insulin, meals, and activity data."""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.data.database import get_sync_session

logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles ingestion of multi-modal diabetes data."""

    def __init__(self, session: Optional[Session] = None):
        self.session = session or get_sync_session()

    def ingest_cgm_reading(
        self,
        patient_id: int,
        timestamp: datetime,
        glucose_mg_dl: float,
        trend: Optional[str] = None,
        trend_rate: Optional[float] = None,
        device_id: Optional[str] = None,
    ) -> bool:
        """Ingest a single CGM reading."""
        try:
            self.session.execute(
                text("""
                    INSERT INTO cgm_readings (time, patient_id, glucose_mg_dl, trend, trend_rate, device_id)
                    VALUES (:time, :patient_id, :glucose, :trend, :trend_rate, :device_id)
                    ON CONFLICT (time, patient_id) DO UPDATE SET
                        glucose_mg_dl = EXCLUDED.glucose_mg_dl,
                        trend = EXCLUDED.trend,
                        trend_rate = EXCLUDED.trend_rate
                """),
                {
                    "time": timestamp,
                    "patient_id": patient_id,
                    "glucose": glucose_mg_dl,
                    "trend": trend,
                    "trend_rate": trend_rate,
                    "device_id": device_id,
                },
            )
            self.session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to ingest CGM reading: {e}")
            self.session.rollback()
            return False

    def ingest_cgm_batch(self, patient_id: int, df: pd.DataFrame) -> int:
        """
        Batch ingest CGM readings from a DataFrame.

        Expected columns: time/timestamp, glucose/glucose_mg_dl
        Optional columns: trend, trend_rate, device_id
        """
        # Normalize column names
        df = df.copy()
        col_mapping = {
            "timestamp": "time",
            "glucose": "glucose_mg_dl",
            "bg": "glucose_mg_dl",
            "cgm": "glucose_mg_dl",
        }
        df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})

        required_cols = ["time", "glucose_mg_dl"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        df["patient_id"] = patient_id

        # Fill optional columns
        for col in ["trend", "trend_rate", "device_id"]:
            if col not in df.columns:
                df[col] = None

        count = 0
        for _, row in df.iterrows():
            if self.ingest_cgm_reading(
                patient_id=patient_id,
                timestamp=row["time"],
                glucose_mg_dl=row["glucose_mg_dl"],
                trend=row.get("trend"),
                trend_rate=row.get("trend_rate"),
                device_id=row.get("device_id"),
            ):
                count += 1

        logger.info(f"Ingested {count} CGM readings for patient {patient_id}")
        return count

    def ingest_insulin_dose(
        self,
        patient_id: int,
        timestamp: datetime,
        dose_units: float,
        insulin_type: str = "rapid",
        is_meal_bolus: bool = False,
        is_correction: bool = False,
    ) -> bool:
        """Ingest an insulin dose record."""
        try:
            self.session.execute(
                text("""
                    INSERT INTO insulin_doses (time, patient_id, dose_units, insulin_type, is_meal_bolus, is_correction)
                    VALUES (:time, :patient_id, :dose, :type, :meal, :correction)
                """),
                {
                    "time": timestamp,
                    "patient_id": patient_id,
                    "dose": dose_units,
                    "type": insulin_type,
                    "meal": is_meal_bolus,
                    "correction": is_correction,
                },
            )
            self.session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to ingest insulin dose: {e}")
            self.session.rollback()
            return False

    def ingest_meal(
        self,
        patient_id: int,
        timestamp: datetime,
        carbs_grams: float,
        meal_type: Optional[str] = None,
        description: Optional[str] = None,
        protein_grams: Optional[float] = None,
        fat_grams: Optional[float] = None,
        glycemic_index: Optional[int] = None,
    ) -> bool:
        """Ingest a meal record."""
        try:
            self.session.execute(
                text("""
                    INSERT INTO meals (time, patient_id, carbs_grams, meal_type, description,
                                       protein_grams, fat_grams, glycemic_index)
                    VALUES (:time, :patient_id, :carbs, :meal_type, :description,
                            :protein, :fat, :gi)
                """),
                {
                    "time": timestamp,
                    "patient_id": patient_id,
                    "carbs": carbs_grams,
                    "meal_type": meal_type,
                    "description": description,
                    "protein": protein_grams,
                    "fat": fat_grams,
                    "gi": glycemic_index,
                },
            )
            self.session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to ingest meal: {e}")
            self.session.rollback()
            return False

    def ingest_activity(
        self,
        patient_id: int,
        start_time: datetime,
        activity_type: str,
        duration_minutes: int,
        intensity: str = "moderate",
        heart_rate_avg: Optional[int] = None,
        calories_burned: Optional[int] = None,
    ) -> bool:
        """Ingest an activity/exercise record."""
        try:
            end_time = start_time + timedelta(minutes=duration_minutes)
            self.session.execute(
                text("""
                    INSERT INTO activities (start_time, patient_id, end_time, duration_minutes,
                                           activity_type, intensity, heart_rate_avg, calories_burned)
                    VALUES (:start, :patient_id, :end, :duration, :type, :intensity, :hr, :calories)
                """),
                {
                    "start": start_time,
                    "patient_id": patient_id,
                    "end": end_time,
                    "duration": duration_minutes,
                    "type": activity_type,
                    "intensity": intensity,
                    "hr": heart_rate_avg,
                    "calories": calories_burned,
                },
            )
            self.session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to ingest activity: {e}")
            self.session.rollback()
            return False

    def create_patient(
        self,
        external_id: str,
        age: int,
        weight_kg: float,
        height_cm: float,
        diabetes_type: str = "type1",
        gender: Optional[str] = None,
        hba1c_baseline: Optional[float] = None,
        total_daily_insulin: Optional[float] = None,
        carb_ratio: Optional[float] = None,
        correction_factor: Optional[float] = None,
    ) -> int:
        """Create a new patient record and return the patient ID."""
        try:
            result = self.session.execute(
                text("""
                    INSERT INTO patients (external_id, age, gender, weight_kg, height_cm,
                                         diabetes_type, hba1c_baseline, total_daily_insulin,
                                         carb_ratio, correction_factor)
                    VALUES (:ext_id, :age, :gender, :weight, :height, :dtype,
                            :hba1c, :tdi, :cr, :cf)
                    ON CONFLICT (external_id) DO UPDATE SET
                        age = EXCLUDED.age,
                        weight_kg = EXCLUDED.weight_kg,
                        updated_at = NOW()
                    RETURNING id
                """),
                {
                    "ext_id": external_id,
                    "age": age,
                    "gender": gender,
                    "weight": weight_kg,
                    "height": height_cm,
                    "dtype": diabetes_type,
                    "hba1c": hba1c_baseline,
                    "tdi": total_daily_insulin,
                    "cr": carb_ratio,
                    "cf": correction_factor,
                },
            )
            self.session.commit()
            row = result.fetchone()
            return row[0]
        except Exception as e:
            logger.error(f"Failed to create patient: {e}")
            self.session.rollback()
            raise

    def get_patient_cgm_history(
        self,
        patient_id: int,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """Retrieve CGM history for a patient."""
        result = self.session.execute(
            text("""
                SELECT time, glucose_mg_dl, trend, trend_rate
                FROM cgm_readings
                WHERE patient_id = :patient_id
                  AND time BETWEEN :start AND :end
                ORDER BY time ASC
            """),
            {"patient_id": patient_id, "start": start_time, "end": end_time},
        )
        rows = result.fetchall()
        return pd.DataFrame(rows, columns=["time", "glucose_mg_dl", "trend", "trend_rate"])

    def get_patient_insulin_history(
        self,
        patient_id: int,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """Retrieve insulin dose history for a patient."""
        result = self.session.execute(
            text("""
                SELECT time, dose_units, insulin_type, is_meal_bolus, is_correction
                FROM insulin_doses
                WHERE patient_id = :patient_id
                  AND time BETWEEN :start AND :end
                ORDER BY time ASC
            """),
            {"patient_id": patient_id, "start": start_time, "end": end_time},
        )
        rows = result.fetchall()
        return pd.DataFrame(
            rows, columns=["time", "dose_units", "insulin_type", "is_meal_bolus", "is_correction"]
        )

    def get_patient_meal_history(
        self,
        patient_id: int,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """Retrieve meal history for a patient."""
        result = self.session.execute(
            text("""
                SELECT time, carbs_grams, meal_type, protein_grams, fat_grams, glycemic_index
                FROM meals
                WHERE patient_id = :patient_id
                  AND time BETWEEN :start AND :end
                ORDER BY time ASC
            """),
            {"patient_id": patient_id, "start": start_time, "end": end_time},
        )
        rows = result.fetchall()
        return pd.DataFrame(
            rows, columns=["time", "carbs_grams", "meal_type", "protein_grams", "fat_grams", "glycemic_index"]
        )

    def close(self):
        """Close the database session."""
        self.session.close()
