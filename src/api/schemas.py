"""Pydantic schemas for API request/response models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# Patient Schemas
class PatientCreate(BaseModel):
    external_id: str
    age: int = Field(ge=0, le=120)
    gender: Optional[str] = None
    weight_kg: float = Field(gt=0, le=500)
    height_cm: float = Field(gt=0, le=300)
    diabetes_type: str = "type1"
    hba1c_baseline: Optional[float] = Field(default=None, ge=4, le=20)
    carb_ratio: Optional[float] = Field(default=10, gt=0)
    correction_factor: Optional[float] = Field(default=50, gt=0)


class PatientResponse(BaseModel):
    id: int
    external_id: str
    age: int
    gender: Optional[str]
    weight_kg: float
    height_cm: float
    diabetes_type: str
    created_at: datetime


# CGM Schemas
class CGMReading(BaseModel):
    timestamp: datetime
    glucose_mg_dl: float = Field(gt=20, lt=600)
    trend: Optional[str] = None
    trend_rate: Optional[float] = None
    device_id: Optional[str] = None


class CGMBatchIngest(BaseModel):
    patient_id: int
    readings: list[CGMReading]


# Insulin Schemas
class InsulinDose(BaseModel):
    timestamp: datetime
    dose_units: float = Field(gt=0, le=100)
    insulin_type: str = "rapid"
    is_meal_bolus: bool = False
    is_correction: bool = False


# Meal Schemas
class Meal(BaseModel):
    timestamp: datetime
    carbs_grams: float = Field(ge=0, le=500)
    meal_type: Optional[str] = None
    description: Optional[str] = None
    protein_grams: Optional[float] = Field(default=None, ge=0)
    fat_grams: Optional[float] = Field(default=None, ge=0)
    glycemic_index: Optional[int] = Field(default=None, ge=0, le=100)


# Prediction Schemas
class PredictionRequest(BaseModel):
    patient_id: int
    horizon_minutes: int = Field(default=60, ge=30, le=120)


class PredictionResponse(BaseModel):
    patient_id: int
    current_glucose: float
    predictions: dict[str, float]  # {"30min": 120, "60min": 115, ...}
    confidence_intervals: dict[str, tuple[float, float]]
    timestamp: datetime
    risk_level: str  # "low", "normal", "elevated", "high"


# Simulation Schemas
class SimulationRequest(BaseModel):
    patient_id: int
    carbs_grams: Optional[float] = None
    insulin_units: Optional[float] = None
    exercise_minutes: Optional[int] = None
    exercise_intensity: Optional[str] = "moderate"


class SimulationResponse(BaseModel):
    patient_id: int
    current_glucose: float
    simulated_trajectory: list[dict]  # [{"time": 0, "glucose": 120}, ...]
    peak_glucose: float
    peak_time_minutes: int
    time_to_baseline_minutes: int
    recommendations: list[str]


# Explanation Schemas
class ExplanationRequest(BaseModel):
    patient_id: int
    horizon_minutes: int = 60


class ExplanationResponse(BaseModel):
    patient_id: int
    horizon_minutes: int
    predicted_glucose: float
    top_factors: list[dict]  # [{"feature": "...", "importance": 0.5, "description": "..."}]
    explanation_text: str
    risk_level: str


# Chat Schemas
class ChatRequest(BaseModel):
    patient_id: int
    message: str
    include_context: bool = True


class ChatResponse(BaseModel):
    patient_id: int
    response: str
    suggested_actions: Optional[list[str]] = None
    alerts: Optional[list[dict]] = None
    timestamp: datetime


# Statistics Schemas
class GlucoseStats(BaseModel):
    patient_id: int
    period_hours: int
    average_glucose: float
    std_glucose: float
    min_glucose: float
    max_glucose: float
    time_in_range: float  # percentage
    time_below_range: float
    time_above_range: float
    coefficient_of_variation: float
    hypo_events: int
    hyper_events: int


# Drift Detection Schemas
class DriftStatus(BaseModel):
    patient_id: int
    drift_detected: bool
    drift_type: Optional[str] = None
    metric_values: dict
    last_checked: datetime
    retrain_recommended: bool


# Retraining Schemas
class RetrainRequest(BaseModel):
    patient_id: Optional[int] = None  # None = global retrain
    force: bool = False


class RetrainResponse(BaseModel):
    status: str
    job_id: str
    estimated_time_minutes: int
    message: str
