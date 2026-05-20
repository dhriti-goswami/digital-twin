"""SQLAlchemy ORM models for the Digital Twin database."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, ForeignKey, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Patient(Base):
    """Patient model."""
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(255), unique=True, nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(10))
    weight_kg = Column(Float, nullable=False)
    height_cm = Column(Float, nullable=False)
    diabetes_type = Column(String(50), default="type1")
    hba1c_baseline = Column(Float)
    total_daily_insulin = Column(Float)
    carb_ratio = Column(Float, default=10.0)
    correction_factor = Column(Float, default=50.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    cgm_readings = relationship("CGMReading", back_populates="patient", cascade="all, delete-orphan")
    insulin_doses = relationship("InsulinDose", back_populates="patient", cascade="all, delete-orphan")
    meals = relationship("Meal", back_populates="patient", cascade="all, delete-orphan")


class CGMReading(Base):
    """Continuous Glucose Monitor readings."""
    __tablename__ = "cgm_readings"
    __table_args__ = (
        UniqueConstraint('time', 'patient_id', name='uq_cgm_time_patient'),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    time = Column(DateTime, nullable=False)
    glucose_mg_dl = Column(Float, nullable=False)
    trend = Column(String(50))
    trend_rate = Column(Float)
    device_id = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="cgm_readings")


class InsulinDose(Base):
    """Insulin dose records."""
    __tablename__ = "insulin_doses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    time = Column(DateTime, nullable=False)
    dose_units = Column(Float, nullable=False)
    insulin_type = Column(String(50), default="rapid")  # rapid, long, mixed
    is_meal_bolus = Column(Boolean, default=False)
    is_correction = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="insulin_doses")


class Meal(Base):
    """Meal records."""
    __tablename__ = "meals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    time = Column(DateTime, nullable=False)
    carbs_grams = Column(Float, nullable=False)
    meal_type = Column(String(50))  # breakfast, lunch, dinner, snack
    description = Column(Text)
    protein_grams = Column(Float)
    fat_grams = Column(Float)
    glycemic_index = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="meals")


class Activity(Base):
    """Physical activity records."""
    __tablename__ = "activities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    time = Column(DateTime, nullable=False)
    duration_minutes = Column(Integer, nullable=False)
    intensity = Column(String(50), default="moderate")  # light, moderate, vigorous
    activity_type = Column(String(100))
    calories_burned = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
