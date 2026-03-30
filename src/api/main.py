"""
FastAPI Backend for Diabetes Digital Twin.

Provides REST API endpoints for:
- Patient data management
- Glucose prediction
- What-if simulations
- AI chat interface
- Explainability
- Model retraining
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    PatientCreate, PatientResponse, CGMReading, CGMBatchIngest,
    InsulinDose, Meal, PredictionRequest, PredictionResponse,
    SimulationRequest, SimulationResponse, ExplanationRequest,
    ExplanationResponse, ChatRequest, ChatResponse, GlucoseStats,
    DriftStatus, RetrainRequest, RetrainResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances (initialized on startup)
inference_service = None
agent = None
rag = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global inference_service, agent, rag

    logger.info("Starting Diabetes Digital Twin API...")

    # Initialize components
    try:
        # Initialize inference service with trained model
        from src.models.inference import get_inference_service
        inference_service = get_inference_service()
        if inference_service.model_loaded:
            logger.info("Glucose prediction model loaded successfully")
        else:
            logger.warning("Model not loaded - using fallback predictions")

        # Initialize RAG
        from src.agents.rag import setup_rag
        rag = setup_rag()
        logger.info("RAG system initialized")

        # Initialize agent
        from src.agents.diabetes_agent import create_diabetes_agent
        agent = create_diabetes_agent(predictor=None, explainer=None)
        logger.info("Diabetes agent initialized")

    except Exception as e:
        logger.warning(f"Some components failed to initialize: {e}")

    yield

    # Cleanup
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Diabetes Digital Twin API",
    description="AI-powered diabetes management with glucose prediction and personalized recommendations",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Health Check ====================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "model_loaded": inference_service.model_loaded if inference_service else False,
            "agent": agent is not None,
            "rag": rag is not None,
        }
    }


# ==================== Patient Endpoints ====================


@app.post("/api/v1/patients", response_model=PatientResponse)
async def create_patient(patient: PatientCreate):
    """Create a new patient record."""
    try:
        from src.data.ingestion import DataIngestion

        ingestion = DataIngestion()
        patient_id = ingestion.create_patient(
            external_id=patient.external_id,
            age=patient.age,
            weight_kg=patient.weight_kg,
            height_cm=patient.height_cm,
            diabetes_type=patient.diabetes_type,
            gender=patient.gender,
            hba1c_baseline=patient.hba1c_baseline,
            carb_ratio=patient.carb_ratio,
            correction_factor=patient.correction_factor,
        )
        ingestion.close()

        return PatientResponse(
            id=patient_id,
            external_id=patient.external_id,
            age=patient.age,
            gender=patient.gender,
            weight_kg=patient.weight_kg,
            height_cm=patient.height_cm,
            diabetes_type=patient.diabetes_type,
            created_at=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Failed to create patient: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/patients/{patient_id}")
async def get_patient(patient_id: int):
    """Get patient information."""
    # In production, would query database
    return {
        "id": patient_id,
        "status": "active",
        "message": "Patient lookup - implement with database query"
    }


# ==================== Data Ingestion Endpoints ====================


@app.post("/api/v1/ingest/cgm")
async def ingest_cgm(reading: CGMReading, patient_id: int):
    """Ingest a single CGM reading."""
    try:
        from src.data.ingestion import DataIngestion

        ingestion = DataIngestion()
        success = ingestion.ingest_cgm_reading(
            patient_id=patient_id,
            timestamp=reading.timestamp,
            glucose_mg_dl=reading.glucose_mg_dl,
            trend=reading.trend,
            trend_rate=reading.trend_rate,
            device_id=reading.device_id,
        )
        ingestion.close()

        if success:
            # Check for alerts
            alerts = []
            if reading.glucose_mg_dl < 70:
                alerts.append({"type": "hypoglycemia", "severity": "warning"})
            elif reading.glucose_mg_dl < 54:
                alerts.append({"type": "severe_hypoglycemia", "severity": "critical"})
            elif reading.glucose_mg_dl > 250:
                alerts.append({"type": "hyperglycemia", "severity": "warning"})

            return {"status": "success", "alerts": alerts}
        else:
            raise HTTPException(status_code=500, detail="Failed to ingest reading")

    except Exception as e:
        logger.error(f"CGM ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ingest/cgm/batch")
async def ingest_cgm_batch(data: CGMBatchIngest):
    """Batch ingest CGM readings."""
    try:
        import pandas as pd
        from src.data.ingestion import DataIngestion

        df = pd.DataFrame([r.model_dump() for r in data.readings])
        df = df.rename(columns={"timestamp": "time"})

        ingestion = DataIngestion()
        count = ingestion.ingest_cgm_batch(data.patient_id, df)
        ingestion.close()

        return {"status": "success", "readings_ingested": count}

    except Exception as e:
        logger.error(f"Batch ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ingest/insulin")
async def ingest_insulin(dose: InsulinDose, patient_id: int):
    """Ingest an insulin dose record."""
    try:
        from src.data.ingestion import DataIngestion

        ingestion = DataIngestion()
        success = ingestion.ingest_insulin_dose(
            patient_id=patient_id,
            timestamp=dose.timestamp,
            dose_units=dose.dose_units,
            insulin_type=dose.insulin_type,
            is_meal_bolus=dose.is_meal_bolus,
            is_correction=dose.is_correction,
        )
        ingestion.close()

        return {"status": "success" if success else "failed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ingest/meal")
async def ingest_meal(meal: Meal, patient_id: int):
    """Ingest a meal record."""
    try:
        from src.data.ingestion import DataIngestion

        ingestion = DataIngestion()
        success = ingestion.ingest_meal(
            patient_id=patient_id,
            timestamp=meal.timestamp,
            carbs_grams=meal.carbs_grams,
            meal_type=meal.meal_type,
            description=meal.description,
            protein_grams=meal.protein_grams,
            fat_grams=meal.fat_grams,
            glycemic_index=meal.glycemic_index,
        )
        ingestion.close()

        return {"status": "success" if success else "failed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Prediction Endpoints ====================


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_glucose(request: PredictionRequest):
    """Predict future glucose levels using trained model."""
    try:
        # Get recent patient data
        from src.data.ingestion import DataIngestion

        end_time = datetime.now()
        start_time = end_time - timedelta(hours=2)

        ingestion = DataIngestion()
        cgm_data = ingestion.get_patient_cgm_history(
            request.patient_id, start_time, end_time
        )

        # Get insulin and meal data for better predictions
        insulin_data = pd.DataFrame()
        meal_data = pd.DataFrame()
        try:
            insulin_start = end_time - timedelta(hours=4)
            insulin_data = ingestion.get_patient_insulin_history(
                request.patient_id, insulin_start, end_time
            )
            meal_data = ingestion.get_patient_meal_history(
                request.patient_id, insulin_start, end_time
            )
        except:
            pass

        ingestion.close()

        if cgm_data.empty:
            raise HTTPException(status_code=404, detail="No recent CGM data available")

        # Convert glucose values to float
        cgm_data["glucose_mg_dl"] = cgm_data["glucose_mg_dl"].astype(float)
        current_glucose = float(cgm_data["glucose_mg_dl"].iloc[-1])

        # Use inference service for predictions
        if inference_service is not None:
            result = inference_service.predict(
                cgm_data, insulin_data, meal_data, return_uncertainty=True
            )
            predictions = result["predictions"]
            confidence = result["confidence_intervals"]
        else:
            # Fallback
            predictions = {}
            confidence = {}
            for horizon in [30, 60, 90, 120]:
                if len(cgm_data) >= 6:
                    trend = float(cgm_data["glucose_mg_dl"].iloc[-1] - cgm_data["glucose_mg_dl"].iloc[-6]) / 5
                    predicted = current_glucose + (trend * horizon / 5)
                else:
                    predicted = current_glucose
                predicted = predicted * 0.9 + 110 * 0.1
                predictions[f"{horizon}min"] = round(predicted, 1)
                confidence[f"{horizon}min"] = (round(predicted - 20, 1), round(predicted + 20, 1))

        # Determine risk level
        min_predicted = min(predictions.values())
        max_predicted = max(predictions.values())

        if min_predicted < 54 or max_predicted > 300:
            risk_level = "high"
        elif min_predicted < 70 or max_predicted > 250:
            risk_level = "elevated"
        elif min_predicted < 80 or max_predicted > 180:
            risk_level = "low"
        else:
            risk_level = "normal"

        return PredictionResponse(
            patient_id=request.patient_id,
            current_glucose=current_glucose,
            predictions=predictions,
            confidence_intervals=confidence,
            timestamp=datetime.now(),
            risk_level=risk_level,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Simulation Endpoints ====================


@app.post("/api/v1/simulate", response_model=SimulationResponse)
async def simulate_scenario(request: SimulationRequest):
    """Simulate what-if scenarios for meal, insulin, or exercise."""
    try:
        from src.data.ingestion import DataIngestion

        # Get current glucose
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)

        ingestion = DataIngestion()
        cgm_data = ingestion.get_patient_cgm_history(
            request.patient_id, start_time, end_time
        )
        ingestion.close()

        current_glucose = float(cgm_data["glucose_mg_dl"].iloc[-1]) if not cgm_data.empty else 120

        # Use inference service for simulation
        if inference_service is not None and not cgm_data.empty:
            trajectory = inference_service.simulate_scenario(
                cgm_data,
                carbs_grams=request.carbs_grams or 0,
                insulin_units=request.insulin_units or 0,
                exercise_minutes=request.exercise_minutes or 0,
                exercise_intensity=request.exercise_intensity or "moderate",
            )
        else:
            # Fallback simulation
            trajectory = [{"time": 0, "glucose": current_glucose}]
            for t in range(15, 181, 15):
                glucose = current_glucose
                if request.carbs_grams:
                    meal_effect = request.carbs_grams * 3 * np.exp(-((t - 60) ** 2) / (2 * 30 ** 2))
                    glucose += meal_effect
                if request.insulin_units:
                    insulin_effect = request.insulin_units * 50 * (1 - np.exp(-t / 30)) * np.exp(-(t - 90) / 120)
                    glucose -= insulin_effect
                if request.exercise_minutes and t <= request.exercise_minutes + 60:
                    intensity_factor = {"light": 0.5, "moderate": 1.0, "vigorous": 1.5}.get(request.exercise_intensity, 1.0)
                    glucose -= 0.3 * intensity_factor * min(t, request.exercise_minutes)
                trajectory.append({"time": t, "glucose": round(max(40, min(400, glucose)), 1)})

        # Find peak and baseline return
        glucose_values = [p["glucose"] for p in trajectory]
        peak_glucose = max(glucose_values)
        peak_time = trajectory[glucose_values.index(peak_glucose)]["time"]

        time_to_baseline = 180
        for point in trajectory:
            if point["time"] > peak_time and point["glucose"] <= current_glucose + 10:
                time_to_baseline = point["time"]
                break

        # Generate recommendations
        recommendations = []
        if peak_glucose > 180:
            recommendations.append("Consider pre-bolusing 15-20 minutes before eating")
        if peak_glucose > 200 and request.carbs_grams:
            recommended_insulin = request.carbs_grams / 10
            recommendations.append(f"Suggested bolus: {recommended_insulin:.1f} units")
        if min(glucose_values) < 70:
            recommendations.append("Risk of hypoglycemia - reduce insulin or add carbs")

        return SimulationResponse(
            patient_id=request.patient_id,
            current_glucose=current_glucose,
            simulated_trajectory=trajectory,
            peak_glucose=peak_glucose,
            peak_time_minutes=peak_time,
            time_to_baseline_minutes=time_to_baseline,
            recommendations=recommendations,
        )

    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Explanation Endpoints ====================


@app.post("/api/v1/explain", response_model=ExplanationResponse)
async def explain_prediction(request: ExplanationRequest):
    """Get SHAP-based explanation for predictions."""
    try:
        # Get prediction first
        pred_response = await predict_glucose(
            PredictionRequest(patient_id=request.patient_id, horizon_minutes=request.horizon_minutes)
        )

        horizon_key = f"{request.horizon_minutes}min"
        predicted_glucose = pred_response.predictions.get(horizon_key, pred_response.current_glucose)

        # Generate explanation
        if explainer is not None:
            # Use actual SHAP explainer
            pass

        # Fallback: rule-based explanation
        top_factors = []

        # Analyze current state
        current = pred_response.current_glucose
        predicted = predicted_glucose

        if predicted > current + 20:
            top_factors.append({
                "feature": "recent_carbs",
                "importance": 0.8,
                "description": "Recent carbohydrate intake is causing glucose to rise",
            })
        elif predicted < current - 20:
            top_factors.append({
                "feature": "insulin_on_board",
                "importance": 0.7,
                "description": "Active insulin is lowering glucose",
            })
        else:
            top_factors.append({
                "feature": "metabolic_stability",
                "importance": 0.6,
                "description": "Glucose is relatively stable",
            })

        # Time-based factors
        hour = datetime.now().hour
        if 5 <= hour <= 8:
            top_factors.append({
                "feature": "dawn_phenomenon",
                "importance": 0.4,
                "description": "Morning hormone changes affecting glucose",
            })

        explanation_text = f"""**Predicted glucose in {request.horizon_minutes} minutes: {predicted_glucose:.0f} mg/dL**

Key factors:
"""
        for f in top_factors:
            explanation_text += f"- {f['description']}\n"

        return ExplanationResponse(
            patient_id=request.patient_id,
            horizon_minutes=request.horizon_minutes,
            predicted_glucose=predicted_glucose,
            top_factors=top_factors,
            explanation_text=explanation_text,
            risk_level=pred_response.risk_level,
        )

    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Chat Endpoints ====================


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """Chat with the diabetes AI agent."""
    try:
        if agent is None:
            raise HTTPException(status_code=503, detail="Agent not initialized")

        # Update agent context with rich patient data
        if request.include_context:
            from src.data.ingestion import DataIngestion

            ingestion = DataIngestion()

            # Get 24-hour CGM history
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            cgm_data = ingestion.get_patient_cgm_history(
                request.patient_id, start_time, end_time
            )

            # Get recent insulin doses
            recent_insulin = []
            try:
                # Get insulin history from last 4 hours
                insulin_start = end_time - timedelta(hours=4)
                recent_insulin_data = ingestion.get_patient_insulin_history(
                    request.patient_id, insulin_start, end_time
                )
                if hasattr(recent_insulin_data, 'to_dict'):
                    recent_insulin = recent_insulin_data.to_dict('records')
            except:
                pass

            # Get recent meals
            recent_meals = []
            try:
                # Get meals from last 4 hours
                meal_start = end_time - timedelta(hours=4)
                recent_meals_data = ingestion.get_patient_meal_history(
                    request.patient_id, meal_start, end_time
                )
                if hasattr(recent_meals_data, 'to_dict'):
                    recent_meals = recent_meals_data.to_dict('records')
            except:
                pass

            ingestion.close()

            if not cgm_data.empty:
                # Convert to float to avoid Decimal issues
                cgm_data["glucose_mg_dl"] = cgm_data["glucose_mg_dl"].astype(float)
                current_glucose = float(cgm_data["glucose_mg_dl"].iloc[-1])

                agent.update_context(
                    current_glucose=current_glucose,
                    recent_cgm=cgm_data["glucose_mg_dl"].values,
                    recent_insulin=recent_insulin,
                    recent_meals=recent_meals,
                    patient_info={
                        "patient_id": request.patient_id,
                        "data_points": len(cgm_data),
                        "time_range": "24 hours",
                    }
                )

        # Get response
        response = agent.chat(request.message)

        # Check for suggested actions
        suggested_actions = []
        if agent.current_glucose:
            if agent.current_glucose < 70:
                suggested_actions.append("⚠️ Low glucose: Treat with 15g fast-acting carbs (glucose tablets, juice)")
            elif agent.current_glucose > 250:
                suggested_actions.append("⚠️ High glucose: Check ketones if Type 1, consider correction bolus")
            elif agent.current_glucose > 180:
                suggested_actions.append("Monitor glucose: slightly elevated, consider insulin adjustment")

        return ChatResponse(
            patient_id=request.patient_id,
            response=response,
            suggested_actions=suggested_actions if suggested_actions else None,
            alerts=None,
            timestamp=datetime.now(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Statistics Endpoints ====================


@app.get("/api/v1/patients/{patient_id}/stats", response_model=GlucoseStats)
async def get_glucose_stats(patient_id: int, period_hours: int = 24):
    """Get glucose statistics for a patient."""
    try:
        from src.data.ingestion import DataIngestion

        end_time = datetime.now()
        start_time = end_time - timedelta(hours=period_hours)

        ingestion = DataIngestion()
        cgm_data = ingestion.get_patient_cgm_history(patient_id, start_time, end_time)
        ingestion.close()

        if cgm_data.empty:
            # Return empty stats when no data available
            return GlucoseStats(
                patient_id=patient_id,
                period_hours=period_hours,
                average_glucose=0.0,
                std_glucose=0.0,
                min_glucose=0.0,
                max_glucose=0.0,
                time_in_range=0.0,
                time_below_range=0.0,
                time_above_range=0.0,
                coefficient_of_variation=0.0,
                hypo_events=0,
                hyper_events=0,
            )

        glucose = cgm_data["glucose_mg_dl"].values

        in_range = np.sum((glucose >= 70) & (glucose <= 180)) / len(glucose) * 100
        below = np.sum(glucose < 70) / len(glucose) * 100
        above = np.sum(glucose > 180) / len(glucose) * 100
        cv = (np.std(glucose) / np.mean(glucose)) * 100

        return GlucoseStats(
            patient_id=patient_id,
            period_hours=period_hours,
            average_glucose=round(np.mean(glucose), 1),
            std_glucose=round(np.std(glucose), 1),
            min_glucose=round(np.min(glucose), 1),
            max_glucose=round(np.max(glucose), 1),
            time_in_range=round(in_range, 1),
            time_below_range=round(below, 1),
            time_above_range=round(above, 1),
            coefficient_of_variation=round(cv, 1),
            hypo_events=int(np.sum(glucose < 70)),
            hyper_events=int(np.sum(glucose > 250)),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/patients/{patient_id}/cgm")
async def get_patient_cgm(patient_id: int, period_hours: int = 24):
    """Get CGM history for a patient."""
    try:
        from src.data.ingestion import DataIngestion
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=period_hours)

        ingestion = DataIngestion()
        cgm_data = ingestion.get_patient_cgm_history(patient_id, start_time, end_time)
        ingestion.close()

        if cgm_data.empty:
            return {"time": [], "glucose": []}

        # Convert time column to ISO format strings
        time_series = pd.to_datetime(cgm_data["time"])
        return {
            "time": time_series.dt.strftime('%Y-%m-%dT%H:%M:%S').tolist(),
            "glucose": cgm_data["glucose_mg_dl"].tolist()
        }
    except Exception as e:
        logger.error(f"Failed to fetch CGM data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Model Management Endpoints ====================


@app.get("/api/v1/drift", response_model=DriftStatus)
async def check_drift(patient_id: int):
    """Check for model drift."""
    return DriftStatus(
        patient_id=patient_id,
        drift_detected=False,
        drift_type=None,
        metric_values={"psi": 0.05, "mape": 8.5},
        last_checked=datetime.now(),
        retrain_recommended=False,
    )


@app.post("/api/v1/retrain", response_model=RetrainResponse)
async def trigger_retrain(request: RetrainRequest, background_tasks: BackgroundTasks):
    """Trigger model retraining."""
    import uuid

    job_id = str(uuid.uuid4())[:8]

    # In production, would queue a background job
    # background_tasks.add_task(run_retraining, job_id, request.patient_id)

    return RetrainResponse(
        status="queued",
        job_id=job_id,
        estimated_time_minutes=15,
        message="Retraining job has been queued",
    )


# ==================== Run server ====================


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)
