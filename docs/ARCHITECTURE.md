# Diabetes Digital Twin - Architecture Documentation

**Version:** 2.0
**Last Updated:** 2026-03-30
**Status:** Production Ready

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Component Details](#3-component-details)
4. [Data Flow](#4-data-flow)
5. [ML Pipeline](#5-ml-pipeline)
6. [API Specifications](#6-api-specifications)
7. [Deployment](#7-deployment)
8. [File Structure](#8-file-structure)

---

## 1. System Overview

Multi-scale medical digital twin for diabetes management. Creates a continuously adaptive virtual replica of patients using real data and physics-informed neural networks.

### Core Capabilities

| Feature | Description | Implementation |
|---------|-------------|----------------|
| Glucose Prediction | 30-120 minute multi-horizon forecasting | `src/models/glucose_predictor.py` |
| What-If Simulation | Meal/insulin/exercise impact modeling | `src/models/inference.py` |
| Explainable AI | SHAP feature importance | `scripts/train_model.py` |
| Natural Language Interface | Conversational AI assistant | `src/agents/diabetes_agent.py` |
| Production Inference | Model serving with fallback | `src/models/inference.py` |

### Key Technical Features

- **Physics-Informed Neural Networks (PINN):** Bergman Minimal Model constraints
- **Multi-horizon Output:** Simultaneous prediction at 30, 60, 90, 120 minutes
- **Uncertainty Quantification:** Monte Carlo dropout for confidence intervals
- **Adaptive Learning:** Drift detection triggers retraining

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND LAYER                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                  Streamlit Dashboard (app.py)                   │ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────────────┐│ │
│  │  │ CGM Chart │ │Predictions│ │Simulation │ │   AI Chat         ││ │
│  │  │ (Plotly)  │ │ Display   │ │   Tool    │ │   Interface       ││ │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────────────┘│ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ HTTP/REST
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          API LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                 FastAPI Backend (main.py)                       │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │ │
│  │  │ /predict │ │/simulate │ │  /chat   │ │ /explain │           │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────┬──────────────────┬──────────────────┬────────────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────────┐
│  ML ENGINE   │  │  LLM ENGINE  │  │         DATA LAYER               │
│              │  │              │  │                                  │
│ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌────────────┐ ┌──────────────┐ │
│ │Inference │ │  │ │ Ollama   │ │  │ │   SQLite   │ │   ChromaDB   │ │
│ │ Service  │ │  │ │ Llama-3  │ │  │ │   (Data)   │ │    (RAG)     │ │
│ └──────────┘ │  │ └──────────┘ │  │ └────────────┘ └──────────────┘ │
│ ┌──────────┐ │  │ ┌──────────┐ │  │                                  │
│ │ PyTorch  │ │  │ │LangChain │ │  │ ┌────────────────────────────┐  │
│ │  Model   │ │  │ │  Agent   │ │  │ │   Medical Guidelines (15+) │  │
│ └──────────┘ │  │ └──────────┘ │  │ │   ADA Standards of Care    │  │
└──────────────┘  └──────────────┘  │ └────────────────────────────┘  │
       ▲                            └──────────────────────────────────┘
       │
┌──────┴───────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                              │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │              scripts/train_model.py                             │ │
│  │  • Data loading (UCI, PIMA, CGM traces)                         │ │
│  │  • Feature engineering (40+ features)                           │ │
│  │  • Transformer/LSTM training                                    │ │
│  │  • PINN loss (Bergman constraints)                              │ │
│  │  • SHAP explainability analysis                                 │ │
│  │  • Checkpoint saving                                            │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Details

### 3.1 Inference Service

**Location:** `src/models/inference.py`

Production-ready inference service that:
- Loads trained model from checkpoint
- Prepares features from raw CGM/insulin/meal data
- Returns multi-horizon predictions with confidence intervals
- Falls back to trend-based prediction if model unavailable

```python
# Usage
from src.models.inference import get_inference_service

service = get_inference_service()
result = service.predict(cgm_df, return_uncertainty=True)
# Returns: {"predictions": {"30min": 125.3, ...}, "confidence_intervals": {...}}
```

### 3.2 Training Pipeline

**Location:** `scripts/train_model.py`

Verbose training script with:
- Progress tracking (per-batch, per-epoch)
- ETA estimation
- Per-horizon MAE reporting
- SHAP feature importance analysis
- Early stopping with patience
- Gradient clipping

```bash
python scripts/train_model.py \
  --epochs 100 \
  --batch-size 64 \
  --model transformer \
  --hidden-size 128 \
  --num-layers 4 \
  --shap
```

### 3.3 Glucose Predictor Model

**Location:** `src/models/glucose_predictor.py`

| Architecture | Parameters | Use Case |
|--------------|------------|----------|
| Transformer | ~500K | Best accuracy, longer training |
| LSTM | ~300K | Faster training, good accuracy |

**Physics-Informed Loss:**
```
Total Loss = MSE Loss + λ × Physics Loss
Physics Loss = violation of Bergman Minimal Model constraints
```

### 3.4 Feature Engineering

**Location:** `src/data/preprocessing.py`

40+ features extracted:

| Category | Features |
|----------|----------|
| CGM | glucose_roc, glucose_mean_1h, glucose_cv, glucose_min/max |
| Insulin | iob_rapid, iob_long, recent_bolus_1h, time_since_bolus |
| Meals | cob, recent_carbs_1h, time_since_meal, carb_rate |
| Temporal | hour_sin, hour_cos, day_of_week, is_dawn_window |
| Activity | steps_1h, exercise_intensity, time_since_exercise |

### 3.5 API Endpoints

**Location:** `src/api/main.py`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/predict` | POST | Multi-horizon glucose prediction |
| `/api/v1/simulate` | POST | What-if scenario simulation |
| `/api/v1/chat` | POST | Natural language AI assistant |
| `/api/v1/patients/{id}/stats` | GET | Glucose statistics |
| `/health` | GET | Service health check |

### 3.6 LLM Agent

**Location:** `src/agents/diabetes_agent.py`

LangChain agent with tools:
- `predict_glucose`: Call prediction model
- `simulate_scenario`: Run what-if analysis
- `search_guidelines`: Query medical knowledge base
- `explain_prediction`: Get SHAP explanation

RAG uses 15+ ADA medical guidelines in ChromaDB.

---

## 4. Data Flow

### 4.1 Training Data Pipeline

```
Real Patient Datasets
        │
        ├── UCI Diabetes (70 patients × 30 days)
        ├── PIMA Indians (768 clinical profiles)
        └── CGM Traces (5-minute intervals)
        │
        ▼
┌─────────────────────────────────────────┐
│         Data Parsers                    │
│  UCIDiabetesParser, PIMAParserdataset   │
│  CGMTraceParser                         │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│      GlucoseFeatureEngine               │
│  • 40+ engineered features              │
│  • IOB/COB calculation                  │
│  • Temporal encoding (sin/cos)          │
│  • Sequence windowing (24 steps)        │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│         Training Loop                   │
│  • AdamW optimizer                      │
│  • Cosine annealing LR                  │
│  • Physics-informed loss                │
│  • Early stopping                       │
└─────────────────────────────────────────┘
        │
        ▼
    checkpoints/best_model.pt
```

### 4.2 Inference Pipeline

```
User Request (CGM history)
        │
        ▼
┌─────────────────────────────────────────┐
│     GlucoseInferenceService             │
│  1. Load recent CGM data                │
│  2. Create features (prepare_features)  │
│  3. Run model inference                 │
│  4. Return predictions + uncertainty    │
└─────────────────────────────────────────┘
        │
        ▼
{
  "predictions": {
    "30min": 125.3,
    "60min": 132.1,
    "90min": 128.7,
    "120min": 118.4
  },
  "confidence_intervals": {
    "30min": [115.3, 135.3],
    ...
  }
}
```

---

## 5. ML Pipeline

### 5.1 Model Architecture

**Transformer (Default):**
```
Input (batch, seq_len, features)
    │
    ▼
Input Projection (Linear)
    │
    ▼
Positional Encoding (Sinusoidal)
    │
    ▼
Transformer Encoder (4 layers)
    │
    ▼
Output Heads (4 separate MLPs)
    │
    ▼
Predictions [30min, 60min, 90min, 120min]
```

### 5.2 Training Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error (primary metric) |
| RMSE | Root Mean Square Error |
| Per-horizon MAE | MAE at each prediction horizon |
| Physics Loss | Bergman model constraint violation |

### 5.3 SHAP Explainability

Top features typically include:
1. `glucose_mg_dl` - Current glucose level
2. `glucose_roc` - Rate of change
3. `cob` - Carbs on board
4. `iob_rapid` - Insulin on board
5. `hour_sin/cos` - Time of day

---

## 6. API Specifications

### Predict Endpoint

```bash
POST /api/v1/predict
Content-Type: application/json

{
  "patient_id": "patient_001",
  "cgm_history": [
    {"time": "2026-03-30T10:00:00", "glucose_mg_dl": 120},
    {"time": "2026-03-30T10:05:00", "glucose_mg_dl": 125},
    ...
  ]
}

Response:
{
  "predictions": {
    "30min": 132.5,
    "60min": 145.2,
    "90min": 138.1,
    "120min": 125.8
  },
  "confidence_intervals": {
    "30min": [122.5, 142.5],
    ...
  },
  "model_used": true
}
```

### Simulate Endpoint

```bash
POST /api/v1/simulate
Content-Type: application/json

{
  "patient_id": "patient_001",
  "carbs_grams": 50,
  "insulin_units": 5,
  "exercise_minutes": 30
}

Response:
{
  "trajectory": [
    {"time": 0, "glucose": 120.0},
    {"time": 15, "glucose": 128.5},
    {"time": 30, "glucose": 145.2},
    ...
  ]
}
```

---

## 7. Deployment

### Local Development

```bash
./run.sh all  # Starts API + frontend
```

### Docker

```bash
docker build -t diabetes-twin .
docker run -p 8080:8080 diabetes-twin
```

### Production (Free Tier Options)

| Platform | Type | Notes |
|----------|------|-------|
| Render.com | Full stack | Sleeps after 15min inactivity |
| Railway.app | Full stack | $5 free credit/month |
| Hugging Face Spaces | ML demo | Best for model showcase |
| Streamlit Cloud | Frontend only | Free, always on |

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

---

## 8. File Structure

```
diabetes-digital-twin/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI endpoints
│   │   └── schemas.py           # Pydantic models
│   ├── models/
│   │   ├── glucose_predictor.py # LSTM/Transformer + PINN
│   │   ├── trainer.py           # Training utilities
│   │   ├── inference.py         # Production inference service
│   │   ├── explainer.py         # SHAP integration
│   │   └── drift_detection.py   # Model drift monitoring
│   ├── data/
│   │   ├── preprocessing.py     # Feature engineering
│   │   ├── real_data_parser.py  # Dataset parsers
│   │   ├── database.py          # Database connections
│   │   └── ingestion.py         # Data storage
│   ├── agents/
│   │   ├── diabetes_agent.py    # LangChain agent
│   │   └── rag.py               # ChromaDB RAG
│   └── frontend/
│       └── app.py               # Streamlit dashboard
├── scripts/
│   └── train_model.py           # Model training script
├── checkpoints/
│   └── best_model.pt            # Trained model weights
├── data/
│   ├── raw/                     # Downloaded datasets
│   ├── processed/               # Parsed CSVs
│   └── vectors/                 # ChromaDB persistence
├── docs/
│   ├── ARCHITECTURE.md          # This file
│   ├── TRAINING_METHODOLOGY.md  # ML training guide
│   └── DEPLOYMENT.md            # Deployment guide
├── Dockerfile
├── docker-compose.prod.yml
├── render.yaml
├── run.sh                       # Convenience runner
└── requirements.txt
```

---

## Component Status

| Component | Status | Location |
|-----------|--------|----------|
| Training Script | Complete | `scripts/train_model.py` |
| Inference Service | Complete | `src/models/inference.py` |
| Glucose Predictor | Complete | `src/models/glucose_predictor.py` |
| Feature Engineering | Complete | `src/data/preprocessing.py` |
| FastAPI Backend | Complete | `src/api/main.py` |
| Streamlit Dashboard | Complete | `src/frontend/app.py` |
| LLM Agent | Complete | `src/agents/diabetes_agent.py` |
| RAG System | Complete | `src/agents/rag.py` |
| Docker Deployment | Complete | `Dockerfile`, `docker-compose.prod.yml` |
| Documentation | Complete | `docs/` |
