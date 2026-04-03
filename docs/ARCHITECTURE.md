# Diabetes Digital Twin - Architecture Documentation

**Version:** 2.1
**Last Updated:** 2026-04-03
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

Multi-scale medical digital twin for diabetes management. Creates a continuously adaptive virtual replica of patients using real data and physics-informed neural networks, enhanced with LLM-powered conversational AI.

### Validated Model Performance

| Metric | Value | Clinical Status |
|--------|-------|-----------------|
| Overall MAE | **5.55 mg/dL** | Excellent |
| RMSE | 8.67 mg/dL | Good |
| 30-min MAE | 5.53 mg/dL | Excellent |
| 60-min MAE | 3.98 mg/dL | Excellent |
| 90-min MAE | 5.10 mg/dL | Excellent |
| 120-min MAE | 7.57 mg/dL | Excellent |
| Inference Time | 1.58 ms/batch | Fast |
| Throughput | 20,222 samples/sec | High |

### Core Capabilities

| Feature | Description | Implementation |
|---------|-------------|----------------|
| Glucose Prediction | 30-120 minute multi-horizon forecasting | `src/models/glucose_predictor.py` |
| Production Digital Twin | LLM-integrated patient interface | `src/digital_twin.py` |
| What-If Simulation | Meal/insulin/exercise impact modeling | `src/models/inference.py` |
| Explainable AI | SHAP feature importance | `scripts/train_model.py` |
| Natural Language Interface | Conversational AI assistant | `src/agents/diabetes_agent.py` |
| Model Validation | Comprehensive testing & benchmarks | `scripts/validate_model.py` |

### Key Technical Features

- **Physics-Informed Neural Networks (PINN):** Bergman Minimal Model constraints
- **Multi-horizon Output:** Simultaneous prediction at 30, 60, 90, 120 minutes
- **Uncertainty Quantification:** Confidence intervals for predictions
- **Adaptive Learning:** Drift detection triggers retraining
- **LLM Integration:** Ollama-powered conversational AI with medical RAG

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACES                                │
│  ┌─────────────────┐ ┌─────────────────┐ ┌────────────────────────────┐ │
│  │  Interactive    │ │   Streamlit     │ │        FastAPI             │ │
│  │     CLI         │ │   Dashboard     │ │         REST               │ │
│  │  (digital_twin) │ │   (app.py)      │ │        (main.py)           │ │
│  └────────┬────────┘ └────────┬────────┘ └─────────────┬──────────────┘ │
└───────────┼────────────────────┼───────────────────────┼────────────────┘
            │                    │                       │
            └────────────────────┼───────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      DIGITAL TWIN CORE (src/digital_twin.py)            │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │  DiabetesDigitalTwin                                                ││
│  │  ├── predict()          → Multi-horizon glucose predictions         ││
│  │  ├── chat()             → LLM-powered conversational interface      ││
│  │  ├── simulate_meal()    → What-if meal simulation                   ││
│  │  ├── explain_prediction()→ SHAP-based explanation                   ││
│  │  └── update_context()   → Patient state management                  ││
│  └─────────────────────────────────────────────────────────────────────┘│
└──────────┬────────────────────┬────────────────────┬────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────────┐
│   ML ENGINE      │  │   LLM ENGINE     │  │        DATA LAYER            │
│                  │  │                  │  │                              │
│ ┌──────────────┐ │  │ ┌──────────────┐ │  │ ┌────────────┐ ┌───────────┐│
│ │GlucosePredict│ │  │ │   Ollama     │ │  │ │  ChromaDB  │ │  SQLite   ││
│ │ or (Trained  │ │  │ │  Llama-3:8b  │ │  │ │   (RAG)    │ │  (Data)   ││
│ │ Transformer) │ │  │ └──────────────┘ │  │ └────────────┘ └───────────┘│
│ └──────────────┘ │  │ ┌──────────────┐ │  │                              │
│ ┌──────────────┐ │  │ │  LangChain   │ │  │ ┌──────────────────────────┐│
│ │  Inference   │ │  │ │    Agent     │ │  │ │  Medical Guidelines (15+)││
│ │   Service    │ │  │ └──────────────┘ │  │ │  ADA Standards of Care   ││
│ │  (MAE: 5.55) │ │  │                  │  │ └──────────────────────────┘│
│ └──────────────┘ │  │                  │  │                              │
└──────────────────┘  └──────────────────┘  └──────────────────────────────┘
           ▲
           │
┌──────────┴───────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │              scripts/train_model.py                                 │ │
│  │  • Data loading (UCI, PIMA, CGM traces)                             │ │
│  │  • Feature engineering (43 features)                                │ │
│  │  • Transformer training with PINN loss                              │ │
│  │  • SHAP explainability analysis                                     │ │
│  │  • Checkpoint saving → checkpoints/best_model.pt                    │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │              scripts/validate_model.py                              │ │
│  │  • Model loading and architecture verification                      │ │
│  │  • Performance metrics validation                                   │ │
│  │  • Inference speed benchmarking                                     │ │
│  │  • Clinical accuracy assessment                                     │ │
│  │  • Export validation report                                         │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Details

### 3.1 Production Digital Twin

**Location:** `src/digital_twin.py`

The main production-ready application that integrates all components:

```python
from src.digital_twin import DiabetesDigitalTwin

# Initialize
twin = DiabetesDigitalTwin(
    model_path="checkpoints/best_model.pt",
    llm_model="llama3:8b",
    ollama_url="http://localhost:11434"
)

# Update patient context
twin.update_context(current_glucose=145)

# Get predictions
predictions = twin.predict()
# {"predictions": {"30min": 160.9, "60min": 161.3, ...}, "confidence": {...}}

# Chat with AI
response = twin.chat("Should I eat before exercising?")

# Simulate meal
simulation = twin.simulate_meal(carbs=50, insulin=5)
```

### 3.2 Model Validation

**Location:** `scripts/validate_model.py`

Comprehensive model validation script:

```bash
python scripts/validate_model.py --export-report

# Output:
======================================================================
  GLUCOSE PREDICTION MODEL - VALIDATION REPORT
======================================================================

[1. MODEL CONFIGURATION]
    Model Type:       transformer
    Hidden Size:      128
    Num Layers:       4
    Physics Loss:     True
    Total Parameters: 813,892

[2. PERFORMANCE METRICS]
    Validation Loss:  75.1756
    Mean Abs Error:   5.55 mg/dL
    Root MSE:         8.67 mg/dL

[3. CLINICAL ASSESSMENT]
    Clinical Rating:  EXCELLENT
    Suitable for Use: Yes
======================================================================
```

### 3.3 Inference Service

**Location:** `src/models/inference.py`

Production-ready inference service:
- Loads trained model from checkpoint
- Prepares features from raw CGM/insulin/meal data
- Returns multi-horizon predictions with confidence intervals
- Falls back to trend-based prediction if model unavailable

### 3.4 Training Pipeline

**Location:** `scripts/train_model.py`

Verbose training script with:
- Progress tracking (per-batch, per-epoch)
- ETA estimation
- Per-horizon MAE reporting
- SHAP feature importance analysis
- Early stopping with patience
- Gradient clipping
- Physics-informed loss (PINN)

### 3.5 Glucose Predictor Model

**Location:** `src/models/glucose_predictor.py`

| Architecture | Parameters | Validated MAE |
|--------------|------------|---------------|
| Transformer + PINN | 813,892 | 5.55 mg/dL |

**Physics-Informed Loss:**
```
Total Loss = MSE Loss + λ × Physics Loss
Physics Loss = violation of Bergman Minimal Model constraints
```

### 3.6 Feature Engineering

**Location:** `src/data/preprocessing.py`

43 features extracted:

| Category | Features |
|----------|----------|
| CGM | glucose_roc, glucose_mean_1h, glucose_cv, glucose_min/max |
| Insulin | iob_rapid, iob_long, recent_bolus_1h, time_since_bolus |
| Meals | cob, recent_carbs_1h, time_since_meal, carb_rate |
| Temporal | hour_sin, hour_cos, day_of_week, is_dawn_window |
| Activity | steps_1h, exercise_intensity, time_since_exercise |

### 3.7 LLM Agent

**Location:** `src/agents/diabetes_agent.py`

LangChain agent with tools:
- `predict_glucose`: Call prediction model
- `simulate_scenario`: Run what-if analysis
- `search_guidelines`: Query medical knowledge base
- `explain_prediction`: Get SHAP explanation

RAG uses 15+ ADA medical guidelines in ChromaDB.

---

## 4. Data Flow

### 4.1 Real-Time Prediction Pipeline

```
User Request (current glucose, history)
        │
        ▼
┌─────────────────────────────────────────┐
│      DiabetesDigitalTwin                │
│  1. Update context                      │
│  2. Check for urgent situations         │
│  3. Prepare features (43 features)      │
│  4. Run model inference                 │
│  5. Return predictions + confidence     │
└─────────────────────────────────────────┘
        │
        ▼
{
  "predictions": {
    "30min": 160.9,
    "60min": 161.3,
    "90min": 160.9,
    "120min": 159.6
  },
  "confidence": {
    "30min": (150.1, 171.7),
    ...
  },
  "model_used": true
}
```

### 4.2 Chat Pipeline with LLM

```
User Message
    │
    ▼
┌─────────────────────────────────────────┐
│  1. Check urgent situations (hypo/hyper)│
│  2. Build patient context               │
│  3. Get predictions from model          │
│  4. Search RAG for medical guidelines   │
│  5. Build LLM prompt with context       │
│  6. Call Ollama (Llama-3)               │
│  7. Return personalized response        │
└─────────────────────────────────────────┘
    │
    ▼
Personalized AI Response
```

---

## 5. ML Pipeline

### 5.1 Model Architecture

**Transformer (Validated):**
```
Input (batch, seq_len=24, features=43)
    │
    ▼
Input Projection (Linear 43 → 128)
    │
    ▼
Positional Encoding (Sinusoidal)
    │
    ▼
Transformer Encoder (4 layers, 8 heads)
    │
    ▼
Global Avg Pooling
    │
    ▼
Output Heads (4 separate MLPs)
    │
    ▼
Predictions [30min, 60min, 90min, 120min]
```

### 5.2 Training Metrics

| Metric | Description | Achieved |
|--------|-------------|----------|
| MAE | Mean Absolute Error | 5.55 mg/dL |
| RMSE | Root Mean Square Error | 8.67 mg/dL |
| Per-horizon MAE | MAE at each prediction horizon | 5.53/3.98/5.10/7.57 |
| Physics Loss | Bergman model constraint violation | 0.0 |

### 5.3 SHAP Feature Importance

Top features (from `checkpoints/shap/feature_importance.csv`):

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | glucose_roc_15min | 6.30 |
| 2 | glucose_roc_30min | 5.30 |
| 3 | glucose_mean_1h | 4.69 |
| 4 | glucose_roc_5min | 3.66 |
| 5 | hour_sin | 3.28 |

---

## 6. API Specifications

### Predict Endpoint

```bash
POST /api/v1/predict
Content-Type: application/json

{
  "patient_id": "patient_001",
  "cgm_history": [
    {"time": "2026-04-03T10:00:00", "glucose_mg_dl": 120},
    {"time": "2026-04-03T10:05:00", "glucose_mg_dl": 125},
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

### Chat Endpoint

```bash
POST /api/v1/chat
Content-Type: application/json

{
  "patient_id": "patient_001",
  "message": "My glucose is 145, should I exercise?",
  "current_glucose": 145
}

Response:
{
  "response": "With your glucose at 145 mg/dL, you're in a good range for exercise...",
  "predictions": {...}
}
```

---

## 7. Deployment

### Local Development

```bash
# Interactive CLI
python -m src.digital_twin

# API Server + Dashboard
./run.sh all
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

---

## 8. File Structure

```
diabetes-digital-twin/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── __main__.py              # Module entry point
│   ├── digital_twin.py          # Main production application
│   ├── api/
│   │   ├── main.py              # FastAPI endpoints
│   │   └── schemas.py           # Pydantic models
│   ├── models/
│   │   ├── glucose_predictor.py # Transformer + PINN
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
│   ├── train_model.py           # Model training script
│   └── validate_model.py        # Model validation script
├── checkpoints/
│   ├── best_model.pt            # Trained model weights
│   ├── validation_report.json   # Validation metrics
│   └── shap/                    # SHAP analysis outputs
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
├── run.sh
└── requirements.txt
```

---

## Component Status

| Component | Status | Location |
|-----------|--------|----------|
| Production Digital Twin | Complete | `src/digital_twin.py` |
| Model Validation | Complete | `scripts/validate_model.py` |
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
