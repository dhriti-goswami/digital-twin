# Diabetes Digital Twin

AI-powered personalized diabetes management system with glucose prediction, what-if simulation, and LLM-powered conversational interface.

## Overview

A production-ready digital twin platform that creates a continuously adaptive virtual replica of diabetes patients. Features multi-horizon glucose prediction using Physics-Informed Neural Networks (PINN) trained on real patient data, integrated with an LLM-powered AI assistant.

**Key Capabilities:**
- 30-120 minute glucose forecasting (Transformer + PINN)
- LLM-powered conversational AI assistant (Ollama/Llama-3)
- What-if simulation for meals, insulin, and exercise
- SHAP-based explainable predictions
- RAG system with 15+ ADA medical guidelines
- Automatic drift detection and model retraining

## Model Performance

| Metric | Value | Clinical Assessment |
|--------|-------|---------------------|
| Overall MAE | **5.55 mg/dL** | Excellent (FDA threshold: <15 mg/dL) |
| RMSE | 8.67 mg/dL | Good |
| 30-min MAE | 5.53 mg/dL | Excellent |
| 60-min MAE | 3.98 mg/dL | Excellent |
| 90-min MAE | 5.10 mg/dL | Excellent |
| 120-min MAE | 7.57 mg/dL | Excellent |

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/diabetes-digital-twin.git
cd diabetes-digital-twin
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the digital twin (interactive CLI)
python -m src.digital_twin

# 3. Or start the full stack
./run.sh all
```

## Usage

### Interactive CLI (Recommended for Testing)

```bash
python -m src.digital_twin
```

Commands available:
- `/predict` - Get glucose predictions
- `/explain` - Explain current predictions
- `/meal N` - Simulate meal with N grams carbs
- `/glucose N` - Update current glucose
- `/status` - Show current status
- Or just type any message to chat with the AI

### Prediction Mode

```bash
python -m src.digital_twin --mode predict --glucose 145
```

Output:
```
Current: 145.0 mg/dL

Predictions:
  30min: 160.9 mg/dL (150.1-171.7)
  60min: 161.3 mg/dL (147.6-175.0)
  90min: 160.9 mg/dL (144.2-177.5)
  120min: 159.6 mg/dL (140.0-179.2)
```

### Chat Mode (Requires Ollama)

```bash
# Start Ollama first: ollama serve
python -m src.digital_twin --mode chat --glucose 145 --message "What should I do before exercising?"
```

### API Server

```bash
python -m src.digital_twin --mode server
# API available at http://localhost:8080
# Swagger docs at http://localhost:8080/docs
```

### Streamlit Dashboard

```bash
python -m src.digital_twin --mode dashboard
# Dashboard at http://localhost:8501
```

## Model Training

Train the glucose prediction model:

```bash
python scripts/train_model.py --epochs 100 --batch-size 64 --model transformer --shap
```

Validate the trained model:

```bash
python scripts/validate_model.py --export-report
```

**Training Output:**
- Model checkpoint: `checkpoints/best_model.pt`
- SHAP analysis: `checkpoints/shap/`
- Validation report: `checkpoints/validation_report.json`

See [Training Methodology](docs/TRAINING_METHODOLOGY.md) for detailed documentation.

## Project Structure

```
diabetes-digital-twin/
├── src/
│   ├── digital_twin.py          # Main production application
│   ├── api/main.py              # FastAPI backend
│   ├── frontend/app.py          # Streamlit dashboard
│   ├── models/
│   │   ├── glucose_predictor.py # Transformer/LSTM + PINN
│   │   └── inference.py         # Production inference service
│   ├── data/
│   │   ├── preprocessing.py     # Feature engineering (43 features)
│   │   └── real_data_parser.py  # Dataset parsers
│   └── agents/
│       ├── diabetes_agent.py    # LangChain + Ollama agent
│       └── rag.py               # Medical guidelines RAG
├── scripts/
│   ├── train_model.py           # Model training with SHAP
│   └── validate_model.py        # Model validation
├── checkpoints/                 # Trained model weights
├── data/
│   ├── raw/                     # Downloaded datasets
│   ├── processed/               # Parsed CSVs
│   └── vectors/                 # ChromaDB RAG storage
└── docs/
    ├── ARCHITECTURE.md          # System architecture
    ├── TRAINING_METHODOLOGY.md  # ML training guide
    └── DEPLOYMENT.md            # Deployment options
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Diabetes Digital Twin                       │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐    │
│  │   Trained     │  │  LLM Agent    │  │    RAG        │    │
│  │  Transformer  │  │  (Ollama)     │  │  (Medical     │    │
│  │  + PINN Model │  │  Llama-3:8b   │  │  Guidelines)  │    │
│  │  MAE: 5.55    │  │               │  │  15 docs      │    │
│  └───────────────┘  └───────────────┘  └───────────────┘    │
│           │                  │                  │            │
│           └──────────────────┼──────────────────┘            │
│                              ▼                               │
│                   ┌───────────────────┐                      │
│                   │  Digital Twin     │                      │
│                   │  - Predictions    │                      │
│                   │  - Explanations   │                      │
│                   │  - Chat/Guidance  │                      │
│                   │  - Simulations    │                      │
│                   └───────────────────┘                      │
│                              │                               │
│              ┌───────────────┼───────────────┐               │
│              ▼               ▼               ▼               │
│         ┌─────────┐    ┌─────────┐    ┌─────────┐           │
│         │  CLI    │    │  API    │    │Dashboard│           │
│         └─────────┘    └─────────┘    └─────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## Data Sources

Trained on real patient data:

| Dataset | Records | Content |
|---------|---------|---------|
| UCI Diabetes | 70 patients | CGM, insulin, meals (30 days each) |
| PIMA Indians | 768 patients | Clinical profiles, glucose tolerance |
| 130-Hospitals | 101k encounters | EHR, HbA1c, medications |

## Deployment

Free deployment options available. See [Deployment Guide](docs/DEPLOYMENT.md).

```bash
# Docker
docker build -t diabetes-twin .
docker run -p 8080:8080 diabetes-twin

# Or use docker-compose
docker compose -f docker-compose.prod.yml up
```

Supported platforms: Render.com, Railway.app, Fly.io, Hugging Face Spaces, Streamlit Cloud.

## Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `API_URL` | Backend API URL | `http://localhost:8080/api/v1` |
| `MODEL_PATH` | Model checkpoint path | `checkpoints/best_model.pt` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |

## Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - System design and components
- [Training Methodology](docs/TRAINING_METHODOLOGY.md) - ML training details
- [Deployment Guide](docs/DEPLOYMENT.md) - Free hosting options

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- Ollama (optional, for LLM chat features)

## Disclaimer

Research prototype using anonymized patient data. Not intended for medical decision-making. Consult healthcare professionals for diabetes management.

## License

MIT License
