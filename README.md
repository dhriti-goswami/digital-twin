# Diabetes Digital Twin

AI-powered personalized diabetes management system with glucose prediction, what-if simulation, and natural language interface.

## Overview

A digital twin platform that creates a continuously adaptive virtual replica of diabetes patients. Features multi-horizon glucose prediction using Physics-Informed Neural Networks trained on real patient data.

**Key Capabilities:**
- 30-120 minute glucose forecasting (Transformer/LSTM models)
- What-if simulation for meals, insulin, and exercise
- SHAP-based explainable predictions
- Natural language AI assistant (Ollama/Llama-3)
- Automatic drift detection and model retraining

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/diabetes-digital-twin.git
cd diabetes-digital-twin

# 2. Run everything (creates venv, installs deps, starts services)
./run.sh all
```

Or run components individually:

```bash
./run.sh train --epochs 100 --shap   # Train model
./run.sh api                          # Start API only
./run.sh frontend                     # Start dashboard only
```

**Access Points:**
- Dashboard: http://localhost:8501
- API: http://localhost:8080
- API Docs: http://localhost:8080/docs

## Model Training

Train the glucose prediction model on real patient data:

```bash
python scripts/train_model.py --epochs 100 --batch-size 64 --model transformer --shap
```

**Training Output:**
- Model checkpoint: `checkpoints/best_model.pt`
- SHAP analysis: `checkpoints/shap_importance.json`
- Per-horizon MAE for 30/60/90/120 minute predictions

See [Training Methodology](docs/TRAINING_METHODOLOGY.md) for detailed documentation on model architecture, metrics, and parameters.

## Project Structure

```
diabetes-digital-twin/
├── src/
│   ├── api/main.py              # FastAPI backend
│   ├── frontend/app.py          # Streamlit dashboard
│   ├── models/
│   │   ├── glucose_predictor.py # LSTM/Transformer + PINN
│   │   ├── trainer.py           # Training pipeline
│   │   └── inference.py         # Production inference service
│   ├── data/
│   │   ├── preprocessing.py     # Feature engineering (40+ features)
│   │   └── real_data_parser.py  # UCI, PIMA, 130-Hospitals parsers
│   └── agents/
│       ├── diabetes_agent.py    # LangChain + Ollama agent
│       └── rag.py               # Medical guidelines RAG
├── scripts/
│   └── train_model.py           # Model training with verbose progress
├── checkpoints/                 # Trained model weights
├── data/
│   ├── raw/                     # Downloaded datasets
│   └── processed/               # Parsed CSVs
└── docs/
    ├── ARCHITECTURE.md          # System architecture
    ├── TRAINING_METHODOLOGY.md  # ML training guide
    └── DEPLOYMENT.md            # Deployment options
```

## Data Sources

Trained on real patient data:

| Dataset | Records | Content |
|---------|---------|---------|
| UCI Diabetes | 70 patients | CGM, insulin, meals (30 days each) |
| PIMA Indians | 768 patients | Clinical profiles, glucose tolerance |
| 130-Hospitals | 101k encounters | EHR, HbA1c, medications |

## Architecture

```
Streamlit Dashboard
        |
        v (REST API)
FastAPI Backend ─────┬──────┬──────┬───────┐
        |            |      |      |       |
        v            v      v      v       v
   PyTorch      Ollama   ChromaDB  SQLite  Inference
   Models       LLM      (RAG)     (Data)  Service
```

**Physics-Informed Neural Networks:** Models incorporate Bergman Minimal Model constraints for physiologically accurate predictions.

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

## Disclaimer

Research prototype using anonymized patient data. Not intended for medical decision-making. Consult healthcare professionals for diabetes management.

## License

MIT License
