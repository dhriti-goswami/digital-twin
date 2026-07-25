# Diabetes Digital Twin

AI-powered personalized diabetes management system with glucose prediction, what-if simulation, and LLM-powered conversational interface.

## 🚀 Quick Start (One Command!)

```bash
./start.sh
```

That's it! The full application will start automatically:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs

> **Fish shell users:** `./start.sh` uses a `#!/bin/bash` shebang and works directly. No activation needed — it resolves `.venv/bin/uvicorn` internally.

See [DEPLOY.md](DEPLOY.md) for detailed deployment options.

---

## Overview

Multi-horizon glucose forecasting for type 1 diabetes on the OhioT1DM dataset, with a
Bergman-minimal-model constraint and a patient-specific insulin-sensitivity estimate.

> **Correction notice.** An earlier version of this README reported model performance
> that was not reproducible. Those numbers came from a pipeline whose forecast horizons
> were mislabelled (rows with missing CGM were dropped before windows were sliced, so a
> value labelled "+30 min" was not 30 minutes ahead), whose Clarke error grid could not
> assign zone E, and which reported no naive baseline — both of its headline figures
> (30.4 and 78.5 mg/dL RMSE at 30 min) were in fact **worse than predicting no change**.
> It also described the model as a Physics-Informed Neural Network while every script
> that produced a checkpoint passed `use_pinn=False`, so no reported result came from a
> PINN. The pipeline has been rebuilt and re-measured. See
> [`docs/RESULTS.md`](docs/RESULTS.md) and [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

**What is measured now** (research pipeline, `twin/`):

- 30–120 minute forecasting with a **genuinely Bergman-constrained** trajectory: all
  three states integrated, exact matrix-exponential propagation, analytic `dG/dt` from
  a cubic B-spline head.
- **Patient-specific insulin sensitivity** `S_I`, validated against pre-registered
  falsification criteria.
- Two split protocols: the official OhioT1DM temporal holdout and leave-one-subject-out.
- 234 tests, each named for a specific defect it guards against.

**Not part of the research claims:** the FastAPI/Next.js application, the LLM
assistant, and the RAG layer under `src/`. They are retained as an application but are
out of scope, unverified, and produce no reported number. The RAG corpus is
author-paraphrased text, not ingested clinical guidelines.

## Model Performance

OhioT1DM, 12 subjects, 26,498 gap-strict test windows. Metrics computed **per subject,
then reported as mean ± SD across subjects**. Every number is regenerated from stored
predictions by `python -m twin.eval.results_doc`.

### Official protocol — temporal holdout (personalised)

The test files are the *same* subjects over the next ~10 days. This is what every
published OhioT1DM number uses. It is **not** cross-subject generalisation.

| Horizon | Persistence MAE | Model MAE | Persistence RMSE | Model RMSE | Skill |
|---------|-----------------|-----------|------------------|------------|-------|
| 30 min  | 16.87 | **13.25** | 23.37 | **19.06** | 18.4% |
| 60 min  | 28.22 | **22.29** | 38.15 | **30.89** | 19.0% |
| 90 min  | 36.55 | **28.77** | 48.69 | **39.04** | 19.8% |
| 120 min | 42.90 | **33.38** | 56.43 | **44.78** | 20.7% |

### Leave-one-subject-out — subject-disjoint

No data at all from the test subject. Both protocols score the identical windows, so
the difference isolates the value of subject-specific history.

| Horizon | LOSO MAE | Personalisation gap |
|---------|----------|---------------------|
| 30 min  | **14.66** | +1.41 |
| 60 min  | **24.85** | +2.56 |
| 120 min | **37.56** | +4.18 |

### What this does not claim

- **Not state of the art.** Best credible published 30-min MAE on OhioT1DM is
  12.83 mg/dL.
- **`MAE < 15` is not an achievement.** It is the field median — 15 of 17 published
  entries clear it, including a *non-personalised* LSTM at 14.37 — and persistence
  alone reaches 16.36.
- **No citable "clinically acceptable" MAE threshold exists for forecasting.** The
  15 mg/dL figure derives from ISO 15197:2013, a per-reading meter tolerance below
  100 mg/dL measuring the *present*.
- **The physics does not significantly improve accuracy.** No ablation arm beats the
  no-physics baseline after Holm correction. Its value here is a stable
  patient-specific parameter at no accuracy cost.
- **Not clinically deployable.** Hypoglycaemia detection is at or below persistence;
  around 89% Clarke zone A coexists with roughly half of hypoglycaemic events
  undetected. Clarke zone A is not a safety metric.

Persistence is validated against two independently published values to within
0.3 mg/dL, which checks parsing, sequencing, horizon integrity and metrics together.

## Research pipeline

```bash
python -m twin --config configs/official-small.yaml data       # window accounting
python -m twin --config configs/official-small.yaml baselines  # persistence, ROC, ARIMA
python -m twin --config configs/official-small.yaml train
python -m twin --config configs/official-small.yaml ablate     # A0-A7
python -m twin --config configs/official-small.yaml report     # figures and tables
python -m twin.eval.results_doc                                # regenerate RESULTS.md
```

Each stage writes a manifest with the git commit, resolved config, SHA-256 of every
input file, package versions and hardware. Key documents:

| File | Contents |
|---|---|
| [`docs/RESULTS.md`](docs/RESULTS.md) | All results, generated from artifacts |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Every equation, derivation and citation |
| [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) | Outcomes and falsification criteria, fixed in advance |
| [`docs/CITATIONS.md`](docs/CITATIONS.md) | Verified sources, with what could not be verified |

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/diabetes-digital-twin.git
cd diabetes-digital-twin
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd web && npm install && cd ..

# 2. Start the full stack (bash/fish/zsh compatible)
./start.sh
# OR manually:
DB__USE_SQLITE=true .venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8080 &
cd web && npm run dev
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

### Next.js Dashboard

```bash
cd web && npm run dev
# Dashboard at http://localhost:3000
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

| Dataset | Patients | Content | Used for |
|---------|----------|---------|----------|
| UVA/Padova simulator | 30 virtual (10 adolescent, 10 adult, 10 child) | ODE-generated CGM, insulin, meals at 5-min intervals | In-silico pre-training |
| OhioT1DM 2018 | 6 real T1D patients | 8 weeks CGM, bolus/basal insulin, meals, exercise, HR | Fine-tuning & evaluation |
| OhioT1DM 2020 | 6 real T1D patients | Same modalities as 2018 cohort | Fine-tuning & evaluation |

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
