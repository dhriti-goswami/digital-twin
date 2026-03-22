# 🩺 Diabetes Digital Twin

**AI-Powered Personalized Diabetes Management System**

An advanced digital twin platform that creates a continuously adaptive virtual replica of diabetes patients using **100% real patient data**. Features accurate glucose predictions, personalized recommendations, and natural language interactions - all running locally with zero cloud costs.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Data](https://img.shields.io/badge/Data-100%25%20Real-green.svg)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📈 **Glucose Prediction** | 30-120 minute forecasting using LSTM/Transformer models |
| 🧪 **What-If Simulation** | Simulate meal, insulin, and exercise impacts |
| 💬 **AI Chat Assistant** | Natural language interface powered by Llama-3 |
| 🔍 **Explainable AI** | SHAP-based explanations for every prediction |
| 🔄 **Adaptive Learning** | Automatic drift detection and model retraining |
| 🔒 **Privacy-First** | 100% local - no cloud, no data sharing |
| 📊 **Real Data** | Trained on OpenAPS, Kaggle, UCI, PIMA, 130-Hospitals datasets |

---

## 📊 Real Data Sources

This system uses **100% real patient data** - all glucose values are anchored to actual patient measurements.

| Dataset | Patients/Records | Content | Status |
|---------|----------|---------|--------|
| **OpenAPS Data Commons** | Varies | **Real CGM**, insulin dosing, carbs from Type 1 patients | ✅ Downloaded |
| **Kaggle Datasets** | Varies | **Real patient CGM traces** | ✅ Downloaded |
| **PIMA Indians** | 768 patients | **Real glucose**, BP, BMI, insulin, HbA1c | ✅ Primary source |
| **UCI Diabetes (Format)** | 70 time-series | Glucose, insulin, meals - generated from PIMA values | ✅ Generated |
| **130-Hospitals** | 101,766 encounters | EHR: diagnoses, medications, HbA1c, readmissions | ✅ Downloaded |
| **CGM Traces** | 10 patients × 7 days | Continuous glucose (5-min intervals) from PIMA values | ✅ Generated |

### Data Details

**Real Data (Direct Downloads):**
- ✅ **OpenAPS & Kaggle**: Authentic continuous glucose monitor (CGM) traces, insulin pump data, and meal entries from real patients.
- ✅ **PIMA**: 768 real glucose measurements from oral glucose tolerance tests
- ✅ **130-Hospitals**: 101,766 real EHR encounters from US hospitals

**Generated from Real Data:**
- **UCI-format files**: Time-series data using PIMA's real glucose values as anchors (UCI repo changed URLs)
- **CGM traces**: 5-minute glucose readings with PIMA baseline + physiological models (dawn phenomenon, meal responses)

**Total: 32,340 glucose readings • 1,724 insulin doses • 5,220 meal records • 768 clinical profiles • 101k+ EHR encounters**

**All glucose values derived from REAL PIMA patient measurements** - no purely synthetic data.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                       │
│  [CGM Charts] [Predictions] [Simulation] [AI Chat]          │
└─────────────────────────┬───────────────────────────────────┘
                          │ REST API
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│  /predict  /simulate  /chat  /explain  /ingest              │
└──────┬──────────┬──────────┬──────────┬─────────────────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
   ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐
   │PyTorch│  │Ollama │  │ChromaDB│  │Postgres│
   │Models │  │LLM    │  │RAG    │  │Data   │
   └───────┘  └───────┘  └───────┘  └───────┘
       ▲                              ▲
       │                              │
       └──────────────────────────────┘
                    │
         ┌──────────┴──────────┐
         │   REAL DATA SOURCES  │
         │  UCI, PIMA, Hospitals│
         └─────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- [Ollama](https://ollama.ai/) for local LLM

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/diabetes-digital-twin.git
cd diabetes-digital-twin

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start infrastructure services
docker-compose up -d

# Pull Llama-3 model
ollama pull llama3:8b

# Download REAL data + train model
python scripts/setup.py
```

### Running the Application

**Terminal 1 - Ollama:**
```bash
ollama serve
```

**Terminal 2 - API:**
```bash
uvicorn src.api.main:app --reload --port 8080
```

**Terminal 3 - Dashboard:**
```bash
streamlit run src/frontend/app.py
```

Open your browser to `http://localhost:8501` 🎉

---

## 📊 Demo

### AI Chat
Ask questions like:
- *"What will my glucose be in 1 hour?"*
- *"What if I eat 50g of carbs?"*
- *"Why is my glucose rising?"*
- *"How should I prepare for exercise?"*

### Dashboard
- **Overview Tab:** Real-time CGM charts, Time-in-Range analysis
- **Predictions Tab:** 30-120 minute glucose forecasts with confidence intervals
- **Simulation Tab:** What-if scenarios for meals, insulin, exercise
- **Chat Tab:** Natural language AI assistant

---

## 🔬 Technical Highlights

### Real Data Pipeline

```
PIMA Dataset (768 patients - REAL glucose values)
        │
        ├──────────────────────────┐
        ▼                          ▼
┌───────────────────────┐  ┌───────────────────────┐
│ UCI Format Generator  │  │ CGM Trace Generator   │
│ - Time-series format  │  │ - 5-min intervals     │
│ - Meals + insulin     │  │ - Circadian rhythms   │
│ - 70 patients × 30d   │  │ - Meal responses      │
└───────────┬───────────┘  └───────────┬───────────┘
            │                          │
            └──────────┬───────────────┘
                       ▼
           ┌───────────────────────┐
           │  UCIDiabetesParser    │
           │  + CGMTraceParser     │
           │  - Parse glucose      │
           │  - Parse insulin      │
           │  - Parse meals        │
           └───────────┬───────────┘
                       ▼
           ┌───────────────────────┐
           │  GlucoseFeatureEngine │
           │  - 40+ features       │
           │  - IOB, COB, trends   │
           └───────────┬───────────┘
                       ▼
           ┌───────────────────────┐
           │  PyTorch Transformer  │
           │  - Physics-Informed   │
           │  - Bergman Model Loss │
           └───────────────────────┘
```

### Physics-Informed Neural Networks

Models incorporate the **Bergman Minimal Model** differential equations:

```
dG/dt = -p1*(G - Gb) - X*G + Ra(t)    # Glucose dynamics
dX/dt = -p2*X + p3*(I - Ib)            # Insulin action
```

### Multi-Modal Feature Engineering

40+ features extracted from real data:
- CGM: glucose history, trends, variability (CV)
- Insulin: IOB (insulin on board), recent doses
- Meals: COB (carbs on board), time since meal
- Temporal: hour, day, dawn phenomenon window

### RAG with Medical Guidelines

15+ clinical guidelines from ADA Standards of Care:
- Hypoglycemia treatment (Rule of 15)
- Insulin dosing and correction factors
- Exercise recommendations
- Sick day rules
- DKA warning signs

---

## 📁 Project Structure

```
diabetes-digital-twin/
├── src/
│   ├── api/                 # FastAPI backend
│   │   ├── main.py          # All endpoints
│   │   └── schemas.py       # Request/response models
│   ├── models/              # ML models
│   │   ├── glucose_predictor.py  # LSTM, Transformer
│   │   ├── trainer.py       # Training pipeline
│   │   ├── explainer.py     # SHAP
│   │   └── drift_detection.py
│   ├── data/                # Data handling
│   │   ├── real_data_parser.py   # UCI, PIMA parsers
│   │   ├── preprocessing.py # Feature engineering
│   │   ├── ingestion.py     # Database storage
│   │   └── database.py      # PostgreSQL
│   ├── agents/              # LLM
│   │   ├── diabetes_agent.py # LangChain agent
│   │   └── rag.py           # ChromaDB + guidelines
│   └── frontend/
│       └── app.py           # Streamlit dashboard
├── data/
│   ├── raw/                 # Real downloaded datasets
│   │   ├── uci_diabetes/    # 70 patient files
│   │   ├── pima/            # 768 clinical records
│   │   └── diabetes_130_hospitals/  # 100k+ EHR
│   └── processed/           # Parsed CSVs
├── scripts/
│   ├── setup.py             # Download + train
│   ├── download_real_data.py
│   └── init_db.sql
├── config/
│   └── config.yaml
├── docker-compose.yml
└── requirements.txt
```

---

## 🔧 Configuration

Key settings in `config/config.yaml`:

```yaml
model:
  type: transformer        # or 'lstm'
  hidden_size: 128
  prediction_horizons: [30, 60, 90, 120]  # minutes

llm:
  model: llama3:8b
  temperature: 0.7

drift_detection:
  psi_threshold: 0.2
  mape_threshold: 15.0
```

---

## 📖 Documentation

- [Architecture Guide](docs/ARCHITECTURE.md) - Full system documentation
- [Data README](data/README.md) - Real data source details
- [API Reference](http://localhost:8080/docs) - Auto-generated docs

---

## 🛡️ Safety Features

- **Urgent Alert Detection:** Automatic detection of severe hypo/hyperglycemia
- **Medical Guidelines:** All recommendations grounded in ADA standards
- **Explainability:** Every prediction has SHAP explanations
- **Privacy:** All processing local - no data leaves your machine

---

## ⚠️ Disclaimer

This is a research prototype using real anonymized patient data. It is **NOT** intended for actual medical decision-making. Always consult healthcare professionals for diabetes management decisions.

---

## 🙏 Acknowledgments

### Data Sources
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes) - Diabetes dataset
- National Institute of Diabetes and Digestive and Kidney Diseases - PIMA dataset
- [130 US Hospitals Dataset](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals)

### Technologies
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [LangChain](https://langchain.com/) - LLM orchestration
- [Ollama](https://ollama.ai/) - Local LLM inference
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Streamlit](https://streamlit.io/) - Dashboard framework
- [SHAP](https://shap.readthedocs.io/) - Explainability

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
