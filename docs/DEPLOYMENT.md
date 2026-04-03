# Deployment Guide

Deploy your Diabetes Digital Twin to the web for **free**.

---

## Quick Start (Local)

```bash
# 1. Train the model (skip if you already have checkpoints/best_model.pt)
python scripts/train_model.py --epochs 100 --shap

# 2. Validate the model
python scripts/validate_model.py --export-report

# 3. Run the interactive CLI (recommended for testing)
python -m src.digital_twin

# 4. Or start the API server
python -m src.digital_twin --mode server

# 5. Or start the dashboard
python -m src.digital_twin --mode dashboard
```

**Access:**
- Interactive CLI: Terminal
- API: http://localhost:8080
- Dashboard: http://localhost:8501
- API Docs: http://localhost:8080/docs

---

## Usage Modes

### Interactive CLI (Best for Testing)

```bash
python -m src.digital_twin

# Commands:
#   /predict  - Get glucose predictions
#   /explain  - Explain current predictions
#   /meal N   - Simulate meal with N grams carbs
#   /glucose N - Update current glucose
#   /status   - Show current status
#   /quit     - Exit
```

### Prediction Only

```bash
python -m src.digital_twin --mode predict --glucose 145

# Output:
# Current: 145.0 mg/dL
# Predictions:
#   30min: 160.9 mg/dL (150.1-171.7)
#   60min: 161.3 mg/dL (147.6-175.0)
#   ...
```

### Chat Mode (Requires Ollama)

```bash
# First start Ollama: ollama serve
python -m src.digital_twin --mode chat --glucose 145 --message "Should I exercise?"
```

---

## Free Deployment Options

### Option 1: Streamlit Cloud (Easiest - Frontend Only)

**Cost:** FREE forever
**Best for:** Demo/showcase

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Set:
   - Main file: `src/frontend/app.py`
   - Python version: 3.11

**Limitations:** Frontend only, no API/model inference

---

### Option 2: Render.com (Full Stack)

**Cost:** FREE (750 hours/month, sleeps after 15min)
**Best for:** Full deployment with API

1. Push code to GitHub

2. Go to [render.com](https://render.com) and sign up

3. Create a new Web Service for API:
   ```
   Name: diabetes-twin-api
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
   ```

4. Create another Web Service for Frontend:
   ```
   Name: diabetes-twin-frontend
   Build Command: pip install -r requirements.txt
   Start Command: streamlit run src/frontend/app.py --server.port=$PORT --server.address=0.0.0.0
   ```

5. Set environment variable on frontend:
   ```
   API_URL=https://diabetes-twin-api.onrender.com/api/v1
   ```

**Or use Blueprint (automatic):**
- Click "New Blueprint" on Render
- Connect your repo
- It will auto-detect `render.yaml`

---

### Option 3: Railway.app (Recommended)

**Cost:** FREE $5 credit/month (enough for light use)
**Best for:** Easiest full deployment

1. Go to [railway.app](https://railway.app)

2. Click "New Project" → "Deploy from GitHub"

3. Select your repo

4. Railway auto-detects the Dockerfile

5. Add environment variables:
   ```
   PORT=8080
   DATABASE_URL=sqlite:///./data/digital_twin.db
   ```

6. Deploy! Railway gives you a URL like:
   `https://your-app.up.railway.app`

---

### Option 4: Hugging Face Spaces (ML-Focused)

**Cost:** FREE
**Best for:** ML demo with model weights

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)

2. Create new Space with Streamlit SDK

3. Upload your code and model checkpoint

4. Add `requirements.txt`

5. It deploys automatically!

**Example Space structure:**
```
your-space/
├── app.py              # Copy of src/frontend/app.py
├── requirements.txt
├── checkpoints/
│   └── best_model.pt   # Your trained model
└── src/
    └── models/
        └── glucose_predictor.py
```

---

### Option 5: Fly.io (Most Flexible)

**Cost:** FREE tier available
**Best for:** Production-like deployment

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Deploy
fly launch

# Set secrets
fly secrets set DATABASE_URL="sqlite:///./data/digital_twin.db"

# Deploy
fly deploy
```

---

## Docker Deployment

For any platform supporting Docker:

```bash
# Build
docker build -t diabetes-twin .

# Run API
docker run -p 8080:8080 diabetes-twin

# Run with compose
docker compose -f docker-compose.prod.yml up
```

---

## Model Deployment Considerations

### Including Trained Model

Your `checkpoints/best_model.pt` is ~10MB. Options:

1. **Include in repo** (simplest for small models)
   - Add to `.gitignore` exclude: `!checkpoints/best_model.pt`

2. **Use Git LFS** (for larger models)
   ```bash
   git lfs install
   git lfs track "*.pt"
   git add checkpoints/best_model.pt
   ```

3. **Download at runtime** (best for large models)
   - Host on Hugging Face Hub or S3
   - Download in app startup

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_URL` | Backend API URL | `http://localhost:8080/api/v1` |
| `DATABASE_URL` | Database connection | `sqlite:///./data/digital_twin.db` |
| `MODEL_PATH` | Path to model checkpoint | `checkpoints/best_model.pt` |
| `OLLAMA_HOST` | Ollama server for LLM | `http://localhost:11434` |

---

## Production Checklist

- [ ] Train model with `--epochs 100 --shap`
- [ ] Test locally with `docker compose -f docker-compose.prod.yml up`
- [ ] Push to GitHub
- [ ] Deploy to chosen platform
- [ ] Set environment variables
- [ ] Test all endpoints
- [ ] Monitor logs

---

## Cost Comparison

| Platform | Monthly Cost | Sleep? | GPU? |
|----------|-------------|--------|------|
| Streamlit Cloud | FREE | No | No |
| Render.com | FREE | Yes (15min) | No |
| Railway.app | ~$5 free credit | No | No |
| HuggingFace Spaces | FREE | No | Available |
| Fly.io | FREE tier | No | No |
| Vercel | FREE | Edge | No |

---

## URLs After Deployment

After deploying, your app will be available at:

- **Render:** `https://your-app.onrender.com`
- **Railway:** `https://your-app.up.railway.app`
- **HuggingFace:** `https://huggingface.co/spaces/username/app-name`
- **Fly.io:** `https://your-app.fly.dev`

---

## Troubleshooting

### "Model not found"
- Ensure `checkpoints/best_model.pt` exists
- Check file is included in deployment (not in `.gitignore`)

### "API connection failed"
- Check `API_URL` environment variable
- Ensure API service is running
- Check CORS settings

### "Out of memory"
- Reduce batch size
- Use CPU inference
- Upgrade to paid tier

### Slow cold starts
- First request after sleep takes 30-60s
- Consider paid tier for no sleep
