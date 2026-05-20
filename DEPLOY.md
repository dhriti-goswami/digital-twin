# Diabetes Digital Twin - Deployment Guide

Production-ready diabetes management platform with ML-powered glucose prediction.

## 🚀 Quick Start (One Command)

### Option 1: Docker (Recommended for Production)

```bash
docker compose up --build
```

This starts the entire stack:
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8080/docs
- **PostgreSQL:** localhost:5433
- **Redis:** localhost:6379
- **ChromaDB:** http://localhost:8001

### Option 2: Local Development

```bash
# Using the start script
./start.sh

# Or using Make
make dev
```

## 📋 Prerequisites

### For Docker Deployment
- **Docker:** 20.10+
- **Docker Compose:** 2.0+
- **Ollama:** (optional) For AI chat - `ollama serve`

### For Local Development
- **Python:** 3.10 or higher
- **Node.js:** 18 or higher
- **PostgreSQL:** 15+ (optional, uses SQLite by default)

## 🔧 Installation

### First Time Setup

```bash
# 1. Clone the repository (if not already)
git clone <your-repo-url>
cd digital-twin

# 2. Install all dependencies
make install

# This will:
# - Create Python virtual environment
# - Install Python dependencies
# - Install Node.js dependencies
```

### Manual Installation (if Make is not available)

```bash
# Python setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install uvicorn if not already installed
pip install uvicorn[standard]

# Node.js setup
cd web
npm install
cd ..
```

## 🎯 Running the Application

### Development Mode

```bash
# Option 1: Using start script (recommended)
./start.sh

# Option 2: Using Make
make dev

# Option 3: Manual start
# Terminal 1 - Backend
source venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload

# Terminal 2 - Frontend
cd web
npm run dev
```

### Production Mode

```bash
# Build and run production servers
make prod

# Or manually:
cd web && npm run build && cd ..
cd web && npm start &
cd ..
uvicorn src.api.main:app --host 0.0.0.0 --port 8080
```

## 🛑 Stopping Services

```bash
# Stop all services
make stop

# Or press Ctrl+C in the terminal running ./start.sh
```

## 🧪 Testing

```bash
# Run all tests
make test

# Or manually
source venv/bin/activate
pytest tests/ -v
```

## 🧹 Cleaning

```bash
# Clean build artifacts and caches
make clean
```

## 📁 Project Structure

```
digital-twin/
├── start.sh              # One-command deployment script
├── Makefile              # Make commands for easy deployment
├── src/
│   ├── api/              # FastAPI backend
│   ├── models/           # ML models (Transformer + PINN)
│   ├── data/             # Data processing
│   └── digital_twin.py   # Main application
├── web/
│   ├── src/
│   │   ├── app/          # Next.js pages
│   │   ├── components/   # React components
│   │   └── lib/          # API client & utilities
│   └── package.json
├── checkpoints/
│   └── best_model.pt     # Trained ML model
└── requirements.txt
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```bash
# Backend
DATABASE_URL=sqlite:///./data/digital_twin.db
MODEL_PATH=checkpoints/best_model.pt

# Frontend (create web/.env.local)
NEXT_PUBLIC_API_URL=http://localhost:8080
```

### Backend Configuration

The backend runs on port 8080 by default. To change:

```python
# In src/api/main.py
uvicorn.run(app, host="0.0.0.0", port=8080)  # Change port here
```

### Frontend Configuration

The frontend runs on port 3000 by default. To change:

```bash
# In web/package.json
"dev": "next dev -p 3001"  # Change port here
```

## 🐳 Docker Deployment (Recommended)

### Quick Start

```bash
# Start all services
docker compose up --build

# Or run in background
docker compose up -d

# View logs
docker compose logs -f

# Stop all services
docker compose down
```

### Services

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:3000 | Next.js web app |
| Backend | http://localhost:8080 | FastAPI + ML model |
| API Docs | http://localhost:8080/docs | Swagger UI |
| PostgreSQL | localhost:5433 | TimescaleDB |
| Redis | localhost:6379 | Cache |
| ChromaDB | http://localhost:8001 | Vector DB |

### Health Checks

```bash
# Check backend
curl http://localhost:8080/health

# Check frontend
curl http://localhost:3000

# Check database
docker compose exec postgres pg_isready -U postgres

# View all service status
docker compose ps
```

### Configuration

Edit `docker-compose.yml` or create `.env`:

```bash
# Database
DB__POSTGRES_PASSWORD=your_secure_password

# Backend
OLLAMA_HOST=http://host.docker.internal:11434

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8080
```

### Data Persistence

Data is stored in Docker volumes:
- `postgres_data` - Patient data, readings, predictions
- `redis_data` - Cache
- `chroma_data` - Vector embeddings

To backup database:
```bash
docker compose exec postgres pg_dump -U postgres digital_twin > backup.sql
```

To restore:
```bash
cat backup.sql | docker compose exec -T postgres psql -U postgres digital_twin
```

### Troubleshooting Docker

**Backend not connecting?**
```bash
docker compose logs backend
docker compose logs postgres
```

**Port conflicts?**
```bash
# Check what's using ports
sudo lsof -i :3000  # Frontend
sudo lsof -i :8080  # Backend
sudo lsof -i :5433  # PostgreSQL
```

**Fresh start?**
```bash
# Remove all containers and volumes
docker compose down -v
docker compose up --build
```

---

## 💻 Local Development (No Docker)

## 📊 Features

- **ML-Powered Predictions:** Transformer + PINN model (5.55 mg/dL MAE)
- **Real-time Monitoring:** Live glucose tracking and predictions
- **AI Chat Assistant:** RAG-based medical guidance
- **Insulin & Meal Logging:** IOB/COB calculations affect predictions
- **What-If Simulator:** Scenario planning for carbs/insulin/exercise
- **Multi-horizon Forecasting:** 30, 60, 90, 120 minute predictions

## 🔍 Troubleshooting

### Backend won't start

```bash
# Check if port 8080 is in use
lsof -i :8080

# Check backend logs
tail -f backend.log

# Try running backend directly
source venv/bin/activate
python -m src.api.main
```

### Frontend won't start

```bash
# Check if port 3000 is in use
lsof -i :3000

# Check frontend logs
tail -f frontend.log

# Try running frontend directly
cd web
npm run dev
```

### Model not found error

```bash
# Train the model first
python -m src.digital_twin --mode train

# Or download pre-trained model
# (Add download instructions here)
```

### Database errors

```bash
# Reset database
rm -f data/digital_twin.db
python -m src.digital_twin --mode server
```

## 📝 Available Make Commands

```bash
make help       # Show all commands
make install    # Install dependencies
make dev        # Start development servers
make prod       # Build and start production
make stop       # Stop all services
make clean      # Clean build artifacts
make test       # Run tests
make build      # Build frontend only
```

## 🚀 Production Deployment

### Using systemd (Linux)

Create `/etc/systemd/system/digital-twin.service`:

```ini
[Unit]
Description=Digital Twin Backend
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/digital-twin
Environment="PATH=/path/to/digital-twin/venv/bin"
ExecStart=/path/to/digital-twin/venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8080
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable digital-twin
sudo systemctl start digital-twin
```

### Using PM2 (Node.js process manager)

```bash
# Install PM2
npm install -g pm2

# Start backend
pm2 start "uvicorn src.api.main:app --host 0.0.0.0 --port 8080" --name digital-twin-backend

# Start frontend
cd web && pm2 start npm --name digital-twin-frontend -- start

# Save PM2 configuration
pm2 save
pm2 startup
```

## 📞 Support

For issues or questions:
- Check logs: `backend.log` and `frontend.log`
- Review API docs: http://localhost:8080/docs
- Test backend health: http://localhost:8080/health

## 📄 License

[Add your license here]
