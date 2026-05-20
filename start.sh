#!/bin/bash

# Digital Twin - One Command Deployment
# Starts both backend (Python FastAPI) and frontend (Next.js)

set -e

echo "🚀 Starting Diabetes Digital Twin..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $(jobs -p) 2>/dev/null || true
    exit
}

trap cleanup SIGINT SIGTERM

# Check Python
echo -e "${BLUE}Checking Python...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}Python not found. Please install Python 3.10+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"

# Check Node.js
echo -e "${BLUE}Checking Node.js...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js not found. Please install Node.js 18+${NC}"
    exit 1
fi

NODE_VERSION=$(node --version)
echo -e "${GREEN}✓ Node.js $NODE_VERSION${NC}"
echo ""

# Install Python dependencies if needed
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo -e "${BLUE}Creating Python virtual environment...${NC}"
    python -m venv .venv
fi

# Resolve Python binary from venv (works in any shell, including fish)
PYTHON_BIN="python"
PIP_BIN="pip"
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
    PIP_BIN=".venv/bin/pip"
elif [ -f "venv/bin/python" ]; then
    PYTHON_BIN="venv/bin/python"
    PIP_BIN="venv/bin/pip"
fi

# Check if requirements are installed
echo -e "${BLUE}Checking Python dependencies...${NC}"
if ! $PYTHON_BIN -c "import fastapi" 2>/dev/null; then
    echo -e "${BLUE}Installing Python dependencies...${NC}"
    $PIP_BIN install -q -r requirements.txt
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Python dependencies ready${NC}"
fi

# Check if Node modules are installed
cd web
echo -e "${BLUE}Checking Node.js dependencies...${NC}"
if [ ! -d "node_modules" ]; then
    echo -e "${BLUE}Installing Node.js dependencies...${NC}"
    npm install
    echo -e "${GREEN}✓ Node.js dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Node.js dependencies ready${NC}"
fi
cd ..

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if model exists
if [ ! -f "checkpoints/best_model.pt" ]; then
    echo -e "${RED}⚠️  Warning: Trained model not found at checkpoints/best_model.pt${NC}"
    echo "Run training first with: python -m src.digital_twin --mode train"
    echo ""
fi

# Start backend
echo -e "${GREEN}🔧 Starting Python Backend (port 8080)...${NC}"
cd "$SCRIPT_DIR"
# Resolve uvicorn from venv if available
UVICORN_BIN="uvicorn"
if [ -f ".venv/bin/uvicorn" ]; then UVICORN_BIN=".venv/bin/uvicorn";
elif [ -f "venv/bin/uvicorn" ]; then UVICORN_BIN="venv/bin/uvicorn"; fi
DB__USE_SQLITE=true $UVICORN_BIN src.api.main:app --host 0.0.0.0 --port 8080 --reload > backend.log 2>&1 &
BACKEND_PID=$!
echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"

# Wait for backend to be ready
echo -e "${BLUE}Waiting for backend...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Backend ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Backend failed to start. Check backend.log${NC}"
        cat backend.log
        cleanup
    fi
    sleep 1
done

# Start frontend
echo -e "${GREEN}🌐 Starting Next.js Frontend (port 3000)...${NC}"
cd "$SCRIPT_DIR/web"
npm run dev > ../frontend.log 2>&1 &
FRONTEND_PID=$!
echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}✨ Digital Twin is running!${NC}"
echo ""
echo -e "${BLUE}Frontend:${NC} http://localhost:3000"
echo -e "${BLUE}Backend API:${NC} http://localhost:8080"
echo -e "${BLUE}API Docs:${NC} http://localhost:8080/docs"
echo ""
echo -e "${BLUE}Logs:${NC}"
echo "  Backend: $SCRIPT_DIR/backend.log"
echo "  Frontend: $SCRIPT_DIR/frontend.log"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Wait for processes
wait
