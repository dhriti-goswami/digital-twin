#!/bin/bash
# Run the Diabetes Digital Twin application

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    DIABETES DIGITAL TWIN                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check for virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Check dependencies
echo -e "${BLUE}Checking dependencies...${NC}"
pip install -q -r requirements.txt

# Check for trained model
if [ ! -f "checkpoints/best_model.pt" ]; then
    echo -e "${YELLOW}⚠ No trained model found at checkpoints/best_model.pt${NC}"
    echo -e "${YELLOW}  Run: python scripts/train_model.py --epochs 50${NC}"
    echo ""
fi

# Parse arguments
case "$1" in
    api)
        echo -e "${GREEN}Starting API server on http://localhost:8080${NC}"
        echo -e "${BLUE}API docs: http://localhost:8080/docs${NC}"
        uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8080
        ;;
    frontend)
        echo -e "${GREEN}Starting Streamlit frontend on http://localhost:8501${NC}"
        streamlit run src/frontend/app.py --server.port 8501
        ;;
    train)
        echo -e "${GREEN}Starting model training...${NC}"
        python scripts/train_model.py "${@:2}"
        ;;
    all)
        echo -e "${GREEN}Starting all services...${NC}"
        echo ""
        echo -e "${BLUE}Services:${NC}"
        echo "  • API:       http://localhost:8080"
        echo "  • Frontend:  http://localhost:8501"
        echo "  • API Docs:  http://localhost:8080/docs"
        echo ""

        # Start API in background
        uvicorn src.api.main:app --host 0.0.0.0 --port 8080 &
        API_PID=$!

        # Wait for API to start
        sleep 3

        # Start frontend
        streamlit run src/frontend/app.py --server.port 8501 &
        FRONTEND_PID=$!

        # Handle shutdown
        trap "echo 'Shutting down...'; kill $API_PID $FRONTEND_PID 2>/dev/null" EXIT

        # Wait for both processes
        wait
        ;;
    *)
        echo "Usage: ./run.sh [command]"
        echo ""
        echo "Commands:"
        echo "  api       - Start FastAPI backend only"
        echo "  frontend  - Start Streamlit frontend only"
        echo "  train     - Train the model (pass args after, e.g., ./run.sh train --epochs 50)"
        echo "  all       - Start both API and frontend"
        echo ""
        echo "Examples:"
        echo "  ./run.sh all                        # Start everything"
        echo "  ./run.sh train --epochs 100 --shap  # Train model"
        echo "  ./run.sh api                        # API only"
        ;;
esac
