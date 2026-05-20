.PHONY: help install dev prod stop clean test build

help:
	@echo "Digital Twin - Available Commands"
	@echo ""
	@echo "  make install    - Install all dependencies (Python + Node.js)"
	@echo "  make dev        - Start development servers (backend + frontend)"
	@echo "  make prod       - Build and start production servers"
	@echo "  make stop       - Stop all running services"
	@echo "  make clean      - Clean build artifacts and caches"
	@echo "  make test       - Run all tests"
	@echo "  make build      - Build frontend for production"
	@echo ""

install:
	@echo "📦 Installing dependencies..."
	@if [ ! -d "venv" ] && [ ! -d ".venv" ]; then \
		python -m venv venv; \
	fi
	@. venv/bin/activate 2>/dev/null || . .venv/bin/activate; \
	pip install -q -r requirements.txt
	@cd web && npm install
	@echo "✓ All dependencies installed"

dev:
	@./start.sh

prod:
	@echo "🚀 Building and starting production servers..."
	@cd web && npm run build
	@./start.sh

stop:
	@echo "🛑 Stopping services..."
	@pkill -f "uvicorn src.api.main:app" || true
	@pkill -f "next dev" || true
	@pkill -f "next start" || true
	@echo "✓ Services stopped"

clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf web/.next
	@rm -rf web/node_modules/.cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@rm -f backend.log frontend.log
	@echo "✓ Cleaned"

test:
	@echo "🧪 Running tests..."
	@. venv/bin/activate 2>/dev/null || . .venv/bin/activate; \
	python -m pytest tests/ -v

build:
	@echo "🔨 Building frontend..."
	@cd web && npm run build
	@echo "✓ Build complete"
