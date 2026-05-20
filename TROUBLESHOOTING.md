# Troubleshooting Guide

## Common Issues and Solutions

### 1. Backend Failed to Start

**Error**: `Backend failed to start. Check backend.log`

**Solutions**:

```bash
# Check what the error is
cat backend.log

# Common issues:

# A) Port 8080 already in use
lsof -i :8080
# Kill the process using port 8080
kill -9 <PID>

# B) Missing dependencies
source venv/bin/activate  # or: source .venv/bin/activate
pip install -r requirements.txt

# C) Database locked
rm -f data/digital_twin.db

# D) Model file missing
# Train the model first:
python -m src.digital_twin --mode train
```

---

### 2. Frontend Failed to Start

**Error**: Frontend won't start or shows errors

**Solutions**:

```bash
# Check the error
cat frontend.log

# Common issues:

# A) Port 3000 already in use
lsof -i :3000
# Kill the process
kill -9 <PID>

# B) Missing node_modules
cd web
rm -rf node_modules package-lock.json
npm install
cd ..

# C) Build cache issues
cd web
rm -rf .next
npm run build
cd ..
```

---

### 3. Backend Connection Errors

**Error**: "Cannot connect to backend" or "Backend Not Connected"

**Solutions**:

```bash
# 1. Check if backend is actually running
curl http://localhost:8080/health

# 2. Check backend logs
tail -f backend.log

# 3. Restart backend manually
make stop
source venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload

# 4. Check firewall
sudo ufw allow 8080
```

---

### 4. Model Not Loaded

**Error**: "Model not loaded" or "model_loaded: false"

**Solutions**:

```bash
# Check if model file exists
ls -lh checkpoints/best_model.pt

# If missing, train the model
python -m src.digital_twin --mode train

# Check model path in code
grep -r "best_model.pt" src/

# Verify model loads correctly
python -c "import torch; torch.load('checkpoints/best_model.pt')"
```

---

### 5. Database Errors

**Error**: Database locked, connection refused, or table errors

**Solutions**:

```bash
# Reset database (WARNING: deletes all data)
rm -f data/digital_twin.db

# Backup first
cp data/digital_twin.db data/digital_twin.db.backup

# Check database
sqlite3 data/digital_twin.db ".tables"

# Verify schema
sqlite3 data/digital_twin.db ".schema patients"
```

---

### 6. Python Version Issues

**Error**: `SyntaxError` or version-related errors

**Solutions**:

```bash
# Check Python version (need 3.10+)
python --version

# Use specific Python version
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 7. Node.js Version Issues

**Error**: Build errors or compatibility warnings

**Solutions**:

```bash
# Check Node version (need 18+)
node --version

# Update Node.js
# Using nvm:
nvm install 20
nvm use 20

# Or download from nodejs.org
```

---

### 8. Permission Errors

**Error**: `Permission denied` when running scripts

**Solutions**:

```bash
# Make scripts executable
chmod +x start.sh
chmod +x run

# Or run with bash
bash start.sh
```

---

### 9. Memory Errors

**Error**: Out of memory or killed processes

**Solutions**:

```bash
# Check memory usage
free -h

# Reduce batch size (in training)
# Edit src/models/trainer.py:
# batch_size = 16  # Change from 32 to 16

# Use swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

### 10. Gradio/Streamlit Port Conflicts

**Error**: Multiple services trying to use same port

**Solutions**:

```bash
# Stop all services
make stop

# Kill all Python processes
pkill -f python

# Kill all node processes
pkill -f node

# Start fresh
./run
```

---

## Debugging Tips

### Check Logs in Real-Time

```bash
# Backend logs
tail -f backend.log

# Frontend logs
tail -f frontend.log

# Both at once
tail -f backend.log frontend.log
```

### Test Components Separately

```bash
# Test backend only
source venv/bin/activate
uvicorn src.api.main:app --reload

# Test frontend only
cd web
npm run dev
```

### Verify Health

```bash
# Backend health
curl http://localhost:8080/health | jq

# Check if model is loaded
curl http://localhost:8080/health | jq '.model_loaded'

# Test prediction endpoint
curl -X POST http://localhost:8080/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"patient_id": 1, "horizon_minutes": 60}'
```

### Clean Everything

```bash
# Nuclear option - clean everything and start fresh
make clean
rm -rf venv .venv web/node_modules web/.next
make install
./run
```

---

## Getting Help

1. **Check logs**: `backend.log` and `frontend.log`
2. **Health check**: http://localhost:8080/health
3. **API docs**: http://localhost:8080/docs
4. **Test endpoints**: Use curl or Postman
5. **Review code**: Check recent changes

---

## Platform-Specific Issues

### macOS

```bash
# If brew not found
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10

# Install Node
brew install node
```

### Linux

```bash
# Install Python
sudo apt-get update
sudo apt-get install python3.10 python3.10-venv

# Install Node
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### Windows

```powershell
# Use WSL2 (recommended)
wsl --install

# Or use Windows native:
# 1. Install Python from python.org
# 2. Install Node from nodejs.org
# 3. Use Git Bash or PowerShell

# Activate venv on Windows
.\venv\Scripts\activate
```

---

## Still Having Issues?

1. Create a GitHub issue with:
   - Error message
   - Contents of backend.log
   - Contents of frontend.log
   - Python version (`python --version`)
   - Node version (`node --version`)
   - Operating system

2. Include steps to reproduce

3. Mention what you've already tried from this guide
