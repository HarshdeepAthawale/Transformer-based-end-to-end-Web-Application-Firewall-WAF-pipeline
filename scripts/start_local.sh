#!/usr/bin/env bash
# Start frontend, backend, and WAF AI/ML service locally.
# Uses project .venv so PyJWT and python-multipart are available (no stub routes).
# WAF service uses real model if models/waf-distilbert exists, else placeholder.

set -e
cd "$(dirname "$0")/.."
ROOT="$PWD"

# Ensure venv exists and has required deps (fixes PyJWT / python-multipart issues)
if [[ ! -d "$ROOT/.venv" ]]; then
  echo "Creating .venv and installing dependencies..."
  python -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/pip" install -q --upgrade pip
  "$ROOT/.venv/bin/pip" install -q 'PyJWT>=2.8.0' 'python-multipart>=0.0.6'
  "$ROOT/.venv/bin/pip" install -q -r requirements.txt
fi

# Ensure dirs exist
mkdir -p logs data

PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3
echo "Using Python: $PY"

# Start 3 web apps (ports 8080, 8081, 8082) for WAF testing
if bash "$ROOT/scripts/start_apps.sh"; then
  sleep 2
fi

# Start backend (port 3001)
echo "Starting backend on http://localhost:3001 ..."
"$PY" -m uvicorn backend.main:app --host 0.0.0.0 --port 3001 --reload &
BACKEND_PID=$!

# Start WAF service (port 8000). Uses real model if present; --workers 1 required.
echo "Starting WAF AI/ML service on http://localhost:8000 ..."
"$PY" scripts/start_waf_service.py --host 0.0.0.0 --port 8000 --workers 1 &
WAF_PID=$!

# Start frontend (port 3000)
echo "Starting frontend on http://localhost:3000 ..."
(cd frontend && npm run dev) &
FRONTEND_PID=$!

echo ""
echo "Backend services and 3 web apps started."
echo "  Frontend:   http://localhost:3000"
echo "  Backend:    http://localhost:3001"
echo "  WAF (ML):   http://localhost:8000"
echo "  Web App 1:  http://localhost:8080"
echo "  Web App 2:  http://localhost:8081"
echo "  Web App 3:  http://localhost:8082"
echo ""
echo "PIDs: backend=$BACKEND_PID waf=$WAF_PID frontend=$FRONTEND_PID"
echo "To stop: kill $BACKEND_PID $WAF_PID $FRONTEND_PID"
echo "To stop everything (including web apps): ./scripts/stop_all.sh"
echo ""
wait
