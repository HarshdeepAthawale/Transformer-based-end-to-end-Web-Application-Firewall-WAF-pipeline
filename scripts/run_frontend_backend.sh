#!/bin/bash
# Run Frontend + Backend only
# Backend API: http://localhost:3001
# Frontend:    http://localhost:3000

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PID_DIR="${PID_DIR:-/tmp/waf_pids}"
mkdir -p "$PID_DIR"
mkdir -p logs
mkdir -p data

check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

wait_for() {
    local url=$1
    local name=$2
    local max=${3:-60}
    local n=0
    echo -n "Waiting for $name..."
    while [ $n -lt $max ]; do
        if curl -sfL -o /dev/null "$url" 2>/dev/null; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        n=$((n + 1))
        sleep 1
        echo -n "."
    done
    echo -e " ${RED}✗${NC} (timeout)"
    return 1
}

echo "=========================================="
echo "  WAF Platform – Frontend + Backend + DB"
echo "=========================================="
echo ""

# 0. Database (SQLite)
echo "0. Database (SQLite)"
echo "   data/ ensured; backend will init data/waf_dashboard.db on startup"
echo ""

# 1. Backend API (port 3001)
echo "1. Backend API (port 3001)"
if check_port 3001; then
    echo -e "   ${YELLOW}Port 3001 in use – skipping (may already be running)${NC}"
else
    PYTHON="${PYTHON:-python}"
    nohup "$PYTHON" scripts/start_api_server.py >> logs/api_server.log 2>&1 &
    echo $! > "$PID_DIR/api_server.pid"
    echo -e "   ${GREEN}✓${NC} Started (PID: $(cat "$PID_DIR/api_server.pid"))"
    echo "   (Backend loads DB + WAF model; may take 30–90s)"
    sleep 5
    if ! wait_for "http://localhost:3001/health" "Backend" 120; then
        echo -e "   ${RED}Tip: check logs/api_server.log for errors${NC}"
        exit 1
    fi
fi
echo ""

# 2. Frontend (port 3000)
echo "2. Frontend (port 3000)"
if check_port 3000; then
    echo -e "   ${YELLOW}Port 3000 in use – skipping (may already be running)${NC}"
else
    if [ ! -d "frontend/node_modules" ]; then
        echo "   Installing frontend dependencies..."
        (cd frontend && npm install)
    fi
    (cd frontend && nohup npm run dev >> ../logs/frontend.log 2>&1 &)
    # npm run dev spawns child; pgrep next is more reliable than $!
    sleep 2
    NEXT_PID=$(pgrep -f "next dev" | head -1)
    if [ -n "$NEXT_PID" ]; then
        echo "$NEXT_PID" > "$PID_DIR/frontend.pid"
        echo -e "   ${GREEN}✓${NC} Started (PID: $NEXT_PID)"
    else
        echo -e "   ${YELLOW}Started (PID not captured)${NC}"
    fi
    echo "   Waiting for frontend to compile..."
    sleep 5
    wait_for "http://localhost:3000" "Frontend" 90
fi
echo ""

echo "=========================================="
echo "  Services"
echo "=========================================="
echo ""
echo "  Frontend:  http://localhost:3000"
echo "  Backend:   http://localhost:3001"
echo "  Database:  data/waf_dashboard.db (SQLite, init by backend)"
echo ""
echo "  Logs:      logs/api_server.log  logs/frontend.log"
echo "  PIDs:      $PID_DIR"
echo ""
echo "  Stop:      ./scripts/stop_frontend_backend.sh"
echo "            (or: pkill -f start_api_server; pkill -f 'next dev')"
echo ""
