#!/bin/bash
# Start All Services: Frontend + Backend + 3 Web Apps

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "Starting WAF Platform - All Services"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# PID file directory
PID_DIR="/tmp/waf_pids"
mkdir -p "$PID_DIR"

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to wait for service
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=0
    
    echo -n "Waiting for $name..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
        echo -n "."
    done
    echo -e " ${RED}✗${NC} (timeout)"
    return 1
}

# 1. Start 3 Web Applications
echo "Step 1/4: Starting 3 Web Applications..."
if check_port 8080 || check_port 8081 || check_port 8082; then
    echo -e "${YELLOW}⚠ Some web app ports (8080, 8081, 8082) are already in use${NC}"
    echo "  Skipping web apps startup (may already be running)"
else
    if [ -f "scripts/start_apps.sh" ]; then
        bash scripts/start_apps.sh
        echo "  Waiting for apps to start..."
        sleep 3
    else
        echo -e "${YELLOW}⚠ start_apps.sh not found, skipping web apps${NC}"
    fi
fi
echo ""

# 2. Start Backend API Server
echo "Step 2/4: Starting Backend API Server (port 3001)..."
if check_port 3001; then
    echo -e "${YELLOW}⚠ API server (port 3001) is already running${NC}"
else
    cd "$PROJECT_DIR"
    nohup python3 scripts/start_api_server.py > logs/api_server.log 2>&1 &
    echo $! > "$PID_DIR/api_server.pid"
    echo -e "  ${GREEN}✓${NC} API server started (PID: $(cat $PID_DIR/api_server.pid))"
    wait_for_service "http://localhost:3001/health" "API Server"
fi
echo ""

# 3. Start WAF Service (if separate)
echo "Step 3/4: Starting WAF Service (port 8000)..."
if check_port 8000; then
    echo -e "${YELLOW}⚠ WAF service (port 8000) is already running${NC}"
else
    cd "$PROJECT_DIR"
    nohup python3 scripts/start_waf_service.py > logs/waf_service.log 2>&1 &
    echo $! > "$PID_DIR/waf_service.pid"
    echo -e "  ${GREEN}✓${NC} WAF service started (PID: $(cat $PID_DIR/waf_service.pid))"
    wait_for_service "http://localhost:8000/health" "WAF Service"
fi
echo ""

# 4. Start Frontend
echo "Step 4/4: Starting Frontend (port 3000)..."
if check_port 3000; then
    echo -e "${YELLOW}⚠ Frontend (port 3000) is already running${NC}"
else
    cd "$PROJECT_DIR/frontend"
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo "  Installing frontend dependencies..."
        npm install
    fi
    
    # Start Next.js dev server
    nohup npm run dev > ../logs/frontend.log 2>&1 &
    echo $! > "$PID_DIR/frontend.pid"
    echo -e "  ${GREEN}✓${NC} Frontend started (PID: $(cat $PID_DIR/frontend.pid))"
    echo "  Waiting for frontend to compile..."
    sleep 5
    wait_for_service "http://localhost:3000" "Frontend"
fi
echo ""

# Summary
echo "=========================================="
echo "All Services Started!"
echo "=========================================="
echo ""
echo "Services:"
echo "  - Frontend:     http://localhost:3000"
echo "  - API Server:   http://localhost:3001"
echo "  - WAF Service:  http://localhost:8000"
echo "  - Web App 1:    http://localhost:8080 (Juice Shop)"
echo "  - Web App 2:    http://localhost:8081 (WebGoat)"
echo "  - Web App 3:    http://localhost:8082 (DVWA)"
echo ""
echo "PID Files: $PID_DIR"
echo "Logs: logs/"
echo ""
echo "To stop all services:"
echo "  ./scripts/stop_all.sh"
echo ""
