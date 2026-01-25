#!/bin/bash
# Stop All Services

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="/tmp/waf_pids"

echo "=========================================="
echo "Stopping WAF Platform - All Services"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Stop services by PID files
if [ -d "$PID_DIR" ]; then
    for pid_file in "$PID_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            service_name=$(basename "$pid_file" .pid)
            pid=$(cat "$pid_file")
            
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "Stopping $service_name (PID: $pid)..."
                kill "$pid" 2>/dev/null || true
                rm "$pid_file"
                echo -e "  ${GREEN}✓${NC} Stopped"
            else
                echo "Service $service_name (PID: $pid) not running"
                rm "$pid_file"
            fi
        fi
    done
fi

# Stop by process name (fallback)
echo ""
echo "Stopping remaining processes..."

# Frontend (Next.js)
pkill -f "next dev" 2>/dev/null && echo -e "  ${GREEN}✓${NC} Frontend stopped" || echo "  No frontend process found"

# API Server
pkill -f "start_api_server.py" 2>/dev/null && echo -e "  ${GREEN}✓${NC} API server stopped" || echo "  No API server process found"

# WAF Service
pkill -f "start_waf_service.py" 2>/dev/null && echo -e "  ${GREEN}✓${NC} WAF service stopped" || echo "  No WAF service process found"

# Web Apps
pkill -f "simple_web_apps.py" 2>/dev/null && echo -e "  ${GREEN}✓${NC} Web apps stopped" || echo "  No web apps process found"

# Tomcat (if running)
if pgrep -f "catalina" > /dev/null; then
    echo "Stopping Tomcat..."
    /opt/tomcat9/bin/shutdown.sh 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} Tomcat stopped"
fi

echo ""
echo "=========================================="
echo "All Services Stopped!"
echo "=========================================="
