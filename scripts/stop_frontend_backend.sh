#!/bin/bash
# Stop Frontend + Backend only

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="${PID_DIR:-/tmp/waf_pids}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Stopping Frontend + Backend..."
echo ""

# By PID files
for name in api_server frontend; do
    f="$PID_DIR/${name}.pid"
    if [ -f "$f" ]; then
        pid=$(cat "$f")
        if ps -p "$pid" >/dev/null 2>&1; then
            kill "$pid" 2>/dev/null || true
            echo -e "  ${GREEN}✓${NC} Stopped $name (PID: $pid)"
        fi
        rm -f "$f"
    fi
done

# Fallback: by process name
pkill -f "start_api_server.py" 2>/dev/null && echo -e "  ${GREEN}✓${NC} API server stopped" || true
pkill -f "next dev" 2>/dev/null && echo -e "  ${GREEN}✓${NC} Frontend stopped" || true

echo ""
echo "Done."
