#!/bin/bash
# DoS/DDoS Attack Simulation - runs stress test against WAF gateway (no Docker)
# Prerequisites: Redis, Python 3 with requests
# Usage: ./scripts/run_dos_attack_simulation.sh
#
# Before running: start Redis (redis-server) and ensure backend is running.
# Or run: ./scripts/run_dos_attack_simulation.sh --with-services  (starts backend + gateway)

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GATEWAY_PORT=8080
BACKEND_PORT=3001
UPSTREAM_PORT=9999

echo "=============================================="
echo "  DoS/DDoS Attack Simulation (no Docker)"
echo "=============================================="

# Check Redis
if ! redis-cli ping 2>/dev/null | grep -q PONG; then
  echo "ERROR: Redis is not running. Start it with: redis-server"
  exit 1
fi
echo "[OK] Redis is running"

WITH_SERVICES=""
[ "$1" = "--with-services" ] && WITH_SERVICES=1

if [ -n "$WITH_SERVICES" ]; then
  # Start minimal upstream
  if ! curl -s -o /dev/null "http://127.0.0.1:$UPSTREAM_PORT/" 2>/dev/null; then
    echo "[1/3] Starting upstream on port $UPSTREAM_PORT..."
    python3 -m http.server $UPSTREAM_PORT --bind 127.0.0.1 &
    UPSTREAM_PID=$!
    sleep 1
  fi

  # Start backend
  if ! curl -s -o /dev/null "http://127.0.0.1:$BACKEND_PORT/health" 2>/dev/null; then
    echo "[2/3] Starting backend on port $BACKEND_PORT..."
    python -m uvicorn backend.main:app --host 0.0.0.0 --port $BACKEND_PORT &
    sleep 4
  fi

  # Start gateway
  if ! curl -s -o /dev/null "http://127.0.0.1:$GATEWAY_PORT/" 2>/dev/null; then
    echo "[3/3] Starting gateway on port $GATEWAY_PORT..."
    REDIS_URL=redis://localhost:6379 \
    UPSTREAM_URL=http://127.0.0.1:$UPSTREAM_PORT \
    BACKEND_EVENTS_URL=http://127.0.0.1:$BACKEND_PORT/api/events/ingest \
    BACKEND_EVENTS_ENABLED=true \
    python -m uvicorn gateway.main:app --host 0.0.0.0 --port $GATEWAY_PORT &
    echo "      Waiting for gateway startup (WAF model load)..."
    sleep 8
  fi
fi

# Verify gateway is reachable
if ! curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$GATEWAY_PORT/" 2>/dev/null | grep -qE '200|429|502'; then
  echo ""
  echo "ERROR: Gateway not reachable at http://localhost:$GATEWAY_PORT"
  echo "Start it with:"
  echo "  REDIS_URL=redis://localhost:6379 UPSTREAM_URL=http://127.0.0.1:9999 \\"
  echo "  BACKEND_EVENTS_URL=http://127.0.0.1:3001/api/events/ingest \\"
  echo "  python -m uvicorn gateway.main:app --host 0.0.0.0 --port $GATEWAY_PORT"
  exit 1
fi

echo ""
echo "Running stress test..."
STRESS_TEST_BASE_URL=http://localhost:$GATEWAY_PORT python scripts/stress_test_rate_limit.py

echo ""
echo "=============================================="
echo "  View results:"
echo "   Dashboard: http://localhost:3000/dos-protection"
echo "   API: curl http://localhost:3001/api/events/dos-overview?range=1h"
echo "=============================================="
