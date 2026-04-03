#!/usr/bin/env bash
# Run attacks against 3 "web app" endpoints (WAF test targets) and verify dashboard charts in real time.
# Prerequisites: Backend (port 3001) and Frontend (port 3000) should be running.
# Usage: ./scripts/run_attacks_and_verify_charts.sh [duration_sec] [req_per_sec]

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DURATION="${1:-90}"
RATE="${2:-4}"

echo "=============================================="
echo "  WAF Attack Run  3 App Targets + Charts    "
echo "=============================================="
echo ""
echo "This script sends mixed benign + attack traffic to 3 test endpoints"
echo "that go through WAF middleware (traffic and threats will appear in the dashboard)."
echo ""
echo "  Target endpoints (simulating 3 web apps):"
echo "    - /test/endpoint  (app1)"
echo "    - /test/login     (app2)"
echo "    - /test/search    (app3)"
echo ""
echo "  Duration: ${DURATION}s  |  Rate: ${RATE} req/s"
echo "  Dashboard: http://localhost:3000/dashboard"
echo "  Use time range: Last 1 hour or Last 24 hours to see both graphs update."
echo ""

# Check backend
if ! curl -s -o /dev/null -w "%{http_code}" http://localhost:3001/health 2>/dev/null | grep -q 200; then
  echo "ERROR: Backend not responding at http://localhost:3001"
  echo "Start it with: python -m uvicorn backend.main:app --host 0.0.0.0 --port 3001 --reload"
  exit 1
fi
echo "Backend is up."

# Run traffic generator (hits /test/* so WAF middleware logs traffic + threats)
echo ""
echo "Starting traffic generator..."
python3 scripts/generate_live_traffic.py "$DURATION" "$RATE"

echo ""
echo "Done. Open http://localhost:3000/dashboard and check Request Volume & Threats and Top 10 Threat Types."
