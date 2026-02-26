#!/usr/bin/env bash
# Start the 3 local web apps (App1, App2, App3) on ports 8080, 8081, 8082.
# Used by start_all.sh and start_local.sh for WAF reverse-proxy testing.

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PID_DIR="${PID_DIR:-/tmp/waf_pids}"
mkdir -p "$PID_DIR"
mkdir -p logs

check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

if check_port 8080 || check_port 8081 || check_port 8082; then
    echo "  Ports 8080, 8081 or 8082 already in use; skipping web apps."
    exit 0
fi

PY="${PROJECT_DIR}/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3

nohup "$PY" "$PROJECT_DIR/scripts/simple_web_apps.py" >> "$PROJECT_DIR/logs/web_apps.log" 2>&1 &
echo $! > "$PID_DIR/web_apps.pid"
echo "  3 web apps started (ports 8080, 8081, 8082) — PID $(cat "$PID_DIR/web_apps.pid")"
