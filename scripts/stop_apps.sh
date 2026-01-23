#!/bin/bash
# Stop all web applications

set -e

TOMCAT_DIR="/opt/tomcat9"

echo "Stopping web applications..."

# Stop Tomcat if running
if [ -f "$TOMCAT_DIR/bin/shutdown.sh" ]; then
    if pgrep -f "catalina" > /dev/null; then
        "$TOMCAT_DIR/bin/shutdown.sh"
        echo "✓ Tomcat stopped"
    else
        echo "Tomcat is not running"
    fi
fi

# Stop Python apps if running
if [ -f /tmp/waf_web_apps.pid ]; then
    PID=$(cat /tmp/waf_web_apps.pid)
    if ps -p "$PID" > /dev/null 2>&1; then
        kill "$PID" 2>/dev/null || true
        echo "✓ Python applications stopped"
    fi
    rm -f /tmp/waf_web_apps.pid
fi

# Also kill any remaining Python web app processes
pkill -f "simple_web_apps.py" 2>/dev/null && echo "✓ Cleaned up Python web app processes" || true

echo "All applications stopped"
