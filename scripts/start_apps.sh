#!/bin/bash
# Start all web applications (Tomcat or Python-based)

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOMCAT_DIR="/opt/tomcat9"

echo "Starting web applications..."

# Check if Java/Tomcat is available
if [ -d "$TOMCAT_DIR" ] && [ -f "$TOMCAT_DIR/bin/startup.sh" ]; then
    echo "Starting Tomcat applications..."
    
    # Check if Tomcat is already running
    if pgrep -f "catalina" > /dev/null; then
        echo "Tomcat is already running"
    else
        "$TOMCAT_DIR/bin/startup.sh"
        echo "Waiting for Tomcat to start..."
        sleep 5
        echo "✓ Tomcat started"
    fi
    
    # Verify applications
    for port in 8080 8081 8082; do
        if curl -s "http://localhost:$port" > /dev/null 2>&1; then
            echo "✓ App on port $port is responding"
        else
            echo "⚠ App on port $port may not be ready yet"
        fi
    done
else
    echo "Tomcat not found. Starting Python-based applications..."
    cd "$PROJECT_DIR"
    
    # Check if Python apps are already running
    if pgrep -f "simple_web_apps.py" > /dev/null; then
        echo "Python applications are already running"
    else
        # Start Python apps in background
        nohup python3 scripts/simple_web_apps.py > logs/web_apps.log 2>&1 &
        echo $! > /tmp/waf_web_apps.pid
        echo "✓ Python applications started (PID: $(cat /tmp/waf_web_apps.pid))"
        sleep 2
        
        # Verify applications
        for port in 8080 8081 8082; do
            if curl -s "http://localhost:$port" > /dev/null 2>&1; then
                echo "✓ App on port $port is responding"
            else
                echo "⚠ App on port $port may not be ready yet"
            fi
        done
    fi
fi

echo ""
echo "Applications are running on:"
echo "  - App 1: http://localhost:8080"
echo "  - App 2: http://localhost:8081"
echo "  - App 3: http://localhost:8082"
