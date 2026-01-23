#!/bin/bash
# Start all real web applications

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPS_DIR="$PROJECT_DIR/applications"
LOGS_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOGS_DIR"

echo "Starting real web applications..."

# Stop any existing instances
pkill -f "juice-shop" 2>/dev/null || true
pkill -f "webgoat" 2>/dev/null || true
pkill -f "dvwa" 2>/dev/null || true
pkill -f "simple_web_apps.py" 2>/dev/null || true

sleep 2

# Start App 1: Juice Shop (Port 8080)
echo "Starting Juice Shop on port 8080..."
cd "$APPS_DIR/app1-juice-shop"
export PORT=8080
nohup npm start > "$LOGS_DIR/juice-shop.log" 2>&1 &
echo $! > /tmp/juice-shop.pid
echo "✓ Juice Shop started (PID: $(cat /tmp/juice-shop.pid))"

# Start App 2: WebGoat (Port 8081)
echo "Starting WebGoat on port 8081..."
cd "$APPS_DIR/app2-webgoat"
JAR_FILE=$(find webgoat-container/target -name "webgoat-*.jar" 2>/dev/null | head -1)
if [ -z "$JAR_FILE" ]; then
    echo "⚠ Warning: WebGoat JAR not found. Building..."
    if [ -f "./mvnw" ]; then
        ./mvnw clean package -DskipTests -q
    else
        mvn clean package -DskipTests -q
    fi
    JAR_FILE=$(find webgoat-container/target -name "webgoat-*.jar" | head -1)
fi

if [ -n "$JAR_FILE" ]; then
    export JAVA_HOME=${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk}
    nohup java -jar "$JAR_FILE" --server.port=8081 --webgoat.port=8081 --webwolf.port=9091 > "$LOGS_DIR/webgoat.log" 2>&1 &
    echo $! > /tmp/webgoat.pid
    echo "✓ WebGoat started (PID: $(cat /tmp/webgoat.pid))"
else
    echo "✗ Error: Could not find or build WebGoat JAR"
fi

# Start App 3: DVWA (Port 8082)
echo "Starting DVWA on port 8082..."
cd "$APPS_DIR/app3-dvwa"
nohup php -S localhost:8082 > "$LOGS_DIR/dvwa.log" 2>&1 &
echo $! > /tmp/dvwa.pid
echo "✓ DVWA started (PID: $(cat /tmp/dvwa.pid))"

sleep 5

echo ""
echo "Applications starting..."
echo "  - Juice Shop: http://localhost:8080 (waiting for startup...)"
echo "  - WebGoat: http://localhost:8081/WebGoat (waiting for startup...)"
echo "  - DVWA: http://localhost:8082 (waiting for startup...)"
echo ""
echo "Check logs:"
echo "  tail -f $LOGS_DIR/juice-shop.log"
echo "  tail -f $LOGS_DIR/webgoat.log"
echo "  tail -f $LOGS_DIR/dvwa.log"
