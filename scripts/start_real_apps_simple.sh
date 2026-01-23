#!/bin/bash
# Simplified startup script for real applications (starts them one by one)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPS_DIR="$PROJECT_DIR/applications"
LOGS_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOGS_DIR"

echo "Starting real web applications..."

# Stop old Python apps
pkill -f "simple_web_apps.py" 2>/dev/null || true

# Start App 1: Juice Shop
echo "Starting Juice Shop on port 8080..."
cd "$APPS_DIR/app1-juice-shop"
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies first..."
    npm install
fi
export PORT=8080
nohup npm start > "$LOGS_DIR/juice-shop.log" 2>&1 &
JSHOP_PID=$!
echo $JSHOP_PID > /tmp/juice-shop.pid
echo "✓ Juice Shop started (PID: $JSHOP_PID)"
echo "  Wait 30-60 seconds for startup, then visit: http://localhost:8080"

# Start App 2: WebGoat
echo ""
echo "Starting WebGoat on port 8081..."
cd "$APPS_DIR/app2-webgoat"
JAR_FILE=$(find webgoat-container/target -name "webgoat-*.jar" 2>/dev/null | head -1)
if [ -z "$JAR_FILE" ]; then
    echo "Building WebGoat (this will take 5-10 minutes)..."
    ./mvnw clean package -DskipTests
    JAR_FILE=$(find webgoat-container/target -name "webgoat-*.jar" | head -1)
fi

if [ -n "$JAR_FILE" ]; then
    export JAVA_HOME=${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk}
    nohup java -jar "$JAR_FILE" --server.port=8081 --webgoat.port=8081 --webwolf.port=9091 > "$LOGS_DIR/webgoat.log" 2>&1 &
    WGOAT_PID=$!
    echo $WGOAT_PID > /tmp/webgoat.pid
    echo "✓ WebGoat started (PID: $WGOAT_PID)"
    echo "  Wait 60-90 seconds for startup, then visit: http://localhost:8081/WebGoat"
else
    echo "✗ Error: Could not find WebGoat JAR"
fi

# Start App 3: DVWA
echo ""
echo "Starting DVWA on port 8082..."
cd "$APPS_DIR/app3-dvwa"
nohup php -S localhost:8082 > "$LOGS_DIR/dvwa.log" 2>&1 &
DVWA_PID=$!
echo $DVWA_PID > /tmp/dvwa.pid
echo "✓ DVWA started (PID: $DVWA_PID)"
echo "  Visit: http://localhost:8082"
echo "  First time: Click 'Setup / Reset DB' button"

echo ""
echo "=========================================="
echo "Applications Starting..."
echo "=========================================="
echo "  App 1 (Juice Shop): http://localhost:8080"
echo "  App 2 (WebGoat): http://localhost:8081/WebGoat"
echo "  App 3 (DVWA): http://localhost:8082"
echo ""
echo "Check logs:"
echo "  tail -f $LOGS_DIR/juice-shop.log"
echo "  tail -f $LOGS_DIR/webgoat.log"
echo "  tail -f $LOGS_DIR/dvwa.log"
echo ""
echo "Note: Juice Shop and WebGoat take 30-90 seconds to fully start"
