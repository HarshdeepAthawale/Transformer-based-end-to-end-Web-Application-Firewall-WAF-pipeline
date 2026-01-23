#!/bin/bash
# Start applications with fixes applied

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPS_DIR="$PROJECT_DIR/applications"
LOGS_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOGS_DIR"

echo "Starting applications with fixes..."

# Stop old processes
pkill -f "juice-shop" 2>/dev/null || true
pkill -f "webgoat" 2>/dev/null || true
pkill -f "simple_web_apps.py" 2>/dev/null || true
sleep 2

# Start App 1: Juice Shop (with Node.js workaround)
echo ""
echo "Starting Juice Shop on port 8080..."
cd "$APPS_DIR/app1-juice-shop"

# Apply Node.js compatibility workaround
export NODE_OPTIONS="--no-warnings --experimental-specifier-resolution=node"
export PORT=8080

# Check if build exists
if [ ! -f "build/app.js" ]; then
    echo "  Building Juice Shop server..."
    npm run build:server 2>&1 | tail -5
fi

if [ -f "build/app.js" ]; then
    nohup node build/app.js > "$LOGS_DIR/juice-shop.log" 2>&1 &
    JSHOP_PID=$!
    echo $JSHOP_PID > /tmp/juice-shop.pid
    echo "✓ Juice Shop started (PID: $JSHOP_PID)"
    echo "  Wait 30-60 seconds for startup"
    sleep 10
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080 2>/dev/null | grep -q "200\|302"; then
        echo "  ✓ Juice Shop is responding"
    else
        echo "  ⚠ Juice Shop starting (may take time, check logs)"
    fi
else
    echo "✗ Error: build/app.js not found"
fi

# Start App 2: WebGoat (with Java 25)
echo ""
echo "Starting WebGoat on port 8081..."
cd "$APPS_DIR/app2-webgoat"

# Set Java 25
export JAVA_HOME=/usr/lib/jvm/java-25-openjdk

# Find or build JAR
JAR_FILE=$(find webgoat-container/target -name "webgoat-*.jar" 2>/dev/null | head -1)

if [ -z "$JAR_FILE" ]; then
    echo "  Building WebGoat with Java 25 (this will take 5-10 minutes)..."
    if [ -d "$JAVA_HOME" ]; then
        $JAVA_HOME/bin/java -version 2>&1 | head -1
        echo "  Building (please wait)..."
        ./mvnw clean package -DskipTests 2>&1 | tail -15
        JAR_FILE=$(find webgoat-container/target -name "webgoat-*.jar" | head -1)
        if [ -n "$JAR_FILE" ]; then
            echo "  ✓ Build complete: $JAR_FILE"
        fi
    else
        echo "  ✗ Java 25 not found at $JAVA_HOME"
        echo "  Install with: sudo pacman -S jdk25-openjdk"
    fi
fi

if [ -n "$JAR_FILE" ] && [ -f "$JAR_FILE" ]; then
    nohup $JAVA_HOME/bin/java -jar "$JAR_FILE" --server.port=8081 --webgoat.port=8081 --webwolf.port=9091 > "$LOGS_DIR/webgoat.log" 2>&1 &
    WGOAT_PID=$!
    echo $WGOAT_PID > /tmp/webgoat.pid
    echo "✓ WebGoat started (PID: $WGOAT_PID)"
    echo "  Wait 60-90 seconds for startup"
    sleep 15
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:8081/WebGoat 2>/dev/null | grep -q "200\|302"; then
        echo "  ✓ WebGoat is responding"
    else
        echo "  ⚠ WebGoat starting (may take time, check logs)"
    fi
else
    echo "✗ Error: WebGoat JAR not found. Build failed or not completed."
fi

# Start App 3: DVWA
echo ""
echo "Starting DVWA on port 8082..."
cd "$APPS_DIR/app3-dvwa"
nohup php -S localhost:8082 -t . > "$LOGS_DIR/dvwa.log" 2>&1 &
DVWA_PID=$!
echo $DVWA_PID > /tmp/dvwa.pid
echo "✓ DVWA started (PID: $DVWA_PID)"

echo ""
echo "=========================================="
echo "Applications Status"
echo "=========================================="
sleep 5
bash "$PROJECT_DIR/scripts/final_apps_status.sh"
