#!/bin/bash
# Fix issues with Juice Shop and WebGoat

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPS_DIR="$PROJECT_DIR/applications"

echo "=========================================="
echo "Fixing Application Issues"
echo "=========================================="

# Fix 1: Juice Shop - Node.js compatibility
echo ""
echo "Fixing Juice Shop (Node.js compatibility)..."
cd "$APPS_DIR/app1-juice-shop"

# Check Node version
NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
echo "Current Node.js version: $(node --version)"

if [ "$NODE_VERSION" -ge 25 ]; then
    echo "⚠ Node.js v25+ detected. Juice Shop may have compatibility issues."
    echo "  Option 1: Use Node.js 20 LTS (recommended)"
    echo "  Option 2: Try with NODE_OPTIONS workaround"
    
    # Try workaround
    export NODE_OPTIONS="--no-warnings --experimental-specifier-resolution=node"
    echo "  Applied NODE_OPTIONS workaround"
fi

# Fix 2: WebGoat - Java 25 setup
echo ""
echo "Fixing WebGoat (Java 25 setup)..."
cd "$APPS_DIR/app2-webgoat"

# Check if Java 25 is available
if [ -d "/usr/lib/jvm/java-25-openjdk" ]; then
    echo "✓ Java 25 found at /usr/lib/jvm/java-25-openjdk"
    export JAVA_HOME=/usr/lib/jvm/java-25-openjdk
    echo "  JAVA_HOME set to: $JAVA_HOME"
    
    # Check if JAR exists
    JAR_FILE=$(find webgoat-container/target -name "webgoat-*.jar" 2>/dev/null | head -1)
    if [ -z "$JAR_FILE" ]; then
        echo "  Building WebGoat with Java 25..."
        echo "  This will take 5-10 minutes..."
        $JAVA_HOME/bin/java -version
        ./mvnw clean package -DskipTests
        JAR_FILE=$(find webgoat-container/target -name "webgoat-*.jar" | head -1)
        if [ -n "$JAR_FILE" ]; then
            echo "  ✓ WebGoat JAR built: $JAR_FILE"
        else
            echo "  ✗ WebGoat build failed"
        fi
    else
        echo "  ✓ WebGoat JAR found: $JAR_FILE"
    fi
else
    echo "✗ Java 25 not found. Please install Java 25:"
    echo "  sudo pacman -S jdk25-openjdk"
fi

echo ""
echo "=========================================="
echo "Fix Summary"
echo "=========================================="
echo "Juice Shop:"
echo "  - Node.js compatibility issue identified"
echo "  - Try: Use Node.js 20 LTS or apply workaround"
echo ""
echo "WebGoat:"
echo "  - Java 25 configuration checked"
echo "  - Build status verified"
