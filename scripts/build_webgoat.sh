#!/bin/bash
# Build WebGoat with Java 25

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR/applications/app2-webgoat"

# Set JAVA_HOME to Java 25
export JAVA_HOME=/usr/lib/jvm/java-25-openjdk

echo "Building WebGoat with Java 25..."
echo "JAVA_HOME: $JAVA_HOME"
$JAVA_HOME/bin/java -version

# Build WebGoat
./mvnw clean package -DskipTests

echo ""
echo "✓ WebGoat build complete!"
echo "JAR location: $(find webgoat-container/target -name "webgoat-*.jar" | head -1)"
