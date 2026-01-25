#!/bin/bash
# Start 3 Web Applications using Docker

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "Starting 3 Web Applications (Docker)"
echo "=========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Stop existing containers if running
echo "Stopping existing containers (if any)..."
docker-compose -f docker-compose.webapps.yml down 2>/dev/null || true
echo ""

# Start containers
echo "Starting containers..."
docker-compose -f docker-compose.webapps.yml up -d

echo ""
echo "Waiting for applications to start..."
sleep 10

# Check status
echo ""
echo "=========================================="
echo "Application Status"
echo "=========================================="

check_app() {
    local name=$1
    local url=$2
    local port=$3
    
    if curl -s "$url" > /dev/null 2>&1; then
        echo "✅ $name: Running (http://localhost:$port)"
        return 0
    else
        echo "⏳ $name: Starting... (http://localhost:$port)"
        return 1
    fi
}

check_app "Juice Shop" "http://localhost:8080" "8080"
check_app "WebGoat" "http://localhost:8081" "8081"
check_app "DVWA" "http://localhost:8082" "8082"

echo ""
echo "=========================================="
echo "Docker Containers"
echo "=========================================="
docker-compose -f docker-compose.webapps.yml ps

echo ""
echo "=========================================="
echo "Access URLs"
echo "=========================================="
echo "  Juice Shop: http://localhost:8080"
echo "  WebGoat:    http://localhost:8081"
echo "  DVWA:       http://localhost:8082"
echo ""
echo "To view logs:"
echo "  docker-compose -f docker-compose.webapps.yml logs -f"
echo ""
echo "To stop:"
echo "  docker-compose -f docker-compose.webapps.yml down"
echo ""
