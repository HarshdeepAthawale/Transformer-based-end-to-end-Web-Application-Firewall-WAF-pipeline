#!/bin/bash
# Check Status of All Services

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=========================================="
echo "WAF Platform - Service Status"
echo "=========================================="
echo ""

check_service() {
    local name=$1
    local url=$2
    local port=$3
    
    if curl -s "$url" > /dev/null 2>&1; then
        echo "✅ $name: Running (http://localhost:$port)"
        return 0
    else
        echo "❌ $name: Not running (port $port)"
        return 1
    fi
}

# Check services
check_service "Frontend" "http://localhost:3000" "3000"
check_service "API Server" "http://localhost:3001/health" "3001"
check_service "WAF Service" "http://localhost:8000/health" "8000"
check_service "Web App 1 (Juice Shop)" "http://localhost:8080" "8080"
check_service "Web App 2 (WebGoat)" "http://localhost:8081" "8081"
check_service "Web App 3 (DVWA)" "http://localhost:8082" "8082"

echo ""
echo "=========================================="
echo "Quick Access:"
echo "  Frontend:    http://localhost:3000"
echo "  API Docs:    http://localhost:3001/docs"
echo "  WAF Service: http://localhost:8000/docs"
echo "=========================================="
