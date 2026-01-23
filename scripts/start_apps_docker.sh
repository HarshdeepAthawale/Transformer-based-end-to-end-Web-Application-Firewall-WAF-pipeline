#!/bin/bash
# Start all applications using Docker

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=========================================="
echo "Starting All Applications with Docker"
echo "=========================================="

# Stop old containers
echo "Stopping old containers..."
docker stop juice-shop-waf webgoat-waf dvwa-waf 2>/dev/null || true
docker rm juice-shop-waf webgoat-waf dvwa-waf 2>/dev/null || true

# Stop native PHP process if running
pkill -f "php.*8082" 2>/dev/null || true

sleep 2

# Start App 1: Juice Shop
echo ""
echo "Starting Juice Shop (Docker) on port 8080..."
docker run -d -p 8080:3000 --name juice-shop-waf bkimminich/juice-shop
echo "✓ Juice Shop container started"
echo "  Wait 30-60 seconds for startup"
sleep 3

# Start App 2: WebGoat
echo ""
echo "Starting WebGoat (Docker) on port 8081..."
docker run -d -p 8081:8080 -p 9091:9090 --name webgoat-waf webgoat/webgoat
echo "✓ WebGoat container started"
echo "  Wait 60-90 seconds for startup"
sleep 3

# Start App 3: DVWA
echo ""
echo "Starting DVWA (Docker) on port 8082..."
docker run -d -p 8082:80 --name dvwa-waf ghcr.io/digininja/dvwa:latest
echo "✓ DVWA container started"
echo "  Wait 30-60 seconds for startup"
sleep 3

echo ""
echo "=========================================="
echo "All Applications Starting in Docker..."
echo "=========================================="
echo "  App 1 (Juice Shop): http://localhost:8080"
echo "  App 2 (WebGoat): http://localhost:8081/WebGoat"
echo "  App 3 (DVWA): http://localhost:8082"
echo ""
echo "Container Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "NAME|juice-shop|webgoat|dvwa"
echo ""
echo "Check logs:"
echo "  docker logs juice-shop-waf"
echo "  docker logs webgoat-waf"
echo "  docker logs dvwa-waf"
echo ""
echo "Wait 60-90 seconds for all applications to fully start"
