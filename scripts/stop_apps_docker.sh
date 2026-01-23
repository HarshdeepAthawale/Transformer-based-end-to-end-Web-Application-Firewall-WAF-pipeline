#!/bin/bash
# Stop all applications (Docker containers)

set -e

echo "=========================================="
echo "Stopping All Docker Applications"
echo "=========================================="

# Stop Docker containers
echo "Stopping containers..."
docker stop juice-shop-waf webgoat-waf dvwa-waf 2>/dev/null && echo "✓ Stopped Docker containers" || echo "No containers to stop"

# Stop native processes (just in case)
pkill -f "php.*8082" 2>/dev/null || true
pkill -f "juice-shop" 2>/dev/null || true
pkill -f "webgoat" 2>/dev/null || true

echo ""
echo "All applications stopped"
echo ""
echo "To remove containers:"
echo "  docker rm juice-shop-waf webgoat-waf dvwa-waf"
