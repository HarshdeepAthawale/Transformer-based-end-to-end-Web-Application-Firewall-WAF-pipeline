#!/bin/bash
# Check status of all Docker applications

echo "=========================================="
echo "Docker Applications Status"
echo "=========================================="

echo ""
echo "Container Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "NAME|juice-shop|webgoat|dvwa" || echo "No containers running"

echo ""
echo "HTTP Responses:"
for port in 8080 8081 8082; do
    CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port 2>/dev/null || echo "000")
    NAME=""
    case $port in
        8080) NAME="Juice Shop" ;;
        8081) NAME="WebGoat" ;;
        8082) NAME="DVWA" ;;
    esac
    if [ "$CODE" = "200" ] || [ "$CODE" = "302" ]; then
        echo "  ✓ $NAME ($port): HTTP $CODE"
    else
        echo "  ⚠ $NAME ($port): HTTP $CODE (may still be starting)"
    fi
done

echo ""
echo "Container Logs (last 3 lines each):"
echo "--- Juice Shop ---"
docker logs --tail 3 juice-shop-waf 2>/dev/null || echo "  Container not found"
echo ""
echo "--- WebGoat ---"
docker logs --tail 3 webgoat-waf 2>/dev/null || echo "  Container not found"
echo ""
echo "--- DVWA ---"
docker logs --tail 3 dvwa-waf 2>/dev/null || echo "  Container not found"
