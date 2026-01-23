#!/bin/bash
# Check status of all real applications

echo "=========================================="
echo "Real Applications Status Check"
echo "=========================================="

# App 1: Juice Shop
echo ""
echo "App 1: OWASP Juice Shop (Port 8080)"
if pgrep -f "juice-shop" > /dev/null; then
    echo "  Process: ✅ Running (PID: $(pgrep -f 'juice-shop' | head -1))"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo "  HTTP Status: ✅ 200 OK"
    else
        echo "  HTTP Status: ⚠️ $HTTP_CODE (may still be starting)"
    fi
else
    echo "  Process: ❌ Not running"
fi

# App 2: WebGoat
echo ""
echo "App 2: OWASP WebGoat (Port 8081)"
if pgrep -f "webgoat" > /dev/null; then
    echo "  Process: ✅ Running (PID: $(pgrep -f 'webgoat' | head -1))"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8081/WebGoat 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "302" ]; then
        echo "  HTTP Status: ✅ $HTTP_CODE"
    else
        echo "  HTTP Status: ⚠️ $HTTP_CODE (may still be starting)"
    fi
else
    JAR_FILE=$(find applications/app2-webgoat/webgoat-container/target -name "webgoat-*.jar" 2>/dev/null | head -1)
    if [ -n "$JAR_FILE" ]; then
        echo "  Process: ❌ Not running (JAR built: ✅)"
    else
        echo "  Process: ❌ Not running (JAR: ⚠️ Not built - run: bash scripts/build_webgoat.sh)"
    fi
fi

# App 3: DVWA
echo ""
echo "App 3: DVWA (Port 8082)"
if pgrep -f "php.*8082" > /dev/null; then
    echo "  Process: ✅ Running (PID: $(pgrep -f 'php.*8082' | head -1))"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8082 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "302" ]; then
        echo "  HTTP Status: ✅ $HTTP_CODE"
    else
        echo "  HTTP Status: ⚠️ $HTTP_CODE"
    fi
else
    echo "  Process: ❌ Not running"
fi

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "To start all applications:"
echo "  bash scripts/start_real_apps_simple.sh"
echo ""
echo "To check logs:"
echo "  tail -f logs/juice-shop.log"
echo "  tail -f logs/webgoat.log"
echo "  tail -f logs/dvwa.log"
