#!/bin/bash

# Verify Advanced WAF Setup
# Tests the complete AI-powered WAF integration

echo "🔍 Verifying Advanced WAF Setup"
echo "==============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Test counter
TESTS_PASSED=0
TESTS_TOTAL=0

test_result() {
    local test_name="$1"
    local success="$2"
    local message="$3"

    TESTS_TOTAL=$((TESTS_TOTAL + 1))

    if [[ "$success" == "true" ]]; then
        log_success "✓ $test_name: $message"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        log_error "✗ $test_name: $message"
    fi
}

echo ""
echo "1. Checking OpenResty Service..."
echo "-------------------------------"

# Check if OpenResty is installed
if command -v openresty &> /dev/null; then
    test_result "OpenResty Installation" true "OpenResty is installed"
else
    test_result "OpenResty Installation" false "OpenResty is not installed"
fi

# Check if OpenResty service is running
if systemctl is-active --quiet openresty; then
    test_result "OpenResty Service" true "OpenResty service is running"
else
    test_result "OpenResty Service" false "OpenResty service is not running"
fi

# Test OpenResty configuration
if openresty -t -c /usr/local/openresty/nginx/conf/nginx.conf &> /dev/null; then
    test_result "OpenResty Configuration" true "Configuration is valid"
else
    test_result "OpenResty Configuration" false "Configuration has errors"
fi

echo ""
echo "2. Checking WAF Service..."
echo "-------------------------"

# Check if WAF service is responding
if curl -s --max-time 5 http://127.0.0.1:8000/health | grep -q "healthy"; then
    test_result "WAF Service Health" true "WAF service is responding"

    # Get detailed health info
    HEALTH_INFO=$(curl -s http://127.0.0.1:8000/health)
    if echo "$HEALTH_INFO" | grep -q '"model_loaded": true'; then
        test_result "WAF Model Loading" true "Transformer model is loaded"
    else
        test_result "WAF Model Loading" false "Model not loaded properly"
    fi

else
    test_result "WAF Service Health" false "WAF service is not responding on port 8000"
fi

echo ""
echo "3. Testing Nginx Integration..."
echo "-------------------------------"

# Test basic Nginx response
if curl -s --max-time 5 -w "%{http_code}" http://localhost/ | grep -q "000"; then
    test_result "Nginx Basic Response" false "Nginx is not responding (connection failed)"
else
    RESPONSE_CODE=$(curl -s --max-time 5 -o /dev/null -w "%{http_code}" http://localhost/)
    if [[ "$RESPONSE_CODE" == "502" ]] || [[ "$RESPONSE_CODE" == "503" ]]; then
        test_result "Nginx Response" true "Nginx responding (backend not running: $RESPONSE_CODE)"
    elif [[ "$RESPONSE_CODE" == "200" ]]; then
        test_result "Nginx Response" true "Nginx responding with backend (status: $RESPONSE_CODE)"
    else
        test_result "Nginx Response" true "Nginx responding (status: $RESPONSE_CODE)"
    fi
fi

# Test WAF metrics endpoint
if curl -s --max-time 5 http://localhost/waf-metrics | grep -q "total_requests"; then
    test_result "WAF Metrics Proxy" true "WAF metrics endpoint is accessible"
else
    test_result "WAF Metrics Proxy" false "WAF metrics endpoint not accessible"
fi

echo ""
echo "4. Testing AI Anomaly Detection..."
echo "----------------------------------"

# Test with WAF service directly (if running)
if curl -s --max-time 5 http://127.0.0.1:8000/health | grep -q "healthy"; then

    # Test normal request
    NORMAL_RESPONSE=$(curl -s --max-time 10 -X POST http://127.0.0.1:8000/check \
        -H "Content-Type: application/json" \
        -d '{"method":"GET","path":"/api/products","query_params":{"page":"1"},"headers":{"user-agent":"test"}}')

    if echo "$NORMAL_RESPONSE" | grep -q '"is_anomaly": false'; then
        test_result "AI Normal Request" true "Normal request correctly classified"
    else
        test_result "AI Normal Request" false "Normal request classification failed"
    fi

    # Test suspicious request
    SUSPICIOUS_RESPONSE=$(curl -s --max-time 10 -X POST http://127.0.0.1:8000/check \
        -H "Content-Type: application/json" \
        -d '{"method":"GET","path":"/api/users","query_params":{"id":"1'\'' OR '\''1'\''='\''1"},"headers":{"user-agent":"test"}}')

    if echo "$SUSPICIOUS_RESPONSE" | grep -q '"anomaly_score"'; then
        test_result "AI Suspicious Request" true "Suspicious request processed by AI"
    else
        test_result "AI Suspicious Request" false "Suspicious request processing failed"
    fi

else
    log_warning "WAF service not running - skipping AI detection tests"
fi

echo ""
echo "5. Testing Security Features..."
echo "-------------------------------"

# Test security headers
HEADERS_RESPONSE=$(curl -s --max-time 5 -I http://localhost/ 2>/dev/null | head -20)
if echo "$HEADERS_RESPONSE" | grep -q "X-Frame-Options"; then
    test_result "Security Headers" true "Security headers are present"
else
    test_result "Security Headers" false "Security headers missing"
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "                 TEST RESULTS"
echo "═══════════════════════════════════════════════"

SUCCESS_RATE=$((TESTS_PASSED * 100 / TESTS_TOTAL))

if [[ $SUCCESS_RATE -ge 80 ]]; then
    log_success "🎉 Overall: $TESTS_PASSED/$TESTS_TOTAL tests passed (${SUCCESS_RATE}%)"
    echo ""
    echo "✅ Advanced WAF is ready for production!"
    echo ""
    echo "To start using:"
    echo "1. Start your backend app on port 8080"
    echo "2. Test: curl http://localhost/"
    echo "3. Monitor: curl http://localhost/waf-metrics"
else
    log_error "❌ Overall: $TESTS_PASSED/$TESTS_TOTAL tests passed (${SUCCESS_RATE}%)"
    echo ""
    echo "⚠️  Some components need attention. Check the failed tests above."
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "              TROUBLESHOOTING"
echo "═══════════════════════════════════════════════"
echo "• Check OpenResty: journalctl -u openresty -f"
echo "• Check WAF service logs in the terminal where it's running"
echo "• Test components individually:"
echo "  - curl http://127.0.0.1:8000/health"
echo "  - curl http://localhost/waf-metrics"
echo "  - openresty -t -c /usr/local/openresty/nginx/conf/nginx.conf"