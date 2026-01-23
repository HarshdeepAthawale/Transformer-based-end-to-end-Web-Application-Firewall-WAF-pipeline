#!/bin/bash
# Generate test traffic for all applications to populate logs

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$PROJECT_DIR/logs"

echo "=========================================="
echo "Generating Test Traffic for Log Analysis"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to make HTTP requests
make_request() {
    local url=$1
    local method=${2:-GET}
    local data=${3:-""}
    
    if [ "$method" = "GET" ]; then
        curl -s -o /dev/null -w "%{http_code}" "$url" > /dev/null 2>&1
    elif [ "$method" = "POST" ]; then
        curl -s -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json" -d "$data" "$url" > /dev/null 2>&1
    fi
}

# App 1: Juice Shop (Port 8080)
echo ""
echo -e "${YELLOW}Generating traffic for Juice Shop (Port 8080)...${NC}"
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080 | grep -q "200\|302"; then
    echo "  ✓ Juice Shop is responding"
    
    # Generate various requests
    for i in {1..20}; do
        make_request "http://localhost:8080/"
        make_request "http://localhost:8080/api/products"
        make_request "http://localhost:8080/api/products?limit=10"
        make_request "http://localhost:8080/api/products?limit=5&offset=$i"
        make_request "http://localhost:8080/api/categories"
        make_request "http://localhost:8080/api/rest/user/login"
        make_request "http://localhost:8080/#/search?q=test$i"
        make_request "http://localhost:8080/#/basket"
    done
    echo -e "  ${GREEN}✓ Generated 160 requests for Juice Shop${NC}"
else
    echo "  ⚠ Juice Shop not responding, skipping..."
fi

# App 2: WebGoat (Port 8081)
echo ""
echo -e "${YELLOW}Generating traffic for WebGoat (Port 8081)...${NC}"
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8081/WebGoat | grep -q "200\|302"; then
    echo "  ✓ WebGoat is responding"
    
    # Generate various requests
    for i in {1..15}; do
        make_request "http://localhost:8081/WebGoat"
        make_request "http://localhost:8081/WebGoat/login"
        make_request "http://localhost:8081/WebGoat/register"
        make_request "http://localhost:8081/WebGoat/attack"
        make_request "http://localhost:8081/WebGoat/start.mvc"
        make_request "http://localhost:8081/WebGoat/lesson?id=$i"
    done
    echo -e "  ${GREEN}✓ Generated 90 requests for WebGoat${NC}"
else
    echo "  ⚠ WebGoat not responding, skipping..."
fi

# App 3: DVWA (Port 8082)
echo ""
echo -e "${YELLOW}Generating traffic for DVWA (Port 8082)...${NC}"
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8082 | grep -q "200\|302"; then
    echo "  ✓ DVWA is responding"
    
    # Generate various requests
    for i in {1..25}; do
        make_request "http://localhost:8082/"
        make_request "http://localhost:8082/login.php"
        make_request "http://localhost:8082/index.php"
        make_request "http://localhost:8082/vulnerabilities/sqli/?id=$i"
        make_request "http://localhost:8082/vulnerabilities/xss_r/?name=test$i"
        make_request "http://localhost:8082/vulnerabilities/exec/?ip=127.0.0.1"
        make_request "http://localhost:8082/vulnerabilities/upload/"
        make_request "http://localhost:8082/vulnerabilities/csrf/"
    done
    echo -e "  ${GREEN}✓ Generated 200 requests for DVWA${NC}"
else
    echo "  ⚠ DVWA not responding, skipping..."
fi

# Generate some malicious-looking requests for testing
echo ""
echo -e "${YELLOW}Generating suspicious/malicious test patterns...${NC}"
for app_port in 8080 8081 8082; do
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:$app_port | grep -q "200\|302"; then
        # SQL Injection attempts
        make_request "http://localhost:$app_port/api/users?id=1' OR '1'='1"
        make_request "http://localhost:$app_port/search?q=admin'--"
        make_request "http://localhost:$app_port/login?user=admin'/*"
        
        # XSS attempts
        make_request "http://localhost:$app_port/search?q=<script>alert(1)</script>"
        make_request "http://localhost:$app_port/comment?text=<img src=x onerror=alert(1)>"
        
        # Path traversal
        make_request "http://localhost:$app_port/../../etc/passwd"
        make_request "http://localhost:$app_port/....//....//etc/passwd"
        
        # Command injection
        make_request "http://localhost:$app_port/api/exec?cmd=ls"
        make_request "http://localhost:$app_port/api/ping?host=127.0.0.1;cat /etc/passwd"
        
        # Unusual headers
        curl -s -o /dev/null -H "User-Agent: sqlmap/1.0" "http://localhost:$app_port/"
        curl -s -o /dev/null -H "X-Forwarded-For: 192.168.1.100" "http://localhost:$app_port/"
        curl -s -o /dev/null -H "Authorization: Bearer invalid_token_12345" "http://localhost:$app_port/api/data"
    fi
done
echo -e "  ${GREEN}✓ Generated suspicious test patterns${NC}"

# Wait a moment for logs to be written
sleep 2

# Check log file
echo ""
echo -e "${YELLOW}Checking log files...${NC}"
if [ -f "/var/log/nginx/access.log" ]; then
    LOG_SIZE=$(echo "zenbook" | sudo -S wc -l /var/log/nginx/access.log 2>/dev/null | awk '{print $1}')
    echo "  ✓ Nginx access log: $LOG_SIZE lines"
    
    # Show last few entries
    echo ""
    echo "  Last 5 log entries:"
    echo "zenbook" | sudo -S tail -5 /var/log/nginx/access.log 2>/dev/null | sed 's/^/    /'
else
    echo "  ⚠ Nginx access log not found at /var/log/nginx/access.log"
fi

# Check application logs
echo ""
echo -e "${YELLOW}Checking application logs...${NC}"
if [ -f "$LOGS_DIR/juice-shop.log" ]; then
    echo "  ✓ Juice Shop log: $(wc -l < "$LOGS_DIR/juice-shop.log") lines"
fi
if [ -f "$LOGS_DIR/webgoat.log" ]; then
    echo "  ✓ WebGoat log: $(wc -l < "$LOGS_DIR/webgoat.log") lines"
fi
if [ -f "$LOGS_DIR/dvwa.log" ]; then
    echo "  ✓ DVWA log: $(wc -l < "$LOGS_DIR/dvwa.log") lines"
fi

echo ""
echo -e "${GREEN}=========================================="
echo "Test Traffic Generation Complete!"
echo "==========================================${NC}"
echo ""
echo "Summary:"
echo "  - Generated normal traffic patterns"
echo "  - Generated suspicious/malicious patterns"
echo "  - Logs populated for analysis"
echo ""
echo "Next steps:"
echo "  1. Check logs: tail -f /var/log/nginx/access.log"
echo "  2. Proceed to Phase 2: Log Ingestion System"
