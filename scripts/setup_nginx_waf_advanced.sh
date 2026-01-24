#!/bin/bash

# Advanced Nginx Setup with Full AI WAF Integration
# Uses OpenResty Lua for real-time anomaly detection

set -e

echo "Setting up advanced Nginx with full AI WAF integration..."

# Configuration variables
OPENRESTY_CONF_DIR="/usr/local/openresty/nginx/conf"
OPENRESTY_SITES_AVAILABLE="${OPENRESTY_CONF_DIR}/sites-available"
OPENRESTY_SITES_ENABLED="${OPENRESTY_CONF_DIR}/sites-enabled"
OPENRESTY_CONF_FILE="${OPENRESTY_CONF_DIR}/nginx.conf"
WAF_CONF_FILE="${OPENRESTY_SITES_AVAILABLE}/waf-integration"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

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

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (use sudo)"
    exit 1
fi

# Check if OpenResty is installed and running
if ! command -v openresty &> /dev/null; then
    log_error "OpenResty is not installed. Please run: sudo ./scripts/setup_openresty_arch.sh"
    exit 1
fi

if ! systemctl is-active --quiet openresty; then
    log_error "OpenResty service is not running. Starting it..."
    systemctl start openresty
    sleep 2
    if ! systemctl is-active --quiet openresty; then
        log_error "Failed to start OpenResty service"
        exit 1
    fi
fi

log_info "OpenResty is running"

# Create log directory
mkdir -p /var/log/openresty
chmod 755 /var/log/openresty

# Backup existing nginx.conf if it exists
if [[ -f "$OPENRESTY_CONF_FILE" ]]; then
    cp "$OPENRESTY_CONF_FILE" "${OPENRESTY_CONF_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    log_info "Backed up existing nginx.conf"
fi

# Copy advanced WAF configuration
log_info "Installing advanced WAF OpenResty configuration..."
cp "${SCRIPT_DIR}/nginx_waf.conf" "$WAF_CONF_FILE"

# Enable the site
if [[ -L "${OPENRESTY_SITES_ENABLED}/waf-integration" ]]; then
    rm "${OPENRESTY_SITES_ENABLED}/waf-integration"
fi
ln -sf "$WAF_CONF_FILE" "${OPENRESTY_SITES_ENABLED}/waf-integration"

# Create main nginx.conf for OpenResty
log_info "Creating main OpenResty configuration..."
cat > "$OPENRESTY_CONF_FILE" << 'EOF'
worker_processes auto;
worker_rlimit_nofile 65536;

events {
    worker_connections 65536;
    use epoll;
    multi_accept on;
}

http {
    include /usr/local/openresty/nginx/conf/mime.types;
    default_type application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for" '
                    '"$host" sn="$server_name" '
                    'rt=$request_time '
                    'ua="$upstream_addr" us="$upstream_status" '
                    'ut="$upstream_response_time" ul="$upstream_response_length" '
                    'cs=$upstream_cache_status';

    access_log /var/log/openresty/access.log main;
    error_log /var/log/openresty/error.log;

    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 100M;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;

    # Lua package path
    lua_package_path "/usr/local/openresty/lualib/?.lua;;";
    lua_package_cpath "/usr/local/openresty/lualib/?.so;;";

    # Include WAF site
    include /usr/local/openresty/nginx/conf/sites-enabled/*;
}
EOF

# Test configuration
log_info "Testing OpenResty configuration..."
if openresty -t -c "$OPENRESTY_CONF_FILE"; then
    log_success "OpenResty configuration is valid"
else
    log_error "OpenResty configuration has errors"
    exit 1
fi

# Reload OpenResty
log_info "Reloading OpenResty..."
systemctl reload openresty

# Wait a moment for reload
sleep 2

# Verify OpenResty is still running
if systemctl is-active --quiet openresty; then
    log_success "OpenResty reloaded successfully"
else
    log_error "OpenResty failed after reload"
    journalctl -u openresty --no-pager -n 20
    exit 1
fi

log_success "Advanced Nginx WAF integration setup complete!"
echo ""
echo "Configuration details:"
echo "  - OpenResty Config: $OPENRESTY_CONF_FILE"
echo "  - WAF Site Config: $WAF_CONF_FILE"
echo "  - WAF Service URL: http://127.0.0.1:8000"
echo "  - Access Log: /var/log/openresty/access.log"
echo "  - Error Log: /var/log/openresty/error.log"
echo ""
echo "AI-Powered Features:"
echo "  - Real-time anomaly detection using Transformer model"
echo "  - Lua scripting for request interception"
echo "  - Async WAF service calls"
echo "  - Rate limiting and security headers"
echo "  - Comprehensive request normalization"
echo ""
echo "To start the WAF service:"
echo "  cd $PROJECT_ROOT"
echo "  source venv/bin/activate"
echo "  python scripts/start_waf_service.py --host 127.0.0.1 --port 8000"
echo ""
echo "To test the setup:"
echo "  curl -v http://localhost/"
echo "  curl -v http://localhost/api/test"
echo ""
echo "To monitor:"
echo "  curl http://localhost/waf-metrics"
echo "  journalctl -u openresty -f"
echo ""
echo "Note: Make sure your backend application runs on port 8080"