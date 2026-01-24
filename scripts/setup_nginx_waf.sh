#!/bin/bash

# Setup Nginx with WAF Integration
# This script configures Nginx as a reverse proxy with WAF service integration

set -e

echo "Setting up Nginx with WAF integration..."

# Configuration variables
NGINX_CONF_DIR="/etc/nginx"
NGINX_SITES_AVAILABLE="${NGINX_CONF_DIR}/sites-available"
NGINX_SITES_ENABLED="${NGINX_CONF_DIR}/sites-enabled"
NGINX_CONF_FILE="${NGINX_CONF_DIR}/nginx.conf"
WAF_CONF_FILE="${NGINX_SITES_AVAILABLE}/waf-integration"
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

# Check if Nginx is installed
if ! command -v nginx &> /dev/null; then
    log_error "Nginx is not installed. Please install Nginx first:"
    echo "  Ubuntu/Debian: sudo apt install nginx"
    echo "  CentOS/RHEL: sudo yum install nginx"
    echo "  macOS: brew install nginx"
    exit 1
fi

# Check if OpenResty or lua-nginx-module is available
if ! nginx -V 2>&1 | grep -q "lua"; then
    log_warning "Nginx does not have Lua support. Installing OpenResty..."

    # Try to install OpenResty
    if command -v apt &> /dev/null; then
        # Ubuntu/Debian
        apt update
        apt install -y software-properties-common
        add-apt-repository -y ppa:openresty/ppa
        apt update
        apt install -y openresty
        # Replace nginx with openresty
        systemctl stop nginx || true
        systemctl disable nginx || true
        systemctl enable openresty
    elif command -v yum &> /dev/null; then
        # CentOS/RHEL
        yum install -y yum-utils
        yum-config-manager --add-repo https://openresty.org/package/centos/openresty.repo
        yum install -y openresty
    else
        log_error "Please install OpenResty manually for Lua support"
        exit 1
    fi
fi

log_info "Installing required Lua modules..."

# Install lua-resty-http for HTTP client
if command -v apt &> /dev/null; then
    apt install -y lua-resty-http
elif command -v yum &> /dev/null; then
    yum install -y lua-resty-http
else
    log_warning "Please install lua-resty-http manually"
fi

# Create log directory
mkdir -p /var/log/nginx
chmod 755 /var/log/nginx

# Backup existing nginx.conf
if [[ -f "$NGINX_CONF_FILE" ]]; then
    cp "$NGINX_CONF_FILE" "${NGINX_CONF_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    log_info "Backed up existing nginx.conf"
fi

# Copy WAF configuration
log_info "Installing WAF Nginx configuration..."
cp "${SCRIPT_DIR}/nginx_waf.conf" "$WAF_CONF_FILE"

# Enable the site
if [[ -L "${NGINX_SITES_ENABLED}/waf-integration" ]]; then
    rm "${NGINX_SITES_ENABLED}/waf-integration"
fi
ln -s "$WAF_CONF_FILE" "${NGINX_SITES_ENABLED}/waf-integration"

# Disable default site if it exists
if [[ -L "${NGINX_SITES_ENABLED}/default" ]]; then
    rm "${NGINX_SITES_ENABLED}/default"
    log_info "Disabled default Nginx site"
fi

# Test configuration
log_info "Testing Nginx configuration..."
if nginx -t; then
    log_success "Nginx configuration is valid"
else
    log_error "Nginx configuration has errors"
    exit 1
fi

# Reload Nginx
log_info "Reloading Nginx..."
if command -v openresty &> /dev/null; then
    systemctl reload openresty
else
    systemctl reload nginx
fi

log_success "Nginx WAF integration setup complete!"
echo ""
echo "Configuration details:"
echo "  - WAF Service URL: http://127.0.0.1:8000"
echo "  - Nginx Config: $WAF_CONF_FILE"
echo "  - Access Log: /var/log/nginx/waf_access.log"
echo "  - Error Log: /var/log/nginx/waf_error.log"
echo ""
echo "To start the WAF service:"
echo "  cd $PROJECT_ROOT"
echo "  python scripts/start_waf_service.py"
echo ""
echo "To start your backend application on port 8080:"
echo "  # Example for a Node.js app:"
echo "  cd applications/app1-juice-shop && npm start"
echo ""
echo "To test the setup:"
echo "  curl -v http://localhost/"
echo ""
echo "To monitor WAF metrics:"
echo "  curl http://localhost/waf-metrics"