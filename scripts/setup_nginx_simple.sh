#!/bin/bash

# Simple Nginx Setup for WAF Integration
# Uses basic nginx features without Lua for broader compatibility

set -e

echo "Setting up simple Nginx with WAF integration..."

# Configuration variables
NGINX_CONF_DIR="/etc/nginx"
NGINX_SITES_AVAILABLE="${NGINX_CONF_DIR}/sites-available"
NGINX_SITES_ENABLED="${NGINX_SITES_ENABLED:-${NGINX_CONF_DIR}/sites-enabled}"
NGINX_CONF_FILE="${NGINX_CONF_DIR}/nginx.conf"
WAF_CONF_FILE="${NGINX_SITES_AVAILABLE}/waf-simple"
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
    log_info "Example: sudo ./scripts/setup_nginx_simple.sh"
    exit 1
fi

# Detect OS and install nginx
if command -v pacman &> /dev/null; then
    # Arch Linux
    log_info "Detected Arch Linux"
    if ! command -v nginx &> /dev/null; then
        log_info "Installing nginx..."
        pacman -Sy --noconfirm nginx
    fi
elif command -v apt &> /dev/null; then
    # Ubuntu/Debian
    log_info "Detected Ubuntu/Debian"
    if ! command -v nginx &> /dev/null; then
        log_info "Installing nginx..."
        apt update
        apt install -y nginx
    fi
elif command -v yum &> /dev/null; then
    # CentOS/RHEL
    log_info "Detected CentOS/RHEL"
    if ! command -v nginx &> /dev/null; then
        log_info "Installing nginx..."
        yum install -y nginx
    fi
else
    log_error "Unsupported OS. Please install nginx manually."
    exit 1
fi

# Enable and start nginx
log_info "Enabling and starting nginx..."
systemctl enable nginx
systemctl start nginx

# Create log directory
mkdir -p /var/log/nginx
chmod 755 /var/log/nginx

# Backup existing configuration
if [[ -f "$NGINX_CONF_FILE" ]]; then
    cp "$NGINX_CONF_FILE" "${NGINX_CONF_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    log_info "Backed up existing nginx.conf"
fi

# Create sites directories if they don't exist
mkdir -p "$NGINX_SITES_AVAILABLE" "$NGINX_SITES_ENABLED"

# Copy simple WAF configuration
log_info "Installing simple WAF Nginx configuration..."
cp "${SCRIPT_DIR}/nginx_waf_simple.conf" "$WAF_CONF_FILE"

# Enable the site
if [[ -L "${NGINX_SITES_ENABLED}/waf-simple" ]]; then
    rm "${NGINX_SITES_ENABLED}/waf-simple"
fi
ln -sf "$WAF_CONF_FILE" "${NGINX_SITES_ENABLED}/waf-simple"

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
systemctl reload nginx

log_success "Simple Nginx WAF integration setup complete!"
echo ""
echo "Configuration details:"
echo "  - WAF Service URL: http://127.0.0.1:8000"
echo "  - Nginx Config: $WAF_CONF_FILE"
echo "  - Access Log: /var/log/nginx/waf_access.log"
echo "  - Error Log: /var/log/nginx/waf_error.log"
echo ""
echo "Security Features:"
echo "  - XSS pattern blocking"
echo "  - SQL injection pattern blocking"
echo "  - Directory traversal protection"
echo "  - Automated tool detection"
echo "  - Rate limiting"
echo ""
echo "To test the setup:"
echo "  curl -v http://localhost/"
echo "  curl -v http://localhost/api/test"
echo ""
echo "To monitor WAF metrics:"
echo "  curl http://localhost/waf-metrics"
echo ""
echo "Note: This simple configuration uses nginx built-in features."
echo "For advanced AI-powered anomaly detection, the WAF service"
echo "provides real-time analysis that can be integrated separately."