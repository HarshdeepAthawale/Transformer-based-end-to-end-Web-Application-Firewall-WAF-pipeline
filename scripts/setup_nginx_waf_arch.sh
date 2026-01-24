#!/bin/bash

# Setup Nginx with WAF Integration for Arch Linux
# This script configures Nginx with Lua support as a reverse proxy with WAF service integration

set -e

echo "Setting up Nginx with WAF integration on Arch Linux..."

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

# Check if OpenResty is available, if not install it
if ! command -v openresty &> /dev/null; then
    log_info "Installing OpenResty for Lua support..."

    # Update package database
    pacman -Sy --noconfirm

    # Install OpenResty from AUR or official repos
    if pacman -Si openresty &> /dev/null; then
        # If available in official repos
        pacman -S --noconfirm openresty
    else
        log_warning "OpenResty not in official repos. Installing nginx with lua-nginx-module..."

        # Install nginx with lua support
        pacman -S --noconfirm nginx lua-nginx-module

        # Check if lua-nginx-module was installed
        if ! pacman -Q lua-nginx-module &> /dev/null; then
            log_error "lua-nginx-module not available. Please install OpenResty manually:"
            echo "  yay -S openresty"
            echo "  or visit: https://openresty.org/en/installation.html"
            exit 1
        fi
    fi

    # Enable and start OpenResty/nginx
    if command -v openresty &> /dev/null; then
        systemctl enable openresty
        systemctl start openresty
        NGINX_CMD="openresty"
    else
        systemctl enable nginx
        systemctl start nginx
        NGINX_CMD="nginx"
    fi

    log_success "OpenResty/nginx installed and started"
else
    log_info "OpenResty already installed"
    NGINX_CMD="openresty"
    if ! command -v openresty &> /dev/null && command -v nginx &> /dev/null; then
        NGINX_CMD="nginx"
    fi
fi

# Install lua-resty-http if not available
if ! pacman -Q lua-resty-http &> /dev/null 2>&1; then
    log_info "Installing lua-resty-http..."
    if pacman -Si lua-resty-http &> /dev/null; then
        pacman -S --noconfirm lua-resty-http
    else
        log_warning "lua-resty-http not in repos. Will use built-in HTTP client if available."
    fi
fi

# Create necessary directories
mkdir -p /var/log/nginx
chmod 755 /var/log/nginx

# Backup existing nginx.conf if it exists
if [[ -f "$NGINX_CONF_FILE" ]]; then
    cp "$NGINX_CONF_FILE" "${NGINX_CONF_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    log_info "Backed up existing nginx.conf"
fi

# Copy WAF configuration
log_info "Installing WAF Nginx configuration..."
cp "${SCRIPT_DIR}/nginx_waf.conf" "$WAF_CONF_FILE"

# For Arch Linux, we need to adjust the configuration for the actual nginx setup
# Check if we have OpenResty or regular nginx with lua
if command -v openresty &> /dev/null; then
    # OpenResty uses different paths
    NGINX_CONF_DIR="/usr/local/openresty/nginx/conf"
    NGINX_SITES_AVAILABLE="${NGINX_CONF_DIR}/sites-available"
    NGINX_SITES_ENABLED="${NGINX_CONF_DIR}/sites-enabled"

    # Recreate paths
    mkdir -p "$NGINX_SITES_AVAILABLE" "$NGINX_SITES_ENABLED"

    # Copy config to correct location
    cp "${SCRIPT_DIR}/nginx_waf.conf" "$WAF_CONF_FILE"
else
    # Regular nginx
    # Enable the site
    if [[ -L "${NGINX_SITES_ENABLED}/waf-integration" ]]; then
        rm "${NGINX_SITES_ENABLED}/waf-integration"
    fi
    ln -sf "$WAF_CONF_FILE" "${NGINX_SITES_ENABLED}/waf-integration"

    # Disable default site if it exists
    if [[ -L "${NGINX_SITES_ENABLED}/default" ]]; then
        rm "${NGINX_SITES_ENABLED}/default"
        log_info "Disabled default Nginx site"
    fi
fi

# Test configuration
log_info "Testing Nginx configuration..."
if $NGINX_CMD -t; then
    log_success "Nginx configuration is valid"
else
    log_error "Nginx configuration has errors. Please check the configuration manually."
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
if command -v openresty &> /dev/null; then
    echo "  - Nginx Config: $WAF_CONF_FILE"
    echo "  - Using OpenResty with Lua support"
else
    echo "  - Nginx Config: $WAF_CONF_FILE"
    echo "  - Using nginx with lua-nginx-module"
fi
echo "  - Access Log: /var/log/nginx/waf_access.log"
echo "  - Error Log: /var/log/nginx/waf_error.log"
echo ""
echo "To test the setup:"
echo "  curl -v http://localhost/"
echo ""
echo "To monitor WAF metrics:"
echo "  curl http://localhost/waf-metrics"
echo ""
echo "If you encounter issues, check the logs:"
echo "  sudo journalctl -u nginx -f"
echo "  or"
echo "  sudo journalctl -u openresty -f"