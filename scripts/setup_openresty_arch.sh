#!/bin/bash

# Install OpenResty on Arch Linux
# OpenResty provides Lua support for advanced WAF integration

set -e

echo "Installing OpenResty on Arch Linux for advanced WAF integration..."

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

# Update package database
log_info "Updating package database..."
pacman -Sy --noconfirm

# Install dependencies
log_info "Installing dependencies..."
pacman -S --noconfirm base-devel git wget curl pcre pcre2 openssl zlib

# Check if yay (AUR helper) is available
if command -v yay &> /dev/null; then
    log_info "Using yay to install OpenResty from AUR..."
    yay -S --noconfirm openresty

elif command -v paru &> /dev/null; then
    log_info "Using paru to install OpenResty from AUR..."
    paru -S --noconfirm openresty

else
    log_warning "No AUR helper found. Installing manually..."

    # Create temporary directory
    TMP_DIR="/tmp/openresty-install"
    mkdir -p "$TMP_DIR"
    cd "$TMP_DIR"

    # Download and extract OpenResty
    log_info "Downloading OpenResty..."
    wget https://openresty.org/download/openresty-1.21.4.1.tar.gz
    tar -xzf openresty-1.21.4.1.tar.gz
    cd openresty-1.21.4.1

    # Configure and build
    log_info "Building OpenResty..."
    ./configure \
        --prefix=/usr/local/openresty \
        --with-pcre \
        --with-pcre2 \
        --with-openssl=/usr \
        --with-zlib=/usr \
        --with-http_ssl_module \
        --with-http_v2_module \
        --with-http_realip_module \
        --with-http_stub_status_module \
        --with-luajit

    make -j$(nproc)
    make install

    # Create symlink for openresty command
    ln -sf /usr/local/openresty/bin/openresty /usr/local/bin/openresty
    ln -sf /usr/local/openresty/bin/resty /usr/local/bin/resty

    cd /
    rm -rf "$TMP_DIR"
fi

# Verify installation
if command -v openresty &> /dev/null; then
    log_success "OpenResty installed successfully"
    openresty -v
else
    log_error "OpenResty installation failed"
    exit 1
fi

# Install lua-resty-http
log_info "Installing lua-resty-http..."
if pacman -Si lua-resty-http &> /dev/null; then
    pacman -S --noconfirm lua-resty-http
else
    log_warning "lua-resty-http not in repos. Installing manually..."
    mkdir -p /usr/local/openresty/lualib/resty
    cd /usr/local/openresty/lualib/resty

    # Download lua-resty-http
    wget https://raw.githubusercontent.com/ledgetech/lua-resty-http/master/lib/resty/http.lua
    wget https://raw.githubusercontent.com/ledgetech/lua-resty-http/master/lib/resty/http_connect.lua
    wget https://raw.githubusercontent.com/ledgetech/lua-resty-http/master/lib/resty/http_headers.lua
fi

# Install lua-cjson if not available
if ! pacman -Q lua-cjson &> /dev/null 2>&1; then
    log_info "Installing lua-cjson..."
    if pacman -Si lua-cjson &> /dev/null; then
        pacman -S --noconfirm lua-cjson
    else
        log_warning "lua-cjson not in repos. Using built-in cjson if available."
    fi
fi

# Create OpenResty configuration directories
log_info "Setting up OpenResty configuration directories..."
OPENRESTY_CONF_DIR="/usr/local/openresty/nginx/conf"
OPENRESTY_SITES_AVAILABLE="${OPENRESTY_CONF_DIR}/sites-available"
OPENRESTY_SITES_ENABLED="${OPENRESTY_CONF_DIR}/sites-enabled"

mkdir -p "$OPENRESTY_SITES_AVAILABLE" "$OPENRESTY_SITES_ENABLED"
mkdir -p /usr/local/openresty/nginx/logs
mkdir -p /var/log/openresty

# Create systemd service for OpenResty
log_info "Creating systemd service..."
cat > /etc/systemd/system/openresty.service << 'EOF'
[Unit]
Description=OpenResty
After=network.target

[Service]
Type=forking
PIDFile=/usr/local/openresty/nginx/logs/nginx.pid
ExecStartPre=/usr/local/openresty/bin/openresty -t -c /usr/local/openresty/nginx/conf/nginx.conf
ExecStart=/usr/local/openresty/bin/openresty -c /usr/local/openresty/nginx/conf/nginx.conf
ExecReload=/usr/local/openresty/bin/openresty -t -c /usr/local/openresty/nginx/conf/nginx.conf && /usr/local/openresty/bin/openresty -s reload
ExecStop=/usr/local/openresty/bin/openresty -s stop
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

# Enable and start OpenResty
log_info "Enabling and starting OpenResty..."
systemctl daemon-reload
systemctl enable openresty
systemctl start openresty

# Verify OpenResty is running
sleep 2
if systemctl is-active --quiet openresty; then
    log_success "OpenResty service started successfully"
else
    log_error "OpenResty service failed to start"
    journalctl -u openresty --no-pager -n 20
    exit 1
fi

log_success "OpenResty installation and setup complete!"
echo ""
echo "OpenResty Details:"
echo "  - Binary: $(which openresty)"
echo "  - Config: $OPENRESTY_CONF_DIR/nginx.conf"
echo "  - Sites Available: $OPENRESTY_SITES_AVAILABLE"
echo "  - Sites Enabled: $OPENRESTY_SITES_ENABLED"
echo "  - Logs: /usr/local/openresty/nginx/logs"
echo ""
echo "Next steps:"
echo "  1. Run: sudo ./scripts/setup_nginx_waf_advanced.sh"
echo "  2. Start WAF service: python scripts/start_waf_service.py"
echo "  3. Test: curl http://localhost/"