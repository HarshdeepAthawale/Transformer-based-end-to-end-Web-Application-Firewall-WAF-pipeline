#!/bin/bash
# Setup Nginx with detailed logging for WAF pipeline

set -e

echo "Setting up Nginx configuration..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    SUDO="sudo"
else
    SUDO=""
fi

NGINX_CONF="/etc/nginx/nginx.conf"
WAF_CONF="/etc/nginx/conf.d/waf.conf"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if Nginx is installed
if ! command -v nginx &> /dev/null; then
    echo "Nginx is not installed. Installing..."
    if command -v apt-get &> /dev/null; then
        $SUDO apt-get update
        $SUDO apt-get install -y nginx
    elif command -v yum &> /dev/null; then
        $SUDO yum install -y nginx
    else
        echo "Error: Cannot install Nginx automatically"
        exit 1
    fi
fi

# Backup original nginx.conf
if [ ! -f "${NGINX_CONF}.backup" ]; then
    $SUDO cp "$NGINX_CONF" "${NGINX_CONF}.backup"
    echo "✓ Backed up original nginx.conf"
fi

# Add detailed log format to nginx.conf if not present
if ! grep -q "log_format detailed" "$NGINX_CONF"; then
    # Find the http block and add log format
    $SUDO sed -i '/^http {/a\    log_format detailed '\''$remote_addr - $remote_user [$time_local] '\''\n                       '\''"$request" $status $body_bytes_sent '\''\n                       '\''"$http_referer" "$http_user_agent" '\''\n                       '\''"$http_x_forwarded_for" "$http_cookie" '\''\n                       '\''"$content_type" "$content_length" '\''\n                       '\''"$request_body"'\'';' "$NGINX_CONF"
    echo "✓ Added detailed log format"
fi

# Copy WAF configuration to conf.d
if [ -d "/etc/nginx/conf.d" ]; then
    $SUDO cp "$SCRIPT_DIR/nginx_waf.conf" "$WAF_CONF"
    echo "✓ Created WAF configuration at $WAF_CONF"
else
    echo "Warning: /etc/nginx/conf.d not found. You may need to include the configuration manually."
fi

# Test Nginx configuration
if $SUDO nginx -t; then
    echo "✓ Nginx configuration is valid"
    
    # Reload Nginx
    if systemctl is-active --quiet nginx; then
        $SUDO systemctl reload nginx
        echo "✓ Nginx reloaded"
    else
        $SUDO systemctl start nginx
        $SUDO systemctl enable nginx
        echo "✓ Nginx started"
    fi
else
    echo "Error: Nginx configuration test failed"
    exit 1
fi

# Set up log rotation
LOGROTATE_CONF="/etc/logrotate.d/nginx-waf"
$SUDO tee "$LOGROTATE_CONF" > /dev/null << 'EOF'
/var/log/nginx/access.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 www-data adm
    sharedscripts
    postrotate
        systemctl reload nginx > /dev/null 2>&1 || true
    endscript
}

/var/log/nginx/error.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 www-data adm
    sharedscripts
    postrotate
        systemctl reload nginx > /dev/null 2>&1 || true
    endscript
}
EOF

echo "✓ Log rotation configured"

# Ensure log directory exists and has correct permissions
$SUDO mkdir -p /var/log/nginx
$SUDO touch /var/log/nginx/access.log /var/log/nginx/error.log
$SUDO chown www-data:adm /var/log/nginx/*.log 2>/dev/null || $SUDO chown nginx:nginx /var/log/nginx/*.log 2>/dev/null || true
$SUDO chmod 644 /var/log/nginx/*.log

echo ""
echo "Nginx setup complete!"
echo "Access logs: /var/log/nginx/access.log"
echo "Error logs: /var/log/nginx/error.log"
