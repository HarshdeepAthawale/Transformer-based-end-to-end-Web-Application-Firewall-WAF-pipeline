#!/bin/bash

# Complete Advanced WAF Setup for Arch Linux
# This script sets up the full AI-powered WAF with OpenResty and Lua integration

set -e

echo "🚀 Setting up Complete Advanced WAF with AI Integration"
echo "======================================================"

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo ""
echo "Step 1: Installing OpenResty with Lua support..."
echo "------------------------------------------------"
./setup_openresty_arch.sh

echo ""
echo "Step 2: Configuring Advanced WAF Integration..."
echo "------------------------------------------------"
./setup_nginx_waf_advanced.sh

echo ""
echo "Step 3: Verification and Testing..."
echo "-----------------------------------"

# Test OpenResty configuration
log_info "Testing OpenResty configuration..."
if openresty -t -c /usr/local/openresty/nginx/conf/nginx.conf; then
    log_success "OpenResty configuration is valid"
else
    log_error "OpenResty configuration has errors"
    exit 1
fi

# Reload OpenResty to apply changes
log_info "Reloading OpenResty..."
systemctl reload openresty

# Wait for reload
sleep 3

# Verify OpenResty is running
if systemctl is-active --quiet openresty; then
    log_success "OpenResty is running successfully"
else
    log_error "OpenResty failed to start"
    journalctl -u openresty --no-pager -n 20
    exit 1
fi

echo ""
log_success "🎉 Advanced WAF Setup Complete!"
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                     SETUP SUMMARY                           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║ ✅ OpenResty installed with Lua support                    ║"
echo "║ ✅ Advanced WAF configuration deployed                    ║"
echo "║ ✅ Real-time AI anomaly detection ready                   ║"
echo "║ ✅ Lua scripting for request interception                 ║"
echo "║ ✅ Rate limiting and security headers                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                     NEXT STEPS                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║ 1. Start WAF Service:                                      ║"
echo "║    cd $PROJECT_ROOT                                       ║"
echo "║    source venv/bin/activate                               ║"
echo "║    python scripts/start_waf_service.py --port 8000        ║"
echo "║                                                           ║"
echo "║ 2. Start Backend App (port 8080):                        ║"
echo "║    cd applications/app1-juice-shop                        ║"
echo "║    npm start                                              ║"
echo "║                                                           ║"
echo "║ 3. Test Protection:                                       ║"
echo "║    curl -v http://localhost/                              ║"
echo "║    curl -v 'http://localhost/api/test?id=1%27%20OR%20%271%27=%271' ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                     MONITORING                              ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║ • WAF Metrics: curl http://localhost/waf-metrics          ║"
echo "║ • OpenResty Logs: journalctl -u openresty -f              ║"
echo "║ • WAF Service Logs: Check terminal running WAF service    ║"
echo "║ • Health Check: curl http://127.0.0.1:8000/health         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "🔒 AI-Powered Protection Features:"
echo "   • Real-time Transformer-based anomaly detection"
echo "   • Request normalization and parsing"
echo "   • Async processing with thread pools"
echo "   • Configurable detection thresholds"
echo "   • Comprehensive security logging"
echo "   • Fail-open policy for reliability"
echo ""
echo "🚀 Ready for Phase 7: Real-Time Non-Blocking Detection!"