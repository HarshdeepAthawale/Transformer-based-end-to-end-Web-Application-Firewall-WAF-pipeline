#!/bin/bash
# Phase 1 Setup Script - Environment Setup & Web Application Deployment
# This script automates the setup of Phase 1 components

set -e  # Exit on error

echo "=========================================="
echo "Phase 1: Environment Setup"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is in use
port_in_use() {
    lsof -i :"$1" >/dev/null 2>&1
}

echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"

# Check Python
if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Check Java
if ! command_exists java; then
    echo -e "${YELLOW}Warning: Java not found. Installing OpenJDK...${NC}"
    if command_exists apt-get; then
        sudo apt-get update
        sudo apt-get install -y openjdk-11-jdk
    elif command_exists yum; then
        sudo yum install -y java-11-openjdk-devel
    else
        echo -e "${RED}Error: Cannot install Java automatically. Please install Java 8+ manually.${NC}"
        exit 1
    fi
fi
JAVA_VERSION=$(java -version 2>&1 | head -n 1)
echo -e "${GREEN}✓ $JAVA_VERSION${NC}"

# Check if running as root for system operations
if [ "$EUID" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

echo -e "${YELLOW}Step 2: Setting up Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

source venv/bin/activate
pip install --upgrade pip >/dev/null 2>&1
pip install -r requirements.txt
echo -e "${GREEN}✓ Python dependencies installed${NC}"

echo -e "${YELLOW}Step 3: Installing and configuring Nginx...${NC}"
if ! command_exists nginx; then
    if command_exists apt-get; then
        $SUDO apt-get update
        $SUDO apt-get install -y nginx
    elif command_exists yum; then
        $SUDO yum install -y nginx
    else
        echo -e "${RED}Error: Cannot install Nginx automatically${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Nginx installed${NC}"
else
    echo -e "${GREEN}✓ Nginx already installed${NC}"
fi

# Configure Nginx logging
NGINX_CONF="/etc/nginx/nginx.conf"
if [ -f "$NGINX_CONF" ]; then
    # Backup original config
    if [ ! -f "${NGINX_CONF}.backup" ]; then
        $SUDO cp "$NGINX_CONF" "${NGINX_CONF}.backup"
    fi
    
    # Check if detailed log format already exists
    if ! grep -q "log_format detailed" "$NGINX_CONF"; then
        # Add detailed log format
        $SUDO sed -i '/http {/a\    log_format detailed '\''$remote_addr - $remote_user [$time_local] '\''\n                       '\''"$request" $status $body_bytes_sent '\''\n                       '\''"$http_referer" "$http_user_agent" '\''\n                       '\''"$http_x_forwarded_for" "$http_cookie" '\''\n                       '\''"$content_type" "$content_length" '\''\n                       '\''"$request_body"'\'';' "$NGINX_CONF"
        
        # Update access_log to use detailed format
        $SUDO sed -i 's|access_log.*;|access_log /var/log/nginx/access.log detailed;|g' "$NGINX_CONF"
    fi
    echo -e "${GREEN}✓ Nginx logging configured${NC}"
fi

# Start/restart Nginx
if systemctl is-active --quiet nginx; then
    $SUDO systemctl reload nginx
else
    $SUDO systemctl start nginx
    $SUDO systemctl enable nginx
fi
echo -e "${GREEN}✓ Nginx started${NC}"

echo -e "${YELLOW}Step 4: Installing Apache Tomcat...${NC}"
TOMCAT_DIR="/opt/tomcat9"
TOMCAT_VERSION="9.0.65"

if [ ! -d "$TOMCAT_DIR" ]; then
    echo "Downloading Apache Tomcat $TOMCAT_VERSION..."
    cd /tmp
    wget -q "https://archive.apache.org/dist/tomcat/tomcat-9/v${TOMCAT_VERSION}/bin/apache-tomcat-${TOMCAT_VERSION}.tar.gz"
    $SUDO tar -xzf "apache-tomcat-${TOMCAT_VERSION}.tar.gz"
    $SUDO mv "apache-tomcat-${TOMCAT_VERSION}" "$TOMCAT_DIR"
    $SUDO chown -R "$USER:$USER" "$TOMCAT_DIR"
    chmod +x "$TOMCAT_DIR/bin"/*.sh
    rm -f "apache-tomcat-${TOMCAT_VERSION}.tar.gz"
    echo -e "${GREEN}✓ Tomcat installed${NC}"
else
    echo -e "${GREEN}✓ Tomcat already installed${NC}"
fi

cd "$PROJECT_DIR"

echo -e "${YELLOW}Step 5: Creating sample WAR applications...${NC}"
# This will be handled by a separate script
if [ -f "scripts/create_war_apps.sh" ]; then
    bash scripts/create_war_apps.sh
else
    echo -e "${YELLOW}Warning: WAR creation script not found. Creating basic structure...${NC}"
    mkdir -p "$TOMCAT_DIR/webapps/app1" "$TOMCAT_DIR/webapps/app2" "$TOMCAT_DIR/webapps/app3"
fi

echo -e "${YELLOW}Step 6: Configuring Tomcat for multiple applications...${NC}"
# Configure Tomcat server.xml for multiple ports
TOMCAT_SERVER_XML="$TOMCAT_DIR/conf/server.xml"
if [ -f "$TOMCAT_SERVER_XML" ]; then
    # Backup original
    if [ ! -f "${TOMCAT_SERVER_XML}.backup" ]; then
        cp "$TOMCAT_SERVER_XML" "${TOMCAT_SERVER_XML}.backup"
    fi
    
    # Note: Full server.xml configuration will be handled separately
    # as it requires careful XML editing
    echo -e "${GREEN}✓ Tomcat configuration prepared${NC}"
fi

echo -e "${YELLOW}Step 7: Starting Tomcat...${NC}"
# Check if ports are in use
for port in 8080 8081 8082; do
    if port_in_use "$port"; then
        echo -e "${YELLOW}Warning: Port $port is already in use${NC}"
    fi
done

# Start Tomcat
if [ ! -f "$TOMCAT_DIR/bin/catalina.pid" ] || ! pgrep -f "catalina" > /dev/null; then
    "$TOMCAT_DIR/bin/startup.sh" || echo -e "${YELLOW}Warning: Tomcat may already be running${NC}"
    sleep 5
    echo -e "${GREEN}✓ Tomcat started${NC}"
else
    echo -e "${GREEN}✓ Tomcat already running${NC}"
fi

echo ""
echo -e "${GREEN}=========================================="
echo "Phase 1 Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Run: python scripts/test_setup.py"
echo "2. Check logs: tail -f /var/log/nginx/access.log"
echo "3. Test applications: curl http://localhost:8080"
echo ""
