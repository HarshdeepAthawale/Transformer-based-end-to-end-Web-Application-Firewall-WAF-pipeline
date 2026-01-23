#!/bin/bash
# Setup script for real web applications: Juice Shop, WebGoat, DVWA

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPS_DIR="$PROJECT_DIR/applications"

echo "=========================================="
echo "Setting up Real Web Applications"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

cd "$PROJECT_DIR"

# Check if applications are cloned
if [ ! -d "$APPS_DIR/app1-juice-shop" ]; then
    echo -e "${RED}Error: Juice Shop not found. Please clone it first.${NC}"
    exit 1
fi

if [ ! -d "$APPS_DIR/app2-webgoat" ]; then
    echo -e "${RED}Error: WebGoat not found. Please clone it first.${NC}"
    exit 1
fi

if [ ! -d "$APPS_DIR/app3-dvwa" ]; then
    echo -e "${RED}Error: DVWA not found. Please clone it first.${NC}"
    exit 1
fi

echo -e "${YELLOW}Step 1: Setting up Juice Shop (App 1 - Port 8080)...${NC}"
cd "$APPS_DIR/app1-juice-shop"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js not found. Please install Node.js 20+${NC}"
    exit 1
fi

NODE_VERSION=$(node --version)
echo -e "${GREEN}✓ Node.js $NODE_VERSION found${NC}"

# Install dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing Juice Shop dependencies (this may take a few minutes)..."
    npm install
    echo -e "${GREEN}✓ Juice Shop dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Juice Shop dependencies already installed${NC}"
fi

# Create startup script
cat > "$APPS_DIR/app1-juice-shop/start.sh" << 'EOF'
#!/bin/bash
# Start Juice Shop on port 8080
export PORT=8080
npm start
EOF
chmod +x "$APPS_DIR/app1-juice-shop/start.sh"
echo -e "${GREEN}✓ Juice Shop startup script created${NC}"

echo ""
echo -e "${YELLOW}Step 2: Setting up WebGoat (App 2 - Port 8081)...${NC}"
cd "$APPS_DIR/app2-webgoat"

# Check Java
if ! command -v java &> /dev/null; then
    echo -e "${RED}Error: Java not found. Please install Java 17+${NC}"
    exit 1
fi

JAVA_VERSION=$(java -version 2>&1 | head -1)
echo -e "${GREEN}✓ $JAVA_VERSION${NC}"

# Check Maven
if ! command -v mvn &> /dev/null; then
    echo -e "${YELLOW}Maven not found. Installing Maven wrapper...${NC}"
    # WebGoat should have mvnw wrapper
    if [ -f "./mvnw" ]; then
        chmod +x ./mvnw
        echo -e "${GREEN}✓ Maven wrapper found${NC}"
    else
        echo -e "${RED}Error: Maven wrapper not found. Please install Maven.${NC}"
        exit 1
    fi
fi

# Build WebGoat (skip tests for faster build)
if [ ! -f "webgoat-container/target/webgoat-*.jar" ]; then
    echo "Building WebGoat (this may take 5-10 minutes)..."
    if [ -f "./mvnw" ]; then
        ./mvnw clean package -DskipTests
    else
        mvn clean package -DskipTests
    fi
    echo -e "${GREEN}✓ WebGoat built${NC}"
else
    echo -e "${GREEN}✓ WebGoat already built${NC}"
fi

# Create startup script
cat > "$APPS_DIR/app2-webgoat/start.sh" << 'EOF'
#!/bin/bash
# Start WebGoat on port 8081
cd "$(dirname "$0")"
JAR_FILE=$(find webgoat-container/target -name "webgoat-*.jar" | head -1)
if [ -z "$JAR_FILE" ]; then
    echo "Error: WebGoat JAR not found. Please build first."
    exit 1
fi
java -jar "$JAR_FILE" --server.port=8081 --webgoat.port=8081 --webwolf.port=9091
EOF
chmod +x "$APPS_DIR/app2-webgoat/start.sh"
echo -e "${GREEN}✓ WebGoat startup script created${NC}"

echo ""
echo -e "${YELLOW}Step 3: Setting up DVWA (App 3 - Port 8082)...${NC}"
cd "$APPS_DIR/app3-dvwa"

# Check PHP
if ! command -v php &> /dev/null; then
    echo -e "${YELLOW}PHP not found. Installing PHP...${NC}"
    if command -v pacman &> /dev/null; then
        echo "zenbook" | sudo -S pacman -Sy --noconfirm php php-apache mariadb 2>/dev/null || echo "Please install PHP manually"
    else
        echo -e "${RED}Error: Please install PHP 7.3+ manually${NC}"
        exit 1
    fi
fi

PHP_VERSION=$(php --version | head -1)
echo -e "${GREEN}✓ $PHP_VERSION${NC}"

# Configure DVWA
if [ ! -f "config/config.inc.php" ]; then
    cp config/config.inc.php.dist config/config.inc.php
    echo -e "${GREEN}✓ DVWA config file created${NC}"
else
    echo -e "${GREEN}✓ DVWA config file exists${NC}"
fi

# Create startup script for DVWA (using PHP built-in server)
cat > "$APPS_DIR/app3-dvwa/start.sh" << 'EOF'
#!/bin/bash
# Start DVWA on port 8082 using PHP built-in server
cd "$(dirname "$0")"
php -S localhost:8082
EOF
chmod +x "$APPS_DIR/app3-dvwa/start.sh"
echo -e "${GREEN}✓ DVWA startup script created${NC}"

echo ""
echo -e "${GREEN}=========================================="
echo "All applications setup complete!"
echo "==========================================${NC}"
echo ""
echo "Applications ready:"
echo "  - App 1 (Juice Shop): $APPS_DIR/app1-juice-shop/start.sh"
echo "  - App 2 (WebGoat): $APPS_DIR/app2-webgoat/start.sh"
echo "  - App 3 (DVWA): $APPS_DIR/app3-dvwa/start.sh"
echo ""
echo "To start all applications, run:"
echo "  bash scripts/start_real_apps.sh"
