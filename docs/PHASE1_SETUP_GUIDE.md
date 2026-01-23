# Phase 1 Setup Guide - Complete Implementation

This guide provides step-by-step instructions for setting up Phase 1 of the Transformer-based WAF pipeline.

## Prerequisites Check

Before starting, verify you have:

```bash
# Check Python version (3.9+ required)
python3 --version

# Check Java version (8+ required for WAR apps)
java -version
javac -version

# Check available disk space (20GB recommended)
df -h

# Check available RAM (8GB minimum)
free -h
```

## Option 1: Automated Setup (Recommended)

The easiest way to set up Phase 1 is using the automated script:

```bash
# Make scripts executable (if not already)
chmod +x scripts/*.sh scripts/*.py

# Run the main setup script
bash scripts/setup_phase1.sh
```

This script will:
1. Check prerequisites
2. Set up Python virtual environment
3. Install Python dependencies
4. Install and configure Nginx
5. Install Apache Tomcat
6. Create and deploy WAR applications
7. Configure logging
8. Start services

## Option 2: Manual Setup

### Step 1: Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Nginx Setup

```bash
# Install Nginx
sudo apt-get update
sudo apt-get install nginx

# Run Nginx setup script
sudo bash scripts/setup_nginx.sh

# Or manually configure (see nginx_waf.conf)
```

### Step 3: Web Applications

#### Option A: Java/Tomcat Applications

```bash
# Install Tomcat (if not done by setup script)
cd /opt
sudo wget https://archive.apache.org/dist/tomcat/tomcat-9/v9.0.65/bin/apache-tomcat-9.0.65.tar.gz
sudo tar -xzf apache-tomcat-9.0.65.tar.gz
sudo mv apache-tomcat-9.0.65 tomcat9
sudo chown -R $USER:$USER /opt/tomcat9
chmod +x /opt/tomcat9/bin/*.sh

# Create and deploy WAR applications
bash scripts/create_war_apps.sh

# Configure Tomcat
bash scripts/configure_tomcat.sh

# Start Tomcat
/opt/tomcat9/bin/startup.sh
```

#### Option B: Python Applications (Alternative)

If Java/Tomcat setup is problematic:

```bash
# Start Python-based web applications
python scripts/simple_web_apps.py

# Or run in background
nohup python scripts/simple_web_apps.py > logs/web_apps.log 2>&1 &
```

### Step 4: Verify Setup

```bash
# Run test script
python scripts/test_setup.py

# Test applications manually
curl http://localhost:8080
curl http://localhost:8081
curl http://localhost:8082

# Check logs
tail -f /var/log/nginx/access.log
```

## Configuration Files

### config/config.yaml

Main configuration file with:
- Web server settings
- Application ports and log paths
- Data directory paths
- Model configuration
- WAF service settings

### .env

Environment variables (create manually if needed):
```bash
ENV=development
LOG_LEVEL=INFO
LOG_FILE=logs/waf.log
MODEL_NAME=transformer_waf
MODEL_VERSION=1.0.0
ANOMALY_THRESHOLD=0.7
```

## Starting and Stopping Applications

```bash
# Start all applications
bash scripts/start_apps.sh

# Stop all applications
bash scripts/stop_apps.sh
```

## Generating Test Traffic

To generate test requests for log analysis:

```bash
# Simple test requests
for i in {1..10}; do
    curl -s "http://localhost:8080/test?param=value$i"
    curl -s -X POST "http://localhost:8080/api/data" -d "data=test$i"
    curl -s "http://localhost:8081/api/endpoint?id=$i"
    curl -s -X POST "http://localhost:8082/data" -H "Content-Type: application/json" -d '{"key":"value"}'
done

# Check logs
tail -20 /var/log/nginx/access.log
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port
sudo lsof -i :8080

# Kill process
sudo kill -9 <PID>

# Or change port in config/config.yaml
```

### Permission Denied on Log Files

```bash
# Add user to appropriate group
sudo usermod -a -G adm $USER
sudo usermod -a -G www-data $USER

# Fix permissions
sudo chmod 644 /var/log/nginx/access.log
sudo chown www-data:adm /var/log/nginx/access.log
```

### Tomcat Not Starting

```bash
# Check JAVA_HOME
echo $JAVA_HOME

# Set JAVA_HOME if needed
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Check Tomcat logs
tail -f /opt/tomcat9/logs/catalina.out

# Verify ports are available
netstat -tuln | grep -E '808[0-2]'
```

### Nginx Configuration Errors

```bash
# Test configuration
sudo nginx -t

# Check error log
sudo tail -f /var/log/nginx/error.log

# Restore backup if needed
sudo cp /etc/nginx/nginx.conf.backup /etc/nginx/nginx.conf
sudo nginx -t
sudo systemctl reload nginx
```

### Python Applications Not Starting

```bash
# Check if ports are available
netstat -tuln | grep -E '808[0-2]'

# Check Python version
python3 --version

# Run with verbose output
python3 scripts/simple_web_apps.py

# Check for errors in logs
cat logs/web_apps.log
```

## Verification Checklist

After setup, verify:

- [ ] Python virtual environment created and activated
- [ ] All Python dependencies installed
- [ ] Project directory structure exists
- [ ] Configuration files created (config.yaml)
- [ ] Nginx installed and configured
- [ ] Nginx detailed logging enabled
- [ ] Three web applications accessible (ports 8080, 8081, 8082)
- [ ] Log files being generated
- [ ] Test script passes all checks
- [ ] Can generate and view access logs

## Next Steps

Once Phase 1 is complete:

1. **Verify log generation**: Ensure logs are being written with detailed format
2. **Generate test traffic**: Create diverse HTTP requests to populate logs
3. **Review log format**: Verify logs contain all necessary fields
4. **Proceed to Phase 2**: Log Ingestion System

## Support

For issues:
1. Check the troubleshooting section above
2. Review log files for errors
3. Run test script: `python scripts/test_setup.py`
4. Check service status: `systemctl status nginx`

## Files Created

- `scripts/setup_phase1.sh` - Main setup script
- `scripts/setup_nginx.sh` - Nginx configuration
- `scripts/create_war_apps.sh` - WAR application creation
- `scripts/configure_tomcat.sh` - Tomcat configuration
- `scripts/simple_web_apps.py` - Python web applications
- `scripts/start_apps.sh` - Start applications
- `scripts/stop_apps.sh` - Stop applications
- `scripts/test_setup.py` - Verification script
- `config/config.yaml` - Main configuration
- `requirements.txt` - Python dependencies
