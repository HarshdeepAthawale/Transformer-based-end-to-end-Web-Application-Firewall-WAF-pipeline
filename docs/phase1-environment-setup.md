# Phase 1: Environment Setup & Web Application Deployment

## Overview
This phase establishes the foundation for the entire WAF pipeline. We'll set up the development environment, deploy sample web applications, and configure web servers to generate detailed access logs that will feed our Transformer-based anomaly detection system.

## Objectives
- Deploy 3 sample WAR applications to a web server
- Configure Apache/Nginx with comprehensive access logging
- Set up Python development environment with all dependencies
- Create organized project structure
- Verify log generation and accessibility

## Prerequisites
- Linux/Unix system (Ubuntu 20.04+ recommended)
- Java JDK 8+ (for WAR applications)
- Python 3.9 or higher
- Root/sudo access for web server configuration
- At least 8GB RAM, 20GB disk space

## Detailed Implementation Steps

### 1.1 Web Server Installation & Configuration

#### Option A: Apache HTTP Server

```bash
# Install Apache
sudo apt-get update
sudo apt-get install apache2

# Install mod_jk for Tomcat integration (if using Tomcat)
sudo apt-get install libapache2-mod-jk

# Configure Apache to enable detailed logging
sudo nano /etc/apache2/apache2.conf
```

**Apache Log Configuration:**
```apache
# Enable detailed logging format
LogFormat "%h %l %u %t \"%r\" %>s %O \"%{Referer}i\" \"%{User-Agent}i\" \"%{X-Forwarded-For}i\" \"%{Cookie}i\" \"%{Content-Type}i\" \"%{Content-Length}i\" \"%{POST_DATA}e\"" detailed

# Set log location
CustomLog ${APACHE_LOG_DIR}/access.log detailed
ErrorLog ${APACHE_LOG_DIR}/error.log

# Enable mod_rewrite for request logging
LoadModule rewrite_module modules/mod_rewrite.so
```

#### Option B: Nginx

```bash
# Install Nginx
sudo apt-get install nginx

# Configure Nginx logging
sudo nano /etc/nginx/nginx.conf
```

**Nginx Log Configuration:**
```nginx
http {
    log_format detailed '$remote_addr - $remote_user [$time_local] '
                       '"$request" $status $body_bytes_sent '
                       '"$http_referer" "$http_user_agent" '
                       '"$http_x_forwarded_for" "$http_cookie" '
                       '"$content_type" "$content_length" '
                       '"$request_body"';

    access_log /var/log/nginx/access.log detailed;
    error_log /var/log/nginx/error.log;
}
```

### 1.2 Deploy Sample WAR Applications

#### Install Apache Tomcat

```bash
# Download Tomcat 9
cd /opt
sudo wget https://archive.apache.org/dist/tomcat/tomcat-9/v9.0.65/bin/apache-tomcat-9.0.65.tar.gz
sudo tar -xzf apache-tomcat-9.0.65.tar.gz
sudo mv apache-tomcat-9.0.65 tomcat9

# Set permissions
sudo chown -R $USER:$USER /opt/tomcat9
chmod +x /opt/tomcat9/bin/*.sh
```

#### Deploy Applications

```bash
# Create application directories
mkdir -p /opt/tomcat9/webapps/app1
mkdir -p /opt/tomcat9/webapps/app2
mkdir -p /opt/tomcat9/webapps/app3

# Extract WAR files (assuming you have 3 WAR files)
# Place them in webapps directory
# Example: app1.war, app2.war, app3.war

# Configure server.xml for multiple ports
sudo nano /opt/tomcat9/conf/server.xml
```

**Tomcat server.xml Configuration:**
```xml
<!-- App 1 on port 8080 -->
<Service name="Catalina">
    <Connector port="8080" protocol="HTTP/1.1" />
    <Engine name="Catalina" defaultHost="localhost">
        <Host name="localhost" appBase="webapps/app1" />
    </Engine>
</Service>

<!-- App 2 on port 8081 -->
<Service name="Catalina2">
    <Connector port="8081" protocol="HTTP/1.1" />
    <Engine name="Catalina2" defaultHost="localhost">
        <Host name="localhost" appBase="webapps/app2" />
    </Engine>
</Service>

<!-- App 3 on port 8082 -->
<Service name="Catalina3">
    <Connector port="8082" protocol="HTTP/1.1" />
    <Engine name="Catalina3" defaultHost="localhost">
        <Host name="localhost" appBase="webapps/app3" />
    </Engine>
</Service>
```

#### Start Tomcat

```bash
/opt/tomcat9/bin/startup.sh

# Verify applications are running
curl http://localhost:8080
curl http://localhost:8081
curl http://localhost:8082
```

### 1.3 Python Environment Setup

```bash
# Create project directory
mkdir -p ~/waf-pipeline
cd ~/waf-pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

#### Install Core Dependencies

```bash
# Create requirements.txt
cat > requirements.txt << EOF
# Deep Learning Framework
torch>=2.0.0
transformers>=4.30.0
tokenizers>=0.13.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Web Framework (for WAF service)
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0

# HTTP Client
requests>=2.31.0
httpx>=0.24.0

# Log Processing
python-json-logger>=2.0.7
loguru>=0.7.0

# Utilities
python-dotenv>=1.0.0
pyyaml>=6.0
tqdm>=4.65.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
EOF

# Install dependencies
pip install -r requirements.txt
```

### 1.4 Project Structure Creation

```bash
# Create directory structure
mkdir -p src/{ingestion,parsing,tokenization,model,training,inference,integration,learning}
mkdir -p data/{raw,processed,normalized,training,validation,test}
mkdir -p models/{checkpoints,vocabularies,deployed}
mkdir -p config
mkdir -p tests/{unit,integration,performance}
mkdir -p logs
mkdir -p docs
mkdir -p scripts

# Create initial files
touch src/__init__.py
touch src/ingestion/__init__.py
touch src/parsing/__init__.py
touch src/tokenization/__init__.py
touch src/model/__init__.py
touch src/training/__init__.py
touch src/inference/__init__.py
touch src/integration/__init__.py
touch src/learning/__init__.py
```

**Project Structure:**
```
waf-pipeline/
├── src/
│   ├── ingestion/          # Log ingestion modules
│   ├── parsing/            # Request parsing modules
│   ├── tokenization/       # Tokenization modules
│   ├── model/              # Model architecture
│   ├── training/           # Training scripts
│   ├── inference/          # Inference engine
│   ├── integration/       # Web server integration
│   └── learning/          # Continuous learning
├── data/
│   ├── raw/               # Raw log files
│   ├── processed/         # Processed logs
│   ├── normalized/        # Normalized requests
│   ├── training/          # Training datasets
│   ├── validation/       # Validation datasets
│   └── test/              # Test datasets
├── models/
│   ├── checkpoints/       # Model checkpoints
│   ├── vocabularies/      # Tokenizer vocabularies
│   └── deployed/          # Production models
├── config/                # Configuration files
├── tests/                 # Test suites
├── logs/                  # Application logs
├── docs/                  # Documentation
├── scripts/               # Utility scripts
├── requirements.txt
└── README.md
```

### 1.5 Log Configuration & Verification

#### Create Log Rotation Configuration

```bash
# Apache log rotation
sudo nano /etc/logrotate.d/apache2-waf
```

```conf
/var/log/apache2/access.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 www-data adm
    sharedscripts
    postrotate
        systemctl reload apache2 > /dev/null
    endscript
}
```

#### Verify Log Generation

```bash
# Generate test requests
for i in {1..10}; do
    curl -s http://localhost:8080/test?param=value$i
    curl -s -X POST http://localhost:8080/api/data -d "data=test$i"
done

# Check logs
tail -f /var/log/apache2/access.log
# or for Nginx
tail -f /var/log/nginx/access.log
```

### 1.6 Configuration Files

#### Create Main Configuration

```bash
cat > config/config.yaml << EOF
# Web Server Configuration
web_server:
  type: "apache"  # or "nginx"
  log_path: "/var/log/apache2/access.log"
  error_log_path: "/var/log/apache2/error.log"
  log_format: "detailed"
  
# Application Configuration
applications:
  app1:
    port: 8080
    log_path: "/opt/tomcat9/logs/localhost_access_log.txt"
  app2:
    port: 8081
    log_path: "/opt/tomcat9/logs/localhost_access_log.txt"
  app3:
    port: 8082
    log_path: "/opt/tomcat9/logs/localhost_access_log.txt"

# Data Paths
data:
  raw_logs: "data/raw"
  processed: "data/processed"
  normalized: "data/normalized"
  training: "data/training"
  validation: "data/validation"
  test: "data/test"

# Model Configuration
model:
  checkpoint_dir: "models/checkpoints"
  vocabulary_dir: "models/vocabularies"
  deployed_dir: "models/deployed"

# WAF Service
waf_service:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 5.0
EOF
```

#### Create Environment File

```bash
cat > .env << EOF
# Environment
ENV=development

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/waf.log

# Model
MODEL_NAME=transformer_waf
MODEL_VERSION=1.0.0

# Detection Threshold
ANOMALY_THRESHOLD=0.7
EOF
```

## Testing & Validation

### Test Script

```python
# scripts/test_setup.py
import requests
import os
import sys

def test_web_applications():
    """Test if all web applications are accessible"""
    apps = [
        ("http://localhost:8080", "App 1"),
        ("http://localhost:8081", "App 2"),
        ("http://localhost:8082", "App 3")
    ]
    
    for url, name in apps:
        try:
            response = requests.get(url, timeout=5)
            print(f"✓ {name} ({url}): Status {response.status_code}")
        except Exception as e:
            print(f"✗ {name} ({url}): Error - {e}")
            return False
    return True

def test_log_files():
    """Verify log files exist and are writable"""
    log_paths = [
        "/var/log/apache2/access.log",
        "/opt/tomcat9/logs/localhost_access_log.txt"
    ]
    
    for log_path in log_paths:
        if os.path.exists(log_path):
            if os.access(log_path, os.R_OK):
                print(f"✓ Log file readable: {log_path}")
            else:
                print(f"✗ Log file not readable: {log_path}")
                return False
        else:
            print(f"✗ Log file not found: {log_path}")
            return False
    return True

if __name__ == "__main__":
    print("Testing Environment Setup...")
    print("\n1. Testing Web Applications:")
    apps_ok = test_web_applications()
    
    print("\n2. Testing Log Files:")
    logs_ok = test_log_files()
    
    if apps_ok and logs_ok:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
```

Run the test:
```bash
python scripts/test_setup.py
```

## Deliverables Checklist

- [ ] Apache/Nginx installed and configured
- [ ] 3 WAR applications deployed and accessible
- [ ] Detailed access logging enabled
- [ ] Python 3.9+ virtual environment created
- [ ] All dependencies installed
- [ ] Project directory structure created
- [ ] Configuration files created
- [ ] Log files being generated and accessible
- [ ] Test script passes all checks

## Common Issues & Solutions

### Issue: Tomcat not starting
**Solution:** Check JAVA_HOME environment variable
```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

### Issue: Permission denied on log files
**Solution:** Add user to appropriate groups
```bash
sudo usermod -a -G adm www-data
sudo chmod 644 /var/log/apache2/access.log
```

### Issue: Port already in use
**Solution:** Find and kill process or change port
```bash
sudo lsof -i :8080
sudo kill -9 <PID>
```

## Next Steps

After completing Phase 1, you should have:
- A fully functional web server environment
- Three running web applications
- Comprehensive logging in place
- A clean Python development environment
- Organized project structure

**Proceed to Phase 2:** Log Ingestion System

## References

- [Apache Logging Documentation](https://httpd.apache.org/docs/current/logs.html)
- [Nginx Logging Documentation](https://nginx.org/en/docs/http/ngx_http_log_module.html)
- [Awesome WAF - Testing Methodology](https://github.com/0xInfection/awesome-waf#testing-methodology)
