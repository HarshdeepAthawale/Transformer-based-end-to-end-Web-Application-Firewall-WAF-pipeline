# Transformer-based End-to-End Web Application Firewall (WAF) Pipeline

A comprehensive Transformer-based anomaly detection system for web application security.

## Project Overview

This project implements a complete WAF pipeline using Transformer models to detect anomalous HTTP requests. The system learns from normal traffic patterns and identifies deviations that may indicate attacks.

## Phase 1: Environment Setup & Web Application Deployment

This phase establishes the foundation for the entire WAF pipeline by:
- Setting up web servers (Nginx) with detailed logging
- Deploying 3 sample web applications
- Configuring Python development environment
- Creating organized project structure
- Verifying log generation

### Quick Start

1. **Run the setup script:**
   ```bash
   bash scripts/setup_phase1.sh
   ```

2. **Verify the setup:**
   ```bash
   python scripts/test_setup.py
   ```

3. **Generate test traffic:**
   ```bash
   # Test applications
   curl http://localhost:8080
   curl http://localhost:8081
   curl http://localhost:8082
   ```

### Manual Setup

If you prefer to set up manually:

1. **Install Python dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Install and configure Nginx:**
   ```bash
   sudo apt-get install nginx
   # Configuration will be handled by setup script
   ```

3. **Install Tomcat and deploy applications:**
   ```bash
   bash scripts/create_war_apps.sh
   bash scripts/configure_tomcat.sh
   ```

### Alternative: Python Web Apps

If Java/Tomcat setup is problematic, you can use Python-based web applications:

```bash
# Run all three apps
python scripts/simple_web_apps.py

# Or run individually
python scripts/simple_web_apps.py 1  # App 1 on port 8080
python scripts/simple_web_apps.py 2  # App 2 on port 8081
python scripts/simple_web_apps.py 3  # App 3 on port 8082
```

### Project Structure

```
.
├── src/                    # Source code
│   ├── ingestion/         # Log ingestion modules
│   ├── parsing/           # Request parsing modules
│   ├── tokenization/      # Tokenization modules
│   ├── model/             # Model architecture
│   ├── training/          # Training scripts
│   ├── inference/         # Inference engine
│   ├── integration/       # Web server integration
│   └── learning/          # Continuous learning
├── data/                   # Data directories
│   ├── raw/               # Raw log files
│   ├── processed/         # Processed logs
│   ├── normalized/        # Normalized requests
│   ├── training/          # Training datasets
│   ├── validation/        # Validation datasets
│   └── test/              # Test datasets
├── models/                 # Model files
│   ├── checkpoints/       # Model checkpoints
│   ├── vocabularies/      # Tokenizer vocabularies
│   └── deployed/          # Production models
├── config/                 # Configuration files
├── tests/                  # Test suites
├── logs/                   # Application logs
├── scripts/                # Utility scripts
└── docs/                   # Documentation
```

### Configuration

- **config/config.yaml**: Main configuration file
- **.env**: Environment variables (create from template if needed)

### Testing

Run the test script to verify everything is working:

```bash
python scripts/test_setup.py
```

The test script checks:
- Project structure
- Configuration files
- Python environment
- Web applications accessibility
- Log file accessibility

### Troubleshooting

**Port already in use:**
```bash
sudo lsof -i :8080
sudo kill -9 <PID>
```

**Permission denied on log files:**
```bash
sudo chmod 644 /var/log/nginx/access.log
sudo usermod -a -G adm $USER
```

**Tomcat not starting:**
```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
/opt/tomcat9/bin/startup.sh
```

### Next Steps

After completing Phase 1, proceed to:
- **Phase 2**: Log Ingestion System
- See `docs/README.md` for full documentation

## Requirements

- Python 3.9+
- Java JDK 8+ (for WAR applications)
- Nginx or Apache
- 8GB RAM minimum
- 20GB disk space

## License

[Add your license here]
