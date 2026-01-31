# Transformer-based End-to-End Web Application Firewall (WAF) Pipeline

A comprehensive machine learning pipeline for building and deploying a Transformer-based anomaly detection system for web application security. This system learns from normal HTTP traffic patterns and identifies anomalous requests that may indicate security threats or attacks.

## Overview

This project implements a complete Web Application Firewall (WAF) solution using Transformer neural networks to detect anomalous HTTP requests in real-time. Unlike traditional signature-based WAFs, this system employs unsupervised anomaly detection, learning the baseline behavior of legitimate traffic and flagging deviations that could represent attacks, exploits, or malicious activity.

The system operates as a reverse proxy at Layer 7 (Application Layer), intercepting HTTP/HTTPS requests before they reach backend applications. It processes requests through a multi-stage pipeline: log ingestion, request parsing and normalization, tokenization, and Transformer-based inference to generate anomaly scores.

## Architecture

The WAF pipeline consists of several interconnected components that process web traffic from ingestion to detection:

```
Client Request → Web Server → WAF Service → Transformer Model → Anomaly Score → Action (Allow/Block)
```

### Core Pipeline

1. **Log Ingestion**: Captures HTTP access logs from web servers (Nginx/Apache) in batch or streaming mode
2. **Request Parsing**: Extracts structured data from log entries, handling multiple log formats
3. **Normalization**: Removes dynamic values (UUIDs, timestamps, tokens) to focus on request structure
4. **Tokenization**: Converts normalized requests into token sequences suitable for Transformer models
5. **Model Inference**: DistilBERT-based model generates anomaly scores for each request
6. **Decision Making**: Threshold-based classification determines whether to allow or block requests

### Deployment Architecture

The system integrates with web servers through a microservice architecture. A FastAPI-based WAF service receives requests from the web server (via reverse proxy configuration), performs inference, and returns decisions. The web server then forwards legitimate requests to backend applications or blocks suspicious traffic.

## Key Features

### Anomaly Detection
- Unsupervised learning from benign traffic patterns
- DistilBERT-based Transformer architecture for sequence understanding
- Configurable anomaly thresholds with optimization support
- Multiple loss functions (MSE, weighted MSE, focal loss, contrastive loss)

### Real-Time Protection
- Non-blocking inference for low-latency request processing
- FastAPI-based microservice architecture
- Integration with Nginx and Apache web servers
- Configurable timeout and worker pool settings

### Training and Evaluation
- Comprehensive training pipeline with early stopping
- Multiple evaluation metrics (TPR, FPR, precision, recall, F1, ROC-AUC)
- Threshold optimization for target false positive rates
- Data augmentation for improved model generalization
- Automated report generation with performance metrics

### Continuous Learning
- Incremental data collection from production traffic
- Fine-tuning pipeline for model updates
- Support for model versioning and deployment

### Request Processing
- Support for multiple log formats (Nginx, Apache)
- Dynamic value normalization (UUIDs, timestamps, session tokens)
- HTTP-aware tokenization preserving request structure
- Sequence preparation with padding and truncation

## Technology Stack

### Machine Learning
- **PyTorch**: Deep learning framework for model training and inference
- **Transformers (Hugging Face)**: Pre-trained DistilBERT model and tokenization utilities
- **scikit-learn**: Evaluation metrics and data processing utilities

### Web Framework
- **FastAPI**: High-performance API framework for WAF service
- **Uvicorn**: ASGI server for FastAPI deployment

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations

### Utilities
- **Loguru**: Structured logging
- **PyYAML**: Configuration management
- **Pydantic**: Data validation and settings management

### Model Architecture
- **Base Model**: DistilBERT (distilled version of BERT)
- **Architecture**: 6 transformer layers, 12 attention heads, 768 hidden dimensions
- **Output**: Sigmoid-activated anomaly score (0.0 to 1.0)

## Project Structure

```
.
├── src/                    # Source code modules
│   ├── ingestion/         # Log ingestion (batch and streaming)
│   ├── parsing/           # HTTP request parsing and normalization
│   ├── tokenization/      # Tokenization and sequence preparation
│   ├── model/             # Transformer model architecture
│   ├── training/          # Training pipeline and utilities
│   ├── inference/         # Inference engine
│   ├── integration/       # Web server integration
│   └── learning/          # Continuous learning components
├── data/                   # Data directories
│   ├── raw/               # Raw log files
│   ├── processed/         # Parsed log data
│   ├── normalized/        # Normalized requests
│   ├── training/          # Training datasets
│   ├── validation/        # Validation datasets
│   └── test/              # Test datasets
├── models/                 # Model artifacts
│   ├── checkpoints/       # Training checkpoints
│   ├── vocabularies/      # Tokenizer vocabularies
│   └── deployed/          # Production-ready models
├── config/                 # Configuration files
│   └── config.yaml        # Main configuration
├── scripts/                # Utility scripts
├── tests/                  # Test suites
├── docs/                   # Documentation
└── logs/                   # Application logs
```

## Components

### Log Ingestion (`src/ingestion/`)
Handles collection of HTTP access logs from web servers. Supports both batch processing of historical logs and real-time streaming of live log files. Includes format detection, retry logic, and queue management for high-throughput scenarios.

### Request Parsing and Normalization (`src/parsing/`)
Extracts structured information from log entries, including HTTP method, path, query parameters, headers, and request body. Normalization engine removes dynamic values such as UUIDs, timestamps, session tokens, and IP addresses to create canonical request representations.

### Tokenization (`src/tokenization/`)
HTTP-aware tokenizer that preserves the structure of HTTP requests while converting them into token sequences. Builds vocabulary from training data and handles sequence preparation with padding and truncation for model input.

### Model Architecture (`src/model/`)
DistilBERT-based anomaly detection model with custom classification head. The model processes tokenized request sequences and outputs anomaly probability scores. Includes scoring utilities for threshold-based classification.

### Training Pipeline (`src/training/`)
Complete training infrastructure with support for multiple loss functions, early stopping, learning rate scheduling, and evaluation metrics. Includes data augmentation, threshold optimization, and automated report generation.

### Inference Engine (`src/inference/`)
High-performance inference module for real-time request evaluation. Optimized for low-latency processing with batch support and caching capabilities.

### Integration (`src/integration/`)
Web server integration components for deploying the WAF as a reverse proxy. Includes FastAPI service implementation and configuration templates for Nginx and Apache.

### Continuous Learning (`src/learning/`)
Components for incremental model updates based on new traffic patterns. Supports fine-tuning workflows and model versioning.

## Configuration

The system is configured through `config/config.yaml`, which defines:

- **Web Server Settings**: Log paths, formats, and application configurations
- **Data Paths**: Directories for raw logs, processed data, and datasets
- **Model Configuration**: Architecture parameters (hidden size, layers, attention heads)
- **Training Parameters**: Epochs, batch size, learning rate, loss function selection
- **WAF Service**: Host, port, workers, timeout settings
- **Evaluation Settings**: Metrics, threshold optimization, test split ratios

Configuration supports environment-specific overrides and can be extended for additional features.

## Documentation

Comprehensive documentation is available in the `docs/` directory, covering:

- Phase-by-phase implementation guides
- Architecture decisions and design patterns
- API reference and integration examples
- Testing and validation procedures
- Deployment and operational considerations

See `docs/README.md` for the complete documentation index.
  


## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend Dashboard                       │
│                   (Next.js on port 3000)                    │
│         Real-time metrics, traffic logs, charts             │
└──────────────────────┬──────────────────────────────────────┘
                       │ REST API + WebSocket
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                     Backend API Server                       │
│                  (FastAPI on port 3001)                     │
│         WAF Middleware + ML Model + Database                │
└──┬─────────┬────────┬────────────────────────────────┬──────┘
   │         │        │                                │
   ↓         ↓        ↓                                ↓
┌─────┐  ┌──────┐ ┌──────────────────────────┐   ┌─────────┐
│Redis│  │Postgres│WAF ML Model               │   │WebSocket│
└─────┘  └──────┘ │(DistilBERT Fine-tuned)   │   └─────────┘
                  └──────────────────────────┘

                 ↓ Protects ↓

┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Juice Shop  │  │   WebGoat    │  │     DVWA     │
│  Port 8080   │  │  Port 8081   │  │  Port 8082   │
└──────────────┘  └──────────────┘  └──────────────┘
```

---