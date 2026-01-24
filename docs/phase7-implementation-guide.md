# Phase 7: Real-Time Non-Blocking Detection - Implementation Guide

## Overview

Phase 7 implements a complete async, non-blocking WAF service capable of handling high-throughput scenarios with concurrent request processing, batching, rate limiting, and model optimization.

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP Request
       ▼
┌─────────────────────────────────────┐
│      FastAPI Application            │
│  ┌───────────────────────────────┐   │
│  │   AsyncWAFService            │   │
│  │  - Thread Pool Executor      │   │
│  │  - Batch Processing          │   │
│  │  - Timeout Handling          │   │
│  └───────────────────────────────┘   │
│                                      │
│  ┌───────────────────────────────┐   │
│  │   RequestQueueManager        │   │
│  │  - Async Queue               │   │
│  │  - Batch Collection          │   │
│  └───────────────────────────────┘   │
│                                      │
│  ┌───────────────────────────────┐   │
│  │   RateLimiter                │   │
│  │  - Global/Per-IP Limiting    │   │
│  └───────────────────────────────┘   │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│   Model Inference (Thread Pool)     │
│  - Batch Processing                 │
│  - Optimized Inference              │
└─────────────────────────────────────┘
```

## Components

### 1. AsyncWAFService (`src/inference/async_waf_service.py`)

Main service class providing async request processing.

**Features:**
- Non-blocking async request processing
- Thread pool executor for CPU-bound operations
- Batch processing support
- Timeout handling
- Metrics collection
- Automatic model architecture inference

**Usage:**
```python
from src.inference.async_waf_service import AsyncWAFService

service = AsyncWAFService(
    model_path="models/checkpoints/best_model.pt",
    vocab_path="models/vocabularies/http_vocab.json",
    threshold=0.5,
    device="cpu",
    max_workers=4,
    batch_size=32,
    timeout=5.0
)

# Check single request
result = await service.check_request_async(
    method="GET",
    path="/api/users",
    query_params={"id": "123"},
    headers={"User-Agent": "Mozilla/5.0"},
    body=None
)

# Check batch of requests
results = await service.check_batch_async([
    {"method": "GET", "path": "/api/test1", ...},
    {"method": "POST", "path": "/api/test2", ...},
])
```

### 2. RequestQueueManager (`src/inference/queue_manager.py`)

Manages async request queue for batching and concurrent processing.

**Features:**
- Async queue with configurable size
- Automatic batch collection with timeout
- Result queue per request
- Background processing task

**Usage:**
```python
from src.inference.queue_manager import RequestQueueManager

queue_manager = RequestQueueManager(
    max_size=1000,
    batch_timeout=0.1,
    batch_size=32
)

# Define processor function
async def processor(requests):
    # Process batch
    return results

# Start processing
queue_manager.start_processing(processor)

# Enqueue request
result = await queue_manager.enqueue({
    'method': 'GET',
    'path': '/api/test',
    ...
})
```

### 3. RateLimiter (`src/inference/rate_limiter.py`)

Rate limiting for request throttling.

**Features:**
- Global rate limiting
- Per-IP rate limiting
- Configurable window and limits
- Wait time calculation

**Usage:**
```python
from src.inference.rate_limiter import RateLimiter, PerIPRateLimiter

# Global rate limiter
limiter = RateLimiter(max_requests=100, window_seconds=1)
if limiter.is_allowed():
    # Process request
    pass

# Per-IP rate limiter
ip_limiter = PerIPRateLimiter(max_requests=10, window_seconds=1)
if ip_limiter.is_allowed("192.168.1.1"):
    # Process request
    pass
```

### 4. Model Optimization (`src/inference/optimization.py`)

Model optimization for faster inference.

**Features:**
- Dynamic quantization (INT8)
- TorchScript compilation
- Model loading with optimization

**Usage:**
```python
from src.inference.optimization import load_optimized_model, optimize_model

# Load and optimize model
model = load_optimized_model(
    model_path="models/checkpoints/best_model.pt",
    optimization="quantization",  # or "torchscript"
    device="cpu"
)

# Or optimize existing model
optimized_model = optimize_model(model, method="quantization")
```

## Configuration

### Main Config (`config/config.yaml`)

```yaml
waf_service:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 5.0
  model_path: "models/checkpoints/best_model.pt"
  vocab_path: "models/vocabularies/http_vocab.json"
  threshold: 0.5
```

### Inference Config (`config/inference.yaml`)

```yaml
inference:
  async:
    max_workers: 4
    batch_size: 32
    timeout: 5.0
  
  optimization:
    quantization: true
    torchscript: false
  
  rate_limiting:
    enabled: true
    max_requests_per_second: 100
    per_ip: false
    max_ips: 10000
  
  queue:
    max_size: 1000
    batch_timeout: 0.1
    batch_size: 32
```

## Running the Service

### Using Startup Script

```bash
python scripts/start_async_waf_service.py
```

The script will:
1. Load configuration from `config/config.yaml` and `config/inference.yaml`
2. Check for model and vocabulary files
3. Initialize the async WAF service
4. Start the FastAPI server

### Using uvicorn directly

```bash
uvicorn src.inference.async_waf_service:app --host 0.0.0.0 --port 8000
```

**Note:** You must initialize the service first:
```python
from src.inference.async_waf_service import initialize_service

initialize_service(
    model_path="models/checkpoints/best_model.pt",
    vocab_path="models/vocabularies/http_vocab.json",
    threshold=0.5
)
```

## API Endpoints

### POST `/check`

Check single request for anomalies.

**Request:**
```json
{
  "method": "GET",
  "path": "/api/users",
  "query_params": {"id": "123"},
  "headers": {"User-Agent": "Mozilla/5.0"},
  "body": null
}
```

**Response:**
```json
{
  "anomaly_score": 0.15,
  "is_anomaly": false,
  "threshold": 0.5,
  "processing_time_ms": 45.2
}
```

### POST `/check/batch`

Check batch of requests.

**Request:**
```json
[
  {"method": "GET", "path": "/api/test1", ...},
  {"method": "POST", "path": "/api/test2", ...}
]
```

**Response:**
```json
[
  {"anomaly_score": 0.1, "is_anomaly": false, ...},
  {"anomaly_score": 0.8, "is_anomaly": true, ...}
]
```

### GET `/metrics`

Get service metrics.

**Response:**
```json
{
  "total_requests": 1000,
  "anomalies_detected": 50,
  "anomaly_rate": 0.05,
  "avg_processing_time_ms": 42.5,
  "device": "cpu",
  "threshold": 0.5,
  "batch_size": 32
}
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "async_waf",
  "model_loaded": true
}
```

## Testing

### Run Performance Tests

```bash
pytest tests/performance/test_concurrent.py -v
```

**Tests include:**
- Concurrent request processing
- Batch processing
- Timeout handling
- Rate limiting
- Queue management
- Metrics collection

### Manual Testing

```python
import requests

# Check single request
response = requests.post("http://localhost:8000/check", json={
    "method": "GET",
    "path": "/api/test",
    "query_params": {},
    "headers": {},
    "body": None
})
print(response.json())

# Check batch
response = requests.post("http://localhost:8000/check/batch", json=[
    {"method": "GET", "path": "/api/test1", ...},
    {"method": "GET", "path": "/api/test2", ...},
])
print(response.json())

# Get metrics
response = requests.get("http://localhost:8000/metrics")
print(response.json())
```

## Performance Considerations

### Thread Pool Size

- **CPU-bound operations**: Set `max_workers` to number of CPU cores
- **I/O-bound operations**: Can use more workers (2-4x CPU cores)
- **Default**: 4 workers

### Batch Size

- **Small batches (8-16)**: Lower latency, higher overhead
- **Large batches (32-64)**: Higher throughput, higher latency
- **Default**: 32

### Timeout

- **Short timeout (1-2s)**: Fast failure, may drop valid requests
- **Long timeout (5-10s)**: Better reliability, slower failure detection
- **Default**: 5.0s

### Model Optimization

- **Quantization**: ~2-4x speedup, minimal accuracy loss
- **TorchScript**: ~1.5-2x speedup, better for production
- **Both**: Maximum speedup but may have compatibility issues

## Integration with Existing WAF Service

The async WAF service can be used alongside or replace the existing WAF service:

```python
# Option 1: Use async service directly
from src.inference.async_waf_service import AsyncWAFService

# Option 2: Use existing service (synchronous)
from src.integration.waf_service import WAFService

# Option 3: Use both (async for high-throughput, sync for low-latency)
```

## Troubleshooting

### Model Not Found

**Error:** `Model checkpoint not found`

**Solution:** Train a model first or update `model_path` in config.yaml

### Vocabulary Not Found

**Error:** `Vocabulary file not found`

**Solution:** Generate vocabulary first or update `vocab_path` in config.yaml

### Timeout Errors

**Error:** Request timeouts

**Solution:**
- Increase `timeout` in config
- Reduce `batch_size` for faster processing
- Increase `max_workers` for more parallelism

### Queue Full

**Error:** `queue_full` in response

**Solution:**
- Increase `max_size` in queue config
- Reduce request rate
- Enable rate limiting

## Next Steps

After completing Phase 7, you should have:
- ✅ Non-blocking concurrent request processing
- ✅ Optimized model inference
- ✅ High-throughput capability
- ✅ Rate limiting and queue management
- ✅ Comprehensive metrics

**Proceed to Phase 8:** Continuous Learning & Incremental Updates
