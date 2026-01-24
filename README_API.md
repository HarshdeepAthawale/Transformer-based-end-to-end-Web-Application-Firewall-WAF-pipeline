# WAF API Server - Quick Start Guide

## Overview

This is the complete backend API server for the Transformer-based WAF Dashboard. It provides:
- REST API endpoints for all dashboard data
- WebSocket server for real-time updates
- Database persistence (SQLite/PostgreSQL)
- Background workers for log processing and metrics aggregation
- Full integration with the ML pipeline

## Prerequisites

- Python 3.9+
- SQLite (default) or PostgreSQL
- Trained model files (optional, for ML detection):
  - `models/checkpoints/best_model.pt`
  - `models/vocabularies/http_vocab.json`

## Installation

1. Install dependencies:
```bash
pip install fastapi uvicorn sqlalchemy psutil loguru pyyaml
```

2. Ensure database directory exists:
```bash
mkdir -p data
```

## Configuration

Edit `config/config.yaml` to configure:
- API server host/port
- Database URL
- WebSocket settings
- Log ingestion path

## Running the Server

### Development Mode
```bash
python scripts/start_api_server.py
```

Or directly:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 3001 --reload
```

### Production Mode
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 3001 --workers 4
```

## API Endpoints

### Metrics
- `GET /api/metrics/realtime` - Get real-time metrics
- `GET /api/metrics/historical?range=24h` - Get historical metrics

### Alerts
- `GET /api/alerts/active` - Get active alerts
- `GET /api/alerts/history?range=24h` - Get alert history
- `POST /api/alerts/{id}/dismiss` - Dismiss an alert
- `POST /api/alerts/{id}/acknowledge` - Acknowledge an alert

### Activities
- `GET /api/activities/recent?limit=10` - Get recent activities
- `GET /api/activities?range=24h` - Get activities by time range

### Traffic
- `GET /api/traffic/recent?limit=50` - Get recent traffic logs
- `GET /api/traffic?range=24h` - Get traffic by time range
- `GET /api/traffic/endpoint/{endpoint}?range=24h` - Get traffic for endpoint

### Threats
- `GET /api/threats/recent?limit=20` - Get recent threats
- `GET /api/threats?range=24h` - Get threats by time range
- `GET /api/threats/type/{type}?range=24h` - Get threats by type
- `GET /api/threats/stats?range=24h` - Get threat statistics

### Charts
- `GET /api/charts/requests?range=24h` - Get requests chart data
- `GET /api/charts/threats?range=24h` - Get threats chart data
- `GET /api/charts/performance?range=24h` - Get performance chart data

### Security
- `GET /api/security/checks` - Get security checks
- `POST /api/security/checks/{id}/run` - Run a security check
- `GET /api/security/compliance-score` - Get compliance score

### Analytics
- `GET /api/analytics/overview?range=24h` - Get analytics overview
- `GET /api/analytics/trends/{metric}?range=24h` - Get trends for metric
- `GET /api/analytics/summary?range=24h` - Get analytics summary

## WebSocket

Connect to `ws://localhost:3001/ws/` for real-time updates.

Message types:
- `metrics` - Real-time metrics updates
- `alert` - New alerts
- `activity` - Activity feed updates
- `threat` - Threat detections
- `traffic` - Traffic log updates

## Database

The database is automatically initialized on first run. Tables created:
- `metrics` - Metrics snapshots
- `alerts` - Security alerts
- `traffic_logs` - HTTP request logs
- `threats` - Detected threats
- `activities` - Activity feed

## Background Workers

The server automatically starts:
1. **Log Processor** - Processes logs from configured log path
2. **Metrics Aggregator** - Aggregates metrics every 60 seconds

## Health Check

```bash
curl http://localhost:3001/health
```

## Frontend Integration

The frontend should connect to:
- API: `http://localhost:3001`
- WebSocket: `ws://localhost:3001/ws/`

Set environment variables:
```bash
NEXT_PUBLIC_API_URL=http://localhost:3001
NEXT_PUBLIC_WS_URL=ws://localhost:3001
```

## Troubleshooting

1. **Database errors**: Ensure `data/` directory exists and is writable
2. **Model not found**: WAF service will run without ML detection if model files are missing
3. **Log ingestion errors**: Check log path in config and ensure file exists
4. **WebSocket connection issues**: Ensure single worker mode for WebSocket support

## Next Steps

- Add authentication/authorization
- Implement IP fencing
- Add geo-fencing
- Integrate threat intelligence feeds
- Add bot detection
- Implement response inspection/DLP
