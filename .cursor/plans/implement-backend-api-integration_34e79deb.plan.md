---
name: implement-backend-api-integration
overview: Implement complete backend API layer to integrate with frontend, including FastAPI server, WebSocket support, data persistence, and ML pipeline integration
todos:
  - id: create-fastapi-server
    content: Create FastAPI server infrastructure with main.py, config.py, database.py, and websocket.py
    status: pending
  - id: implement-database-models
    content: Implement SQLAlchemy data models for metrics, alerts, traffic, threats, security, and activities
    status: pending
  - id: create-rest-api-endpoints
    content: Implement all REST API endpoints (metrics, alerts, activities, charts, traffic, threats, security, analytics)
    status: pending
  - id: implement-websocket-server
    content: Create WebSocket server for real-time updates (metrics, alerts, activities, threats, traffic)
    status: pending
  - id: create-data-services
    content: Implement data collection and aggregation services (metrics, alerts, traffic, threats, security)
    status: pending
  - id: integrate-ml-pipeline
    content: Integrate existing ML pipeline (anomaly detection, log parsing) with API endpoints
    status: pending
  - id: create-background-workers
    content: Implement background workers for log processing, metrics aggregation, and data cleanup
    status: pending
  - id: update-configuration
    content: Update configuration files and create Docker setup for backend services
    status: pending
isProject: false
---

## Overview

The frontend has been updated to consume real-time API endpoints, but the backend is missing the complete API layer. The ML pipeline (ingestion, parsing, tokenization, model) exists, but there's no REST API server, WebSocket support, or data persistence layer. This plan implements the missing backend components to fully integrate frontend and backend.

## Current Backend State

- ✅ ML Pipeline: Ingestion → Parsing → Tokenization → Model → Training
- ❌ API Server: FastAPI REST endpoints (documented but not implemented)
- ❌ WebSocket Server: Real-time updates (mentioned but not implemented)
- ❌ Database: Data persistence for metrics/alerts/traffic (not implemented)
- ❌ Real-time Services: Data collection and aggregation (not implemented)

## Implementation Plan

### 1. Create FastAPI Server Infrastructure

- Set up FastAPI application with proper middleware and CORS
- Implement dependency injection for database and ML components
- Add proper error handling and response models
- Configure environment variables and configuration loading

### 2. Database Layer Implementation

- Set up SQLite/PostgreSQL database with SQLAlchemy
- Create data models for:
- Metrics (real-time and historical)
- Alerts (active and historical)
- Traffic logs (recent and historical)
- Threats (detected attacks)
- Security checks (compliance status)
- Activities (system events)
- Implement database migrations and connection pooling

### 3. REST API Endpoints Implementation

- **Metrics API**: `/api/metrics/realtime`, `/api/metrics/historical`
- **Alerts API**: `/api/alerts/active`, `/api/alerts/history`, `/api/alerts/{id}/dismiss`
- **Activities API**: `/api/activities/recent`, `/api/activities`
- **Charts API**: `/api/charts/requests`, `/api/charts/threats`, `/api/charts/performance`
- **Traffic API**: `/api/traffic/recent`, `/api/traffic`, `/api/traffic/endpoint/{endpoint}`
- **Threats API**: `/api/threats/recent`, `/api/threats`, `/api/threats/stats`
- **Security API**: `/api/security/checks`, `/api/security/compliance-score`
- **Analytics API**: `/api/analytics/overview`, `/api/analytics/trends`

### 4. WebSocket Implementation

- Implement WebSocket server using FastAPI WebSockets
- Create real-time broadcasting for:
- Live metrics updates
- New alerts notifications
- Traffic activity updates
- Threat detection alerts
- Add connection management and heartbeat monitoring

### 5. Real-time Data Collection Services

- **Metrics Collector**: Aggregate data from log streams and model inference
- **Alert Generator**: Create alerts from anomaly detections and threshold breaches
- **Traffic Monitor**: Track and analyze HTTP traffic patterns
- **Threat Detector**: Process model predictions and generate threat intelligence
- **Activity Logger**: Record system events and user actions

### 6. ML Pipeline Integration

- Integrate anomaly detection model with API endpoints
- Create `/check` endpoint for single request analysis
- Implement `/check/batch` endpoint for bulk analysis
- Add model health monitoring and performance metrics
- Implement continuous learning data collection

### 7. Background Services

- Log ingestion workers (batch and streaming)
- Metrics aggregation workers
- Alert processing workers
- Data cleanup and archiving workers
- Model retraining schedulers

### 8. Configuration and Deployment

- Update configuration files for API server settings
- Add Docker configuration for backend services
- Create health check endpoints
- Implement proper logging and monitoring

## Files to Create

### Core API Server

- `src/api/__init__.py` - API package initialization
- `src/api/main.py` - FastAPI application entry point
- `src/api/config.py` - API configuration and settings
- `src/api/database.py` - Database connection and session management
- `src/api/websocket.py` - WebSocket connection management

### Data Models

- `src/api/models/__init__.py` - Models package
- `src/api/models/metrics.py` - Metrics data models
- `src/api/models/alerts.py` - Alert data models
- `src/api/models/traffic.py` - Traffic data models
- `src/api/models/threats.py` - Threat data models
- `src/api/models/security.py` - Security check models
- `src/api/models/activities.py` - Activity log models

### API Routes

- `src/api/routes/__init__.py` - Routes package
- `src/api/routes/metrics.py` - Metrics endpoints
- `src/api/routes/alerts.py` - Alert management endpoints
- `src/api/routes/activities.py` - Activity endpoints
- `src/api/routes/charts.py` - Chart data endpoints
- `src/api/routes/traffic.py` - Traffic monitoring endpoints
- `src/api/routes/threats.py` - Threat detection endpoints
- `src/api/routes/security.py` - Security endpoints
- `src/api/routes/analytics.py` - Analytics endpoints
- `src/api/routes/waf.py` - WAF service endpoints

### Services

- `src/api/services/__init__.py` - Services package
- `src/api/services/metrics_service.py` - Metrics collection and aggregation
- `src/api/services/alert_service.py` - Alert generation and management
- `src/api/services/traffic_service.py` - Traffic analysis and monitoring
- `src/api/services/threat_service.py` - Threat detection and analysis
- `src/api/services/security_service.py` - Security compliance checking
- `src/api/services/analytics_service.py` - Analytics and reporting
- `src/api/services/websocket_service.py` - Real-time broadcasting

### Background Tasks

- `src/api/tasks/__init__.py` - Background tasks package
- `src/api/tasks/log_processor.py` - Log processing workers
- `src/api/tasks/metrics_aggregator.py` - Metrics aggregation
- `src/api/tasks/alert_processor.py` - Alert processing
- `src/api/tasks/data_cleanup.py` - Data maintenance

## Integration Points

### ML Pipeline Integration

- Connect existing `AnomalyDetector` to `/check` endpoints
- Use `LogIngestionSystem` to feed real-time data collection
- Integrate `HTTPRequest` parsing with traffic monitoring
- Connect model scoring with alert generation

### Frontend Integration

- All existing API calls in `frontend/lib/api.ts` will work
- WebSocket connections will provide real-time updates
- Data persistence ensures consistency across sessions
- Error handling provides proper user feedback

## Expected Outcomes

- Complete backend API server with all documented endpoints
- Real-time WebSocket updates for live dashboard data
- Persistent data storage for all dashboard metrics
- Integration with existing ML anomaly detection pipeline
- Production-ready backend services with proper error handling
- Full frontend-backend integration for real-time WAF monitoring