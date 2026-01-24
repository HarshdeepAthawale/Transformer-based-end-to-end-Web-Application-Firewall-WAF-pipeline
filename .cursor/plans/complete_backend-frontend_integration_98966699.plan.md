---
name: Complete Backend-Frontend Integration & Advanced WAF Features
overview: Create a complete backend API server with database persistence, WebSocket support, full ML pipeline integration, and implement all 9 core WAF protection methods (IP fencing, geo-fencing, bot mitigation, threat intelligence, etc.) with advanced dashboard features.
todos:
  - id: database-layer
    content: Create database layer with SQLAlchemy models for metrics, alerts, traffic, threats, security, activities, IP blacklists, geo-rules, bot signatures
    status: pending
  - id: fastapi-infrastructure
    content: Set up FastAPI application with CORS, middleware, error handling, authentication, rate limiting, and dependency injection
    status: pending
  - id: rest-api-endpoints
    content: Implement all REST API endpoints matching frontend expectations plus advanced endpoints for IP management, geo-rules, bot detection, threat intel
    status: pending
  - id: websocket-server
    content: Create WebSocket server for real-time updates (metrics, alerts, activities, threats, traffic, security events)
    status: pending
  - id: data-services
    content: Implement data collection services (metrics, alerts, traffic, threats, security, analytics, IP reputation, geo-location)
    status: pending
  - id: background-workers
    content: Create background workers for log processing, metrics aggregation, alert processing, data cleanup, threat intel updates, IP reputation updates
    status: pending
  - id: ml-pipeline-integration
    content: Integrate existing ML pipeline (log ingestion, parsing, model inference) with API services and enhance with threat classification
    status: pending
  - id: ip-fencing-system
    content: Implement IP fencing with blacklist/whitelist management, reputation scoring, automatic blocking, and IP history tracking
    status: pending
  - id: geo-fencing-system
    content: Implement geo-fencing with country-based allow/deny rules, GeoIP database integration, and geographic threat analysis
    status: pending
  - id: bot-detection-mitigation
    content: Implement bot detection using User-Agent analysis, behavioral patterns, CAPTCHA integration, and automated bot blocking
    status: pending
  - id: threat-intelligence-integration
    content: Integrate threat intelligence feeds (IP reputation, malware signatures, known attack patterns) with automatic updates
    status: pending
  - id: advanced-rate-limiting
    content: Enhance rate limiting with per-IP, per-endpoint, adaptive rate limiting, and DDoS protection mechanisms
    status: pending
  - id: response-inspection-dlp
    content: Implement response inspection for data leakage prevention (DLP), sensitive data detection, and response filtering
    status: pending
  - id: security-rules-engine
    content: Create security rules engine with custom rules, OWASP Top 10 rules, signature-based detection, and rule management API
    status: pending
  - id: advanced-analytics-reporting
    content: Implement advanced analytics with threat trends, attack pattern analysis, compliance reporting, and custom dashboards
    status: pending
  - id: configuration-setup
    content: Update configuration files, create startup scripts, and add environment-based configuration management
    status: pending
  - id: remove-mock-data
    content: Remove all remaining mock/hardcoded data from frontend components and ensure all data comes from backend
    status: pending
  - id: testing-verification
    content: Test all API endpoints, WebSocket connections, security features, and verify frontend displays real data
    status: pending
  - id: performance-optimization
    content: Optimize database queries, implement caching, optimize model inference, and ensure sub-200ms API response times
    status: pending
  - id: security-hardening
    content: Implement authentication, authorization, API key management, audit logging, and security best practices
    status: pending
isProject: false
---

# Complete Backend-Frontend Integration & Advanced WAF Features Plan

## Overview

This comprehensive plan extends the basic backend-frontend integration to include all 9 core WAF protection methods and advanced enterprise features. After completing all 9 phases of the WAF pipeline, we now implement a production-ready system with advanced security capabilities, comprehensive analytics, and a fully integrated dashboard.

## Current State Analysis

### Frontend (✅ Ready)

- API client defined in `frontend/lib/api.ts` with all endpoint interfaces
- Components expect real-time data via REST API and WebSocket
- TypeScript interfaces defined for all data types
- WebSocket manager implemented for real-time updates
- Dashboard pages: Analytics, Performance, Security, Settings, Threats, Traffic

### Backend (⚠️ Partial)

- WAF service exists with basic endpoints: `/check`, `/health`, `/metrics`, `/update-threshold`, `/config`
- ML pipeline fully functional: ingestion → parsing → tokenization → model inference
- Rate limiter implemented (`src/inference/rate_limiter.py`)
- Async WAF service with queue management (`src/inference/async_waf_service.py`)
- Log parsing produces `HTTPRequest` objects
- **Missing**: Database layer, comprehensive REST API, WebSocket server, advanced security features

### Advanced Features Status

- ✅ **Anomaly Scoring**: Implemented via Transformer model
- ✅ **Rate Limiting**: Basic implementation exists
- ⚠️ **IP Fencing**: Not implemented
- ⚠️ **Geo-fencing**: Not implemented
- ⚠️ **Bot Mitigation**: Not implemented
- ⚠️ **Threat Intelligence**: Not implemented
- ⚠️ **Response Inspection/DLP**: Not implemented
- ⚠️ **Security Rules Engine**: Not implemented
- ⚠️ **Advanced Analytics**: Not implemented

## Architecture

```mermaid
flowchart TB
    subgraph Frontend["Frontend Dashboard"]
        UI[React Components]
        API[API Client]
        WS[WebSocket Manager]
        Charts[Analytics Charts]
        Settings[Settings Panel]
    end
    
    subgraph Backend["Backend API Server"]
        FastAPI[FastAPI Application]
        REST[REST Endpoints]
        WS_Server[WebSocket Server]
        Auth[Authentication/Authorization]
        Services[Data Services]
    end
    
    subgraph Security["Security Layer"]
        IPFence[IP Fencing Engine]
        GeoFence[Geo-fencing Engine]
        BotDetect[Bot Detection]
        ThreatIntel[Threat Intelligence]
        RulesEngine[Security Rules Engine]
        RateLimit[Advanced Rate Limiting]
        ResponseInsp[Response Inspector/DLP]
    end
    
    subgraph Data["Data Layer"]
        DB[(PostgreSQL/SQLite)]
        Cache[Redis Cache]
        ThreatDB[Threat Intelligence DB]
        GeoIPDB[GeoIP Database]
    end
    
    subgraph ML["ML Pipeline"]
        Ingestion[Log Ingestion]
        Parser[Log Parser]
        Model[Anomaly Detector]
        WAF[WAF Service]
        Classifier[Threat Classifier]
    end
    
    subgraph Workers["Background Workers"]
        LogWorker[Log Processor]
        MetricsWorker[Metrics Aggregator]
        AlertWorker[Alert Generator]
        ThreatWorker[Threat Intel Updater]
        IPWorker[IP Reputation Updater]
        CleanupWorker[Data Cleanup]
    end
    
    UI --> API
    UI --> WS
    API --> REST
    WS --> WS_Server
    REST --> Auth
    Auth --> Services
    REST --> Security
    Services --> DB
    Services --> Cache
    Security --> ThreatDB
    Security --> GeoIPDB
    LogWorker --> Ingestion
    Ingestion --> Parser
    Parser --> Model
    Model --> Classifier
    Classifier --> WAF
    WAF --> Security
    WAF --> Services
    Security --> Services
    Workers --> DB
    Workers --> ThreatDB
```

## Implementation Plan

### Phase 1: Enhanced Database Layer & Models

**Files to Create:**

- `src/api/database.py` - Database connection, session management, SQLAlchemy setup
- `src/api/models/__init__.py` - Models package initialization
- `src/api/models/metrics.py` - Metrics table (real-time and historical)
- `src/api/models/alerts.py` - Alerts table (active and historical)
- `src/api/models/traffic.py` - Traffic logs table with IP, geo, bot flags
- `src/api/models/threats.py` - Detected threats table with classification
- `src/api/models/security.py` - Security checks table
- `src/api/models/activities.py` - Activity log table
- `src/api/models/ip_blacklist.py` - IP blacklist/whitelist table
- `src/api/models/ip_reputation.py` - IP reputation scores and history
- `src/api/models/geo_rules.py` - Geo-fencing rules (country allow/deny)
- `src/api/models/bot_signatures.py` - Bot detection signatures
- `src/api/models/threat_intel.py` - Threat intelligence data
- `src/api/models/security_rules.py` - Custom security rules
- `src/api/models/users.py` - User authentication and authorization
- `src/api/models/audit_log.py` - Audit logging for security events

**Key Features:**

- PostgreSQL for production (SQLite for development)
- SQLAlchemy ORM with proper relationships and indexes
- Timestamp indexing for time-range queries
- Data retention policies (auto-cleanup old data)
- Full-text search capabilities for threat analysis
- Composite indexes for common query patterns

### Phase 2: FastAPI Server Infrastructure with Security

**Files to Create:**

- `src/api/main.py` - FastAPI application entry point
- `src/api/config.py` - Configuration management
- `src/api/dependencies.py` - Dependency injection
- `src/api/middleware.py` - Request logging, error handling, rate limiting
- `src/api/auth.py` - Authentication and authorization
- `src/api/security_middleware.py` - Security headers, CORS, CSRF protection

**Key Features:**

- JWT-based authentication
- Role-based access control (RBAC)
- API key management
- Rate limiting per user/IP
- CORS configuration
- Security headers (HSTS, CSP, X-Frame-Options)
- Request/response logging
- Error handling with proper HTTP status codes
- Health check endpoints with detailed status
- API versioning (`/api/v1/`)

### Phase 3: Comprehensive REST API Endpoints

**Files to Create:**

- `src/api/routes/__init__.py` - Routes package
- `src/api/routes/metrics.py` - Metrics endpoints
- `src/api/routes/alerts.py` - Alerts endpoints
- `src/api/routes/activities.py` - Activities endpoints
- `src/api/routes/charts.py` - Charts endpoints
- `src/api/routes/traffic.py` - Traffic endpoints
- `src/api/routes/threats.py` - Threats endpoints
- `src/api/routes/security.py` - Security endpoints
- `src/api/routes/analytics.py` - Analytics endpoints
- `src/api/routes/ip_management.py` - IP blacklist/whitelist management
- `src/api/routes/geo_rules.py` - Geo-fencing rules management
- `src/api/routes/bot_detection.py` - Bot detection and management
- `src/api/routes/threat_intel.py` - Threat intelligence management
- `src/api/routes/security_rules.py` - Security rules engine management
- `src/api/routes/users.py` - User management
- `src/api/routes/settings.py` - System settings and configuration

**Endpoint Specifications:**

- All endpoints match frontend expectations from `frontend/lib/api.ts`
- Support time-range queries (`?range=24h`, `?range=7d`, etc.)
- Pagination for large datasets (cursor-based or offset-based)
- Filtering and sorting capabilities
- Proper error responses matching frontend error handling
- OpenAPI/Swagger documentation

### Phase 4: WebSocket Server with Advanced Features

**Files to Create:**

- `src/api/websocket.py` - WebSocket connection manager
- `src/api/websocket_manager.py` - Connection pool management
- `src/api/websocket_auth.py` - WebSocket authentication

**Key Features:**

- Real-time broadcasting for:
                - Metrics updates (every 1-5 seconds)
                - New alerts (immediate)
                - Traffic activity (batched every 2 seconds)
                - Threat detections (immediate)
                - Activity feed updates (batched)
                - Security events (immediate)
                - IP blocking events (immediate)
                - Bot detection events (immediate)
- Connection heartbeat/ping-pong
- Automatic reconnection handling
- Message queuing for disconnected clients
- Channel-based subscriptions (users can subscribe to specific event types)
- Authentication for WebSocket connections

### Phase 5: Advanced Data Collection Services

**Files to Create:**

- `src/api/services/__init__.py` - Services package
- `src/api/services/metrics_service.py` - Aggregate metrics
- `src/api/services/alert_service.py` - Generate alerts
- `src/api/services/traffic_service.py` - Process traffic logs
- `src/api/services/threat_service.py` - Process and classify threats
- `src/api/services/security_service.py` - Security compliance checks
- `src/api/services/analytics_service.py` - Advanced analytics
- `src/api/services/ip_service.py` - IP reputation and management
- `src/api/services/geo_service.py` - Geo-location and geo-fencing
- `src/api/services/bot_service.py` - Bot detection and mitigation
- `src/api/services/threat_intel_service.py` - Threat intelligence integration
- `src/api/services/rules_service.py` - Security rules engine
- `src/api/services/dlp_service.py` - Data leakage prevention
- `src/api/services/websocket_service.py` - WebSocket broadcasting

**Service Responsibilities:**

- **Metrics Service**: Aggregate request counts, block rates, response times, system metrics
- **Alert Service**: Create alerts from anomalies, threshold breaches, security events
- **Traffic Service**: Store parsed HTTPRequest objects, enrich with IP/geo/bot data
- **Threat Service**: Classify threats (SQL injection, XSS, DDoS, etc.), track severity
- **IP Service**: Manage IP blacklists, calculate reputation scores, track IP history
- **Geo Service**: Geo-locate IPs, enforce geo-rules, track geographic threat patterns
- **Bot Service**: Detect bots via User-Agent, behavioral analysis, manage bot signatures
- **Threat Intel Service**: Integrate threat feeds, update threat database, match against known threats
- **Rules Service**: Execute security rules, manage custom rules, OWASP Top 10 rules
- **DLP Service**: Inspect responses for sensitive data, enforce data leakage policies
- **Analytics Service**: Calculate trends, generate insights, attack pattern analysis

### Phase 6: Background Workers & Scheduled Tasks

**Files to Create:**

- `src/api/tasks/__init__.py` - Background tasks package
- `src/api/tasks/log_processor.py` - Process logs from ingestion system
- `src/api/tasks/metrics_aggregator.py` - Aggregate metrics periodically
- `src/api/tasks/alert_processor.py` - Process and escalate alerts
- `src/api/tasks/data_cleanup.py` - Archive/delete old data
- `src/api/tasks/threat_intel_updater.py` - Update threat intelligence feeds
- `src/api/tasks/ip_reputation_updater.py` - Update IP reputation scores
- `src/api/tasks/geoip_updater.py` - Update GeoIP database
- `src/api/tasks/model_performance_monitor.py` - Monitor model performance
- `src/api/tasks/scheduler.py` - Task scheduler using APScheduler or Celery

**Worker Responsibilities:**

- **Log Processor**: Read from log ingestion queue, parse, check with model, enrich with security checks, store results
- **Metrics Aggregator**: Calculate aggregated metrics every minute, store in database
- **Alert Processor**: Check for alert conditions, create alerts, trigger notifications, escalate critical alerts
- **Data Cleanup**: Remove data older than retention period (configurable per data type)
- **Threat Intel Updater**: Fetch and update threat intelligence feeds (hourly/daily)
- **IP Reputation Updater**: Update IP reputation scores based on recent activity
- **GeoIP Updater**: Update GeoIP database monthly
- **Model Performance Monitor**: Track model accuracy, false positive/negative rates, suggest threshold adjustments

### Phase 7: ML Pipeline Integration & Threat Classification

**Integration Points:**

- Connect `LogIngestionSystem` to log processor worker
- Use `ParsingPipeline` to parse logs into `HTTPRequest` objects
- Feed parsed requests to `WAFService.check_request()` for anomaly detection
- Enhance with threat classification (SQL injection, XSS, DDoS, etc.)
- Store detection results in database with full context
- Generate alerts from high-scoring anomalies
- Track model performance metrics
- Integrate with security rules engine

**Files to Create/Modify:**

- `src/api/services/waf_integration.py` - Integration layer between WAF service and API
- `src/api/services/threat_classifier.py` - Enhanced threat classification
- `src/integration/waf_service.py` - Add hooks for metrics collection and security checks

**Threat Classification:**

- Pattern-based classification (SQL injection patterns, XSS patterns, etc.)
- ML-based classification using model outputs
- Hybrid approach combining both methods
- Confidence scoring for each threat type

### Phase 8: IP Fencing System

**Files to Create:**

- `src/api/services/ip_fencing.py` - IP fencing engine
- `src/api/routes/ip_management.py` - IP management endpoints (already mentioned)
- `src/api/models/ip_blacklist.py` - IP blacklist models (already mentioned)
- `src/api/models/ip_reputation.py` - IP reputation models (already mentioned)

**Key Features:**

- **IP Blacklist/Whitelist**: Manual and automatic IP blocking
- **IP Reputation Scoring**: Calculate reputation based on:
                - Historical attack patterns
                - Threat intelligence feeds
                - Recent activity (anomaly scores, blocked requests)
                - Geographic location
- **Automatic Blocking**: Auto-block IPs with low reputation scores
- **IP History Tracking**: Track all requests from each IP
- **Temporary Blocks**: Time-based IP blocks (e.g., 1 hour, 24 hours)
- **IP Unblocking**: Manual and automatic unblocking after cooldown period
- **IP Range Blocking**: Support for CIDR notation
- **Integration**: Integrate with WAF service to check IPs before processing requests

### Phase 9: Geo-fencing System

**Files to Create:**

- `src/api/services/geo_fencing.py` - Geo-fencing engine
- `src/api/routes/geo_rules.py` - Geo-rules management (already mentioned)
- `src/api/models/geo_rules.py` - Geo-rules models (already mentioned)
- `src/api/services/geoip_lookup.py` - GeoIP lookup service

**Key Features:**

- **Country-based Rules**: Allow/deny traffic from specific countries
- **GeoIP Database**: Integrate MaxMind GeoIP2 or similar
- **Geographic Threat Analysis**: Identify high-risk countries
- **Flexible Rules**: 
                - Allow list (only allow specific countries)
                - Deny list (block specific countries)
                - Exception rules (allow specific IPs even if country is blocked)
- **Real-time Geo-location**: Lookup IP location in real-time
- **Geographic Analytics**: Track threats by country, visualize on map
- **Integration**: Check geo-rules before processing requests

### Phase 10: Bot Detection & Mitigation

**Files to Create:**

- `src/api/services/bot_detection.py` - Bot detection engine
- `src/api/routes/bot_detection.py` - Bot management endpoints (already mentioned)
- `src/api/models/bot_signatures.py` - Bot signatures models (already mentioned)
- `src/api/services/bot_behavior_analyzer.py` - Behavioral bot detection

**Key Features:**

- **User-Agent Analysis**: Detect known bot User-Agents
- **Behavioral Patterns**: Detect bots based on:
                - Request patterns (too fast, too regular)
                - Missing headers (Accept-Language, etc.)
                - JavaScript execution capability
                - Mouse/keyboard events (for browser-based detection)
- **Bot Signatures Database**: Maintain database of known bot signatures
- **CAPTCHA Integration**: Optional CAPTCHA for suspicious traffic
- **Automated Blocking**: Auto-block confirmed bots
- **Bot Categories**: Classify bots (search engines, scrapers, malicious bots)
- **Whitelist**: Allow legitimate bots (Googlebot, etc.)
- **Integration**: Check bot status before processing requests

### Phase 11: Threat Intelligence Integration

**Files to Create:**

- `src/api/services/threat_intel_service.py` - Threat intelligence service (already mentioned)
- `src/api/routes/threat_intel.py` - Threat intel management (already mentioned)
- `src/api/models/threat_intel.py` - Threat intel models (already mentioned)
- `src/api/services/threat_feed_parser.py` - Parse threat intelligence feeds

**Key Features:**

- **Threat Feed Integration**: 
                - IP reputation feeds (AbuseIPDB, VirusTotal, etc.)
                - Malware signature feeds
                - Known attack pattern feeds
                - CVE databases
- **Automatic Updates**: Scheduled updates from threat feeds
- **Threat Matching**: Match incoming requests against threat intelligence
- **Threat Scoring**: Calculate threat scores based on multiple sources
- **Historical Threat Data**: Store and query historical threat data
- **Custom Threat Sources**: Support for custom threat intelligence sources
- **Integration**: Use threat intel to enhance anomaly detection and IP reputation

### Phase 12: Advanced Rate Limiting & DDoS Protection

**Files to Create/Modify:**

- `src/api/services/advanced_rate_limiting.py` - Enhanced rate limiting
- `src/api/services/ddos_protection.py` - DDoS protection mechanisms
- `src/inference/rate_limiter.py` - Enhance existing rate limiter

**Key Features:**

- **Per-IP Rate Limiting**: Already implemented, enhance with adaptive limits
- **Per-Endpoint Rate Limiting**: Different limits for different endpoints
- **Adaptive Rate Limiting**: Adjust limits based on traffic patterns
- **DDoS Detection**: Detect DDoS attacks based on:
                - Request volume spikes
                - Multiple IPs targeting same endpoint
                - Unusual traffic patterns
- **Automatic Mitigation**: Auto-block or rate-limit during DDoS attacks
- **Rate Limit Headers**: Return rate limit info in response headers
- **Whitelist Bypass**: Allow whitelisted IPs to bypass rate limits
- **Distributed Rate Limiting**: Support for distributed systems (Redis-based)

### Phase 13: Response Inspection & Data Leakage Prevention (DLP)

**Files to Create:**

- `src/api/services/dlp_service.py` - DLP service (already mentioned)
- `src/api/services/response_inspector.py` - Response inspection engine
- `src/api/models/sensitive_data_patterns.py` - Sensitive data pattern models

**Key Features:**

- **Response Inspection**: Inspect outgoing responses for sensitive data
- **Sensitive Data Detection**: Detect:
                - Credit card numbers
                - Social security numbers
                - Email addresses
                - API keys
                - Passwords
                - Personal information (PII)
- **Pattern Matching**: Use regex and ML for pattern detection
- **Response Filtering**: Block or redact sensitive data in responses
- **Data Leakage Alerts**: Generate alerts when sensitive data is detected
- **Compliance**: Support for GDPR, PCI-DSS, HIPAA compliance
- **Custom Patterns**: Allow custom sensitive data patterns
- **Integration**: Integrate with web server to inspect responses

### Phase 14: Security Rules Engine

**Files to Create:**

- `src/api/services/rules_service.py` - Rules engine service (already mentioned)
- `src/api/routes/security_rules.py` - Rules management (already mentioned)
- `src/api/models/security_rules.py` - Rules models (already mentioned)
- `src/api/services/rule_executor.py` - Rule execution engine

**Key Features:**

- **Custom Rules**: Create custom security rules using:
                - Pattern matching (regex)
                - Header inspection
                - Body inspection
                - Query parameter inspection
                - IP-based rules
                - Geo-based rules
- **OWASP Top 10 Rules**: Pre-built rules for OWASP Top 10 vulnerabilities
- **Rule Priority**: Prioritize rules (high, medium, low)
- **Rule Actions**: Actions when rule matches (block, log, alert, redirect)
- **Rule Testing**: Test rules before deployment
- **Rule Management API**: CRUD operations for rules
- **Rule Performance**: Optimize rule execution for performance
- **Integration**: Execute rules before/after ML model inference

### Phase 15: Advanced Analytics & Reporting

**Files to Create:**

- `src/api/services/advanced_analytics.py` - Advanced analytics service
- `src/api/routes/analytics.py` - Enhanced analytics endpoints
- `src/api/services/report_generator.py` - Report generation service

**Key Features:**

- **Threat Trends**: Analyze threat trends over time
- **Attack Pattern Analysis**: Identify common attack patterns
- **Geographic Analytics**: Threat analysis by geography
- **IP Reputation Analytics**: Track IP reputation trends
- **Model Performance Analytics**: Track model accuracy, false positives/negatives
- **Compliance Reporting**: Generate compliance reports (GDPR, PCI-DSS, etc.)
- **Custom Dashboards**: Support for custom dashboard configurations
- **Export Capabilities**: Export reports as PDF, CSV, JSON
- **Scheduled Reports**: Schedule automated report generation
- **Real-time Insights**: Real-time threat insights and recommendations

### Phase 16: Configuration & Startup

**Files to Create/Modify:**

- `config/config.yaml` - Comprehensive configuration
- `src/api/startup.py` - Application startup/shutdown handlers
- `scripts/start_api_server.py` - API server startup script
- `scripts/setup_database.py` - Database setup and migration script
- Update `docker-compose.yml` - Add API server, database, Redis services

**Configuration Sections:**

```yaml
api_server:
  host: "0.0.0.0"
  port: 3001
  workers: 4
  database:
    url: "postgresql://user:pass@localhost/waf_db"
    pool_size: 20
    max_overflow: 10
  redis:
    url: "redis://localhost:6379/0"
  websocket:
    enabled: true
    ping_interval: 30
    max_connections: 1000
  data_retention:
    metrics_days: 30
    traffic_days: 7
    alerts_days: 90
    threats_days: 90
  security:
    jwt_secret: "${JWT_SECRET}"
    jwt_expiry: 3600
    rate_limit_per_minute: 60
    enable_cors: true
    allowed_origins: ["http://localhost:3000"]
  ip_fencing:
    auto_block_threshold: 0.7
    reputation_update_interval: 3600
  geo_fencing:
    geoip_db_path: "./data/GeoLite2-City.mmdb"
    update_interval: 2592000  # 30 days
  bot_detection:
    enable_behavioral_analysis: true
    captcha_threshold: 0.6
  threat_intel:
    feeds:
   - type: "ip_reputation"
        url: "https://api.abuseipdb.com/api/v2/blacklist"
        update_interval: 3600
    update_interval: 3600
  dlp:
    enable_response_inspection: true
    sensitive_patterns:
   - type: "credit_card"
        pattern: "\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b"
```

### Phase 17: Frontend Integration & Enhanced UI

**Tasks:**

- Verify all API endpoints match frontend expectations
- Test WebSocket connections and real-time updates
- Ensure data types match TypeScript interfaces
- Test error handling and loading states
- Verify time-range queries work correctly
- Test pagination for large datasets
- **Add new frontend features:**
                - IP management UI (blacklist/whitelist)
                - Geo-rules management UI
                - Bot detection dashboard
                - Threat intelligence dashboard
                - Security rules management UI
                - Advanced analytics visualizations
                - Settings panel for all configurations
                - User management UI
                - Audit log viewer

**Files to Create/Modify:**

- `frontend/app/ip-management/page.tsx` - IP management page
- `frontend/app/geo-rules/page.tsx` - Geo-rules management page
- `frontend/app/bot-detection/page.tsx` - Bot detection dashboard
- `frontend/app/threat-intel/page.tsx` - Threat intelligence dashboard
- `frontend/app/security-rules/page.tsx` - Security rules management
- `frontend/components/ip-management/` - IP management components
- `frontend/components/geo-map/` - Geographic threat visualization
- `frontend/lib/api.ts` - Add new API endpoints

### Phase 18: Remove Mock Data & Data Validation

**Files to Review:**

- `frontend/components/metrics-overview.tsx` - Remove hardcoded values
- `frontend/app/threats/page.tsx` - Remove hardcoded data
- All frontend components - Ensure no mock data remains

**Verification:**

- All components fetch from real API
- No mock data in components
- All data comes from backend database
- Data validation on both frontend and backend
- Proper error handling for missing data

### Phase 19: Performance Optimization

**Tasks:**

- Optimize database queries (add indexes, use query optimization)
- Implement Redis caching for frequently accessed data
- Optimize model inference (batching, quantization, GPU acceleration)
- Implement connection pooling
- Add database query result caching
- Optimize WebSocket message broadcasting
- Implement lazy loading for frontend components
- Add pagination for all list endpoints
- Optimize API response times (target: < 200ms for most endpoints)

**Files to Create/Modify:**

- `src/api/services/cache_service.py` - Caching service using Redis
- `src/api/middleware/cache_middleware.py` - Response caching middleware
- Database migration scripts for indexes

### Phase 20: Security Hardening & Testing

**Tasks:**

- Implement authentication and authorization
- Add API key management
- Implement audit logging for all security events
- Add security headers
- Implement input validation and sanitization
- Add SQL injection prevention
- Add XSS prevention
- Security testing (penetration testing)
- Load testing
- Stress testing

**Files to Create:**

- `src/api/auth.py` - Authentication (already mentioned)
- `src/api/models/users.py` - User models (already mentioned)
- `src/api/models/audit_log.py` - Audit log models (already mentioned)
- `src/api/middleware/audit_middleware.py` - Audit logging middleware
- `tests/security/` - Security test suite
- `tests/performance/` - Performance test suite

## Data Flow

1. **Request Flow**:

                        - Client Request → Rate Limiter → IP Fencing Check → Geo-fencing Check → Bot Detection → Security Rules → ML Model → Response Inspector → Response

2. **Log Processing Flow**:

                        - Log files → LogIngestionSystem → Queue → LogProcessor → Parse → Enrich (IP, Geo, Bot) → Model Check → Threat Classification → Store → Generate Alerts

3. **Metrics Collection**:

                        - Model results + System stats + Security events → MetricsService → Database → WebSocket broadcast

4. **Alert Generation**:

                        - Anomaly scores + Threat intel + Security rules → AlertService → Database + WebSocket broadcast + Notifications

5. **API Requests**:

                        - Frontend → REST API → Authentication → Services → Database/Cache → Response

6. **Real-time Updates**:

                        - Background services → WebSocket broadcast → Frontend

## Key Integration Points

### Existing Components to Leverage

- `src/integration/waf_service.py` - WAFService class for anomaly detection
- `src/ingestion/ingestion.py` - LogIngestionSystem for log reading
- `src/parsing/pipeline.py` - ParsingPipeline for log parsing
- `src/model/anomaly_detector.py` - AnomalyDetector for model inference
- `src/model/scoring.py` - AnomalyScorer for scoring requests
- `src/inference/rate_limiter.py` - Rate limiter (enhance)
- `src/inference/async_waf_service.py` - Async WAF service (enhance)

### New Components to Create

- Complete API server infrastructure
- Database models and persistence
- All 9 WAF protection methods
- Advanced analytics and reporting
- Security rules engine
- Threat intelligence integration
- WebSocket broadcasting
- Background workers
- Authentication and authorization
- Audit logging

## Testing Strategy

1. **Unit Tests**: Test each service independently
2. **Integration Tests**: Test API endpoints with test database
3. **End-to-End Tests**: Test full flow from request to response
4. **WebSocket Tests**: Test real-time update delivery
5. **Performance Tests**: Test with high traffic volume
6. **Security Tests**: Penetration testing, vulnerability scanning
7. **Load Tests**: Test system under high load
8. **Stress Tests**: Test system limits and failure scenarios

## Deployment Considerations

- Database migrations for schema changes
- Environment variables for configuration
- Docker Compose for local development
- Kubernetes deployment for production
- Health checks for monitoring
- Logging for debugging (structured logging)
- Rate limiting for API protection
- SSL/TLS for all connections
- Backup and disaster recovery
- Monitoring and alerting (Prometheus, Grafana)
- CI/CD pipeline

## Success Criteria

- ✅ All frontend API calls return real data
- ✅ WebSocket provides real-time updates
- ✅ Historical data available for charts
- ✅ Alerts generated from real detections
- ✅ No mock/hardcoded data in frontend
- ✅ Dashboard displays live WAF metrics
- ✅ All 9 WAF protection methods implemented
- ✅ Performance acceptable (< 200ms API response time)
- ✅ Security hardened (authentication, authorization, audit logging)
- ✅ Advanced analytics and reporting functional
- ✅ Threat intelligence integrated
- ✅ Bot detection and mitigation working
- ✅ IP and geo-fencing operational
- ✅ Response inspection/DLP functional
- ✅ Security rules engine operational

## Priority Implementation Order

1. **High Priority** (Core Functionality):

                        - Phases 1-6: Database, API infrastructure, basic services
                        - Phase 7: ML pipeline integration
                        - Phase 18: Remove mock data

2. **Medium Priority** (Security Features):

                        - Phases 8-12: IP fencing, geo-fencing, bot detection, threat intel, rate limiting
                        - Phase 20: Security hardening

3. **Lower Priority** (Advanced Features):

                        - Phases 13-15: DLP, rules engine, advanced analytics
                        - Phase 19: Performance optimization

This plan provides a comprehensive roadmap for building a production-ready, enterprise-grade WAF system with all advanced features and full frontend-backend integration.