# Feature 9: WAF/Bot API (API-First Rules and Policies)

## Overview

This feature exposes an **API-first** control plane for WAF rules, bot policies, rate limits, and optional multi-tenant/zone configuration. All operations (CRUD) are performed via REST API with authentication ([backend/auth.py](backend/auth.py)); no hardcoded rules or policies in code. This enables automation, Terraform, and multi-zone deployment similar to WAF’s account-level configuration.

## Objectives

- Expose REST API for security rules (create, read, update, delete) with request/response schemas documented.
- Expose REST API for bot policies (score bands, verified bots, signatures) as per Feature 2.
- Expose REST API for rate-limit configuration (e.g. limits per path or per zone) if not already present.
- Protect all mutating endpoints with auth (JWT or API key from config); document auth method.
- Optional: multi-tenant/zone support via scope (e.g. zone_id or tenant_id in DB and API); document schema and endpoints.
- Frontend continues to use these APIs; no duplicate logic.

## Architecture

```mermaid
flowchart TB
    Client[API Client] --> Auth[Auth middleware]
    Auth --> RulesAPI[Rules CRUD]
    Auth --> BotAPI[Bot CRUD]
    Auth --> RateLimitAPI[Rate limit config]
    RulesAPI --> RulesService[RulesService]
    BotAPI --> BotService[BotDetectionService]
    RateLimitAPI --> RateLimitService[Rate limit service]
    RulesService --> DB[(security_rules)]
    BotService --> DB2[(bot_signatures)]
    RateLimitService --> Redis[(Redis)] or DB
    Frontend[Frontend] --> Client
```

## Configuration (no hardcoding)

**Backend** ([backend/config.py](backend/config.py)):

| Variable | Type | Description |
|----------|------|-------------|
| `API_AUTH_REQUIRED` | bool | Require JWT/API key for WAF API endpoints. | `true` |
| `JWT_SECRET` | str | Existing; for JWT validation. | (existing) |
| `API_KEY_HEADER` | str | Optional; header name for API key (e.g. X-API-Key). | |
| `API_KEYS` | str | Optional; comma-separated valid API keys for machine access. | |

**.env.example**: Document API auth and optional API keys.

## Backend

### 1. Security rules API

- **Routes**: Under `backend/routes/security_rules.py` or existing rules router.
  - `GET /api/security-rules` — List rules (query: active_only, pack_id, limit, offset). Response: `{ "data": [ rule_dict ], "total": number }`.
  - `POST /api/security-rules` — Create rule. Body: name, description, rule_type, pattern, match_conditions, applies_to, action, priority, owasp_category, is_active. Response: rule_dict.
  - `GET /api/security-rules/{id}` — Get one rule.
  - `PUT /api/security-rules/{id}` — Update rule. Body: same as create (partial). Response: rule_dict.
  - `DELETE /api/security-rules/{id}` — Delete (or soft-deactivate) rule. Response: 204.
- **Auth**: Apply `Depends(get_current_user)` or API-key check to POST/PUT/DELETE; optionally GET for public read or same auth. Use [backend/auth.py](backend/auth.py).
- **Schemas**: Pydantic models in `backend/schemas/security_rules.py` for request/response. No mock responses; all from DB via [backend/services/rules_service.py](backend/services/rules_service.py) and models.

### 2. Bot policies API

- **Routes**: Under `backend/routes/bot_detection.py` (extend existing).
  - `GET /api/bot/signatures` — Already may exist; ensure full CRUD.
  - `POST /api/bot/signatures` — Create signature.
  - `PUT /api/bot/signatures/{id}` — Update.
  - `DELETE /api/bot/signatures/{id}` — Delete.
  - `GET /api/bot/score-bands` — List score bands (Feature 2).
  - `PUT /api/bot/score-bands` — Update score bands. Body: array of { min_score, max_score, action }.
  - `GET /api/bot/verified` — List verified bots (Feature 2).
  - `POST /api/bot/verified` — Add verified bot.
  - `DELETE /api/bot/verified/{id}` — Remove.
- **Auth**: Same as above. All data from DB.

### 3. Rate limit configuration API

- **Routes**: New or under existing router.
  - `GET /api/rate-limits` — List rate limit configs (e.g. per path or zone). Response: list of config objects (path_prefix, requests_per_minute, window_seconds, etc.).
  - `POST /api/rate-limits` — Create config. Body: path_prefix, requests_per_minute, window_seconds, burst (optional).
  - `PUT /api/rate-limits/{id}` — Update.
  - `DELETE /api/rate-limits/{id}` — Delete.
- **Backend storage**: If rate limits are currently gateway-only (env), introduce DB table `rate_limit_config` (id, path_prefix, requests_per_minute, window_seconds, burst, zone_id optional, is_active) and gateway fetches from backend or backend pushes to Redis. Document in spec.
- **Auth**: Same.

### 4. Multi-tenant/zone (optional)

- **Schema**: Add `zone_id` (or tenant_id) to security_rules, bot_signatures, rate_limit_config. Default zone_id = "default" for single-tenant.
- **API**: All list/create/update endpoints accept optional query or body `zone_id`; filter by zone. Document in API spec.
- **DB**: Migration add column zone_id with default.

### 5. OpenAPI documentation

- Ensure all new/updated endpoints appear in `/docs` and `/redoc` with request/response bodies and auth requirement documented.

## Gateway

- **Config**: If rate limits are driven from API, gateway periodically fetches `GET /api/rate-limits` (with API key or JWT from config) and applies config to Redis or in-memory. Document gateway config vars: `RATE_LIMIT_CONFIG_URL`, `RATE_LIMIT_CONFIG_AUTH_HEADER`, `RATE_LIMIT_CONFIG_REFRESH_SECONDS`.

## Frontend

- No new pages required if existing UI already uses backend APIs. Ensure [frontend/lib/api.ts](frontend/lib/api.ts) uses the same endpoints and sends auth (Bearer token). Document that all mutating actions go through API.

## Data Flow

1. Client (frontend or external) sends authenticated request to backend (e.g. POST /api/security-rules).
2. Backend validates auth, parses body, updates DB via service, returns persisted entity.
3. Gateway or other consumers (e.g. managed rules sync) read via GET endpoints; gateway may poll rate-limits and refresh local config.

## External Integrations

None. API is consumed by frontend and optionally by gateway/automation.

## Database

- **security_rules**: Existing; optional zone_id column.
- **bot_signatures**, **bot_score_bands**, **verified_bots**: As in Feature 2; optional zone_id.
- **rate_limit_config** (new if not exists): id, path_prefix, requests_per_minute, window_seconds, burst, zone_id (nullable), is_active, created_at.

Migration: Add columns or table per above.

## Testing

- **Integration**: With valid JWT or API key, POST a rule, GET list, PUT update, DELETE. Without auth, assert 401. No mocks; use test DB and auth secrets in env.
- **E2E**: Frontend login and CRUD rules/bot policies via UI; verify API calls and persistence.
