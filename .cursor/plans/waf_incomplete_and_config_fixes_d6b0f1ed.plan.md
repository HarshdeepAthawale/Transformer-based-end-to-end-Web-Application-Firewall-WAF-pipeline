---
name: WAF Incomplete and Config Fixes
overview: A plan to address incomplete features, missing setup, and misconfigurations in the WAF project so that first-run works reliably and production deployment is secure and correctly configured.
todos: []
isProject: false
---

# WAF Project: Incomplete Items and Configuration Fixes

This plan addresses gaps and misconfigurations identified from a full codebase review (backend, gateway, frontend, config).

---

## 1. Backend Fixes

### 1.1 Ensure `data/` directory exists before DB init

**Problem:** [backend/database.py](backend/database.py) and [backend/config.py](backend/config.py) use `data/waf_dashboard.db` (relative to project root). SQLAlchemy does not create parent directories; first run can fail if `data/` is missing.

**Action:** In `init_db()` (or before `create_engine`), create the DB directory when using SQLite. For example: resolve the DB file path from `DATABASE_URL`, get its parent, and call `Path(...).mkdir(parents=True, exist_ok=True)` when the URL is `sqlite:///...`.

### 1.2 Seed first admin user (optional env-driven)

**Problem:** [.env.example](.env.example) documents "Demo credentials (optional - defaults: [admin@waf.example](mailto:admin@waf.example) / admin123)" and `ADMIN_EMAIL` / `ADMIN_PASSWORD`, but no code seeds an admin user. First-run has no way to log in without manual DB or API user creation.

**Action:** Add a one-time seed in [backend/database.py](backend/database.py) (e.g. after `create_all`): if the users table is empty, create one user with username/email from `os.getenv("ADMIN_EMAIL", "admin@waf.example")` and password from `os.getenv("ADMIN_PASSWORD", "admin123")`, role admin. Optionally guard with an env flag (e.g. `SEED_ADMIN=true`) so production can disable it.

### 1.3 Require auth for general settings endpoints

**Problem:** In [backend/routes/settings.py](backend/routes/settings.py), `GET ""` and `PUT ""` (general account settings) have no auth dependency; only `PUT /alerting` uses `require_waf_api_auth`. Unauthenticated callers can read/update theme, notifications, webhook_url, etc.

**Action:** Add `Depends(get_current_user)` (or `require_waf_api_auth` for API parity) to the handlers for `GET ""` and `PUT ""` so only authenticated users can read/update general settings.

### 1.4 JWT secret must be set in production

**Problem:** [backend/auth.py](backend/auth.py) falls back to `secrets.token_urlsafe(32)` when `JWT_SECRET` is unset. That value is not persisted, so every restart invalidates all existing tokens.

**Action:** (1) In [backend/config.py](backend/config.py) or at app startup, when not in a dev mode (e.g. `DEBUG` false or explicit `ENV=production`), require `JWT_SECRET` to be non-empty and fail startup with a clear message if missing. (2) In [.env.example](.env.example), state that `JWT_SECRET` is required in production and must be a long random string.

### 1.5 Align config.yaml with actual model path

**Problem:** [config/config.yaml](config/config.yaml) uses `models/waf-anomaly` for `waf_service.model_path` and `integration.waf_service.model_path`, while [backend/core/waf_factory.py](backend/core/waf_factory.py), [backend/main.py](backend/main.py), and scripts use `models/waf-distilbert`. The YAML is never used for the running model path.

**Action:** Either (a) change `config/config.yaml` to use `models/waf-distilbert` so it matches code, or (b) have backend/gateway read the model path from the YAML when present (and keep env override). Prefer (a) for minimal change unless you want a single source of truth in YAML.

---

## 2. Gateway Fixes

### 2.1 Default or document BACKEND_EVENTS_URL

**Problem:** [gateway/config.py](gateway/config.py) defaults `BACKEND_EVENTS_URL` to `""`. If `.env` is not used or the var is unset, the gateway never sends events; dashboard shows no rate-limit/DDoS/WAF metrics.

**Action:** Either (1) default `BACKEND_EVENTS_URL` to `http://localhost:3001/api/events/ingest` when running outside Docker (e.g. detect by env or keep empty and document), or (2) leave default empty and in [.env.example](.env.example) add a prominent note that gateway event reporting requires setting `BACKEND_EVENTS_URL` and `BACKEND_EVENTS_ENABLED=true`. Prefer (2) to avoid hardcoding localhost in code.

### 2.2 Set BOT_BACKEND_URL in Docker

**Problem:** [docker-compose.yml](docker-compose.yml) does not set `BOT_BACKEND_URL` for the gateway. [gateway/bot_score.py](gateway/bot_score.py) returns `None` when the URL is empty (fail-open), so bot scoring never runs in the default compose stack.

**Action:** In [docker-compose.yml](docker-compose.yml), add for the gateway service: `BOT_BACKEND_URL: http://backend:3001` so bot management works when the gateway and backend are both in compose.

### 2.3 Document gateway feature URLs

**Problem:** When `UPLOAD_SCAN_ENABLED`, `FIREWALL_AI_ENABLED`, `CREDENTIAL_LEAK_ENABLED`, or `MANAGED_RULES_ENABLED` are true, the gateway needs `UPLOAD_SCAN_BACKEND_URL`, `FIREWALL_AI_BACKEND_URL`, `CREDENTIAL_LEAK_BACKEND_URL`, `MANAGED_RULES_BACKEND_URL` respectively. All default to `""` in [gateway/config.py](gateway/config.py); missing URL means the feature is effectively off.

**Action:** In [.env.example](.env.example), add a short "Gateway → Backend URLs" section listing each feature and its required env var (e.g. "For bot scoring set BOT_BACKEND_URL", "For upload scanning set UPLOAD_SCAN_BACKEND_URL", etc.) with example values for local and Docker.

---

## 3. Rate limit config from backend (optional)

**Problem:** [.env.example](.env.example) says the gateway can "fetch rate limit config from backend API" via `RATE_LIMIT_CONFIG_URL` and `RATE_LIMIT_CONFIG_AUTH_HEADER`. [gateway/rate_limit.py](gateway/rate_limit.py) only uses env vars and does not call any backend.

**Action (choose one):** (A) Implement in gateway: periodic fetch from `RATE_LIMIT_CONFIG_URL` (e.g. GET from backend `/api/rate-limits` or a dedicated config endpoint), parse response, and update `RedisRateLimiter` max_requests/window_seconds (with auth header when set). (B) Remove or reword the comment in `.env.example` to state that rate limits are env-only for now. Prefer (B) unless you need dynamic rate limit config from the dashboard.

---

## 4. Frontend / Auth

### 4.1 Configurable Google admin role

**Problem:** [frontend/lib/auth.ts](frontend/lib/auth.ts) hardcodes `user.role = user.email === 'harshdeepathawale27@gmail.com' ? 'admin' : 'user'` for Google sign-in.

**Action:** Replace with config-driven behavior: e.g. read `ADMIN_EMAILS` (comma-separated) from env and set role to admin when `user.email` is in that list; otherwise set a default role (e.g. viewer). Document in `.env.example`.

---

## 5. Configuration and docs (no code change required)

**Required for core behavior (document in README or a single "Setup" doc):**

- **Redis:** Backend and gateway need `REDIS_URL`. Backend needs it for IP blacklist sync; without it, "Block IP" does not take effect at the gateway.
- **JWT:** Set `JWT_SECRET` in production (and require it per 1.4).
- **NextAuth:** Set `NEXTAUTH_URL` and `NEXTAUTH_SECRET` for frontend login.
- **Backend URL:** Frontend server-side uses `BACKEND_URL` (e.g. `http://backend:3001` in Docker).

**Gateway → backend (for dashboard and features):** Document that when using the gateway, set `BACKEND_EVENTS_URL` for event metrics; set `BOT_BACKEND_URL`, `UPLOAD_SCAN_BACKEND_URL`, `FIREWALL_AI_BACKEND_URL`, `CREDENTIAL_LEAK_BACKEND_URL`, `MANAGED_RULES_BACKEND_URL` when the corresponding feature is enabled.

**First-run:** Document (1) create or ensure `data/` exists if using default SQLite path, (2) set Redis and JWT, (3) optional seed admin via env (after 1.2), (4) for ML WAF: run finetune script or place model and set `WAF_MODEL_PATH`, (5) for geo: obtain GeoLite2 and set `GEOIP_DB_PATH` (see [docs/GEOIP_SETUP.md](docs/GEOIP_SETUP.md)).

---

## 6. Implementation order (suggested)

1. **Critical:** 1.1 (data dir), 1.3 (settings auth), 1.4 (JWT required in prod), 2.2 (Docker BOT_BACKEND_URL).
2. **High:** 1.2 (seed admin), 1.5 (config.yaml model path), 2.1 and 2.3 (events URL + docs).
3. **Medium:** 4.1 (Google admin config), section 5 (consolidate setup docs).
4. **Optional:** Section 3 (rate limit config: implement or clarify in .env.example).

No changes to application behavior are required for "monitor-only" or dev setups beyond ensuring `data/` exists and Redis/JWT/NextAuth are set; the rest improves security, first-run experience, and production correctness.