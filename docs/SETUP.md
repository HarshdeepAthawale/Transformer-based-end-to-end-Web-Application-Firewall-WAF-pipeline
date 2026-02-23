# WAF Setup and Configuration

This document describes required and optional configuration for running the WAF stack, and first-run steps.

## Required for core behavior

### Redis
- **Backend and gateway** both need `REDIS_URL` (e.g. `redis://localhost:6379`).
- The backend uses Redis to sync the IP blacklist; **without Redis, "Block IP" from the dashboard does not take effect at the gateway.**
- The gateway uses Redis for rate limiting and DDoS state.

### JWT
- Set `JWT_SECRET` to a long random string (e.g. `openssl rand -base64 32`).
- **In production:** when `ENV=production`, the backend requires `JWT_SECRET` to be set and will not start if it is missing (tokens would otherwise reset on every restart).

### NextAuth (frontend login)
- Set `NEXTAUTH_URL` (e.g. `http://localhost:3000`) and `NEXTAUTH_SECRET` (at least 32 characters) so the dashboard login and session work.

### Backend URL (frontend)
- The frontend proxies API requests using `BACKEND_URL` (server-side). For Docker, set `BACKEND_URL=http://backend:3001`.

---

## Local development (frontend, backend, WAF service)

To avoid stub routes and missing features, use the project virtual environment so **PyJWT** and **python-multipart** are installed (required for auth, settings, security-rules, audit, and upload scan).

**One-time setup:** From the project root, create a venv and install dependencies:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Start all three services (one command):**

```bash
./scripts/start_local.sh
```

This starts the backend (port 3001), WAF AI/ML service (port 8000, real model if `models/waf-distilbert` exists), and frontend (port 3000). The script creates `.venv` and installs PyJWT and python-multipart if needed.

**Or start manually in three terminals:**

```bash
# Terminal 1 – backend
.venv/bin/python -m uvicorn backend.main:app --host 0.0.0.0 --port 3001 --reload

# Terminal 2 – WAF service (use --workers 1; omit --placeholder to use real model)
.venv/bin/python scripts/start_waf_service.py --host 0.0.0.0 --port 8000 --workers 1

# Terminal 3 – frontend
cd frontend && npm run dev
```

---

## Gateway to backend (for dashboard and features)

When the gateway and backend run together, set these so the dashboard shows metrics and optional features work:

- **Event metrics (rate limit, DDoS, WAF blocks):** Set `BACKEND_EVENTS_URL` (e.g. `http://localhost:3001/api/events/ingest`) and `BACKEND_EVENTS_ENABLED=true`. If `BACKEND_EVENTS_URL` is empty, the gateway does not send events and the dashboard will not show these metrics.
- **Bot scoring:** Set `BOT_BACKEND_URL` (e.g. `http://localhost:3001`; in Docker `http://backend:3001`).
- **Upload scanning:** When `UPLOAD_SCAN_ENABLED=true`, set `UPLOAD_SCAN_BACKEND_URL`.
- **Firewall for AI:** When `FIREWALL_AI_ENABLED=true`, set `FIREWALL_AI_BACKEND_URL`.
- **Credential leak (HIBP):** When `CREDENTIAL_LEAK_ENABLED=true`, set `CREDENTIAL_LEAK_BACKEND_URL`.
- **Managed rules:** When `MANAGED_RULES_ENABLED=true`, set `MANAGED_RULES_BACKEND_URL`.

See [.env.example](../.env.example) for the "Gateway → Backend URLs" section and example values.

---

## First-run checklist

1. **Copy env:** `cp .env.example .env` and set at least Redis, JWT, and NextAuth variables as above.
2. **Data directory:** When using the default SQLite URL, the backend creates the `data/` directory automatically. For a custom SQLite path, ensure the parent directory exists.
3. **Redis:** Start Redis (or use Docker Compose). Backend startup will fail visibly if blacklist sync requires Redis and it is unavailable.
4. **Optional seed admin:** If the users table is empty, the backend seeds one admin user from `ADMIN_EMAIL` (default `admin@waf.example`) and `ADMIN_PASSWORD` (default `admin123`). Set `SEED_ADMIN=false` to disable (e.g. in production).
5. **ML WAF (optional):** To enable the transformer-based WAF (not placeholder mode), run the fine-tune script or place a pre-trained model, then set `WAF_MODEL_PATH` if needed (default is `models/waf-distilbert`). See [scripts/README.md](../scripts/README.md) and `scripts/finetune_waf_model.py`.
6. **Geo rules (optional):** For geo-fencing and geo rules, obtain the GeoLite2-City database and set `GEOIP_DB_PATH`. See [GEOIP_SETUP.md](GEOIP_SETUP.md).

---

## References

- [.env.example](../.env.example) – all environment variables with comments
- [Rate limiting](rate-limiting.md) – Redis and gateway rate limit config
- [DDoS protection](ddos-protection.md) – DDoS and gateway config
- [GEOIP_SETUP.md](GEOIP_SETUP.md) – GeoLite2 for geo rules
