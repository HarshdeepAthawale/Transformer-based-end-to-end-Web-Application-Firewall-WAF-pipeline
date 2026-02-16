# Production Deployment Guide

## Overview

For production, run the WAF stack behind a reverse proxy with TLS termination. The backend and gateway do not handle SSL directly; use Nginx or Traefik.

## Architecture

```
                    [Internet]
                         |
                    [Nginx/Traefik]
                    TLS Termination
                         |
         +---------------+---------------+
         |               |               |
    [Frontend]     [Gateway:8080]   [Backend:3001]
         |               |               |
         |         +-----+-----+          |
         |         |           |          |
         |      [Redis]   [MongoDB]   [PostgreSQL/SQLite]
         |         |           |          |
         +---------+-----------+----------+
```

## Deployment Steps

### 1. Full Stack with Docker Compose

```bash
cp .env.example .env
# Edit .env: set DATABASE_URL, REDIS_URL, NEXTAUTH_SECRET, etc.
docker compose up -d
```

- Dashboard: http://localhost:3000
- Backend API: http://localhost:3001
- Gateway: http://localhost:8080

### 2. Reverse Proxy (Nginx) with SSL

Example Nginx config for TLS:

```nginx
server {
    listen 443 ssl http2;
    server_name waf.example.com;

    ssl_certificate /etc/letsencrypt/live/waf.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/waf.example.com/privkey.pem;

    location / {
        proxy_pass http://frontend:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/ {
        proxy_pass http://backend:3001/api/;
        proxy_set_header Host $host;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /ws/ {
        proxy_pass http://backend:3001/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### 3. Environment Variables for Production

Run `python scripts/validate_production_env.py` before deploying to verify required vars.

#### Required

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL (not SQLite); e.g. `postgresql://user:pass@host:5432/waf_db` |
| `REDIS_URL` | Required for rate limit, DDoS, blacklist |
| `NEXTAUTH_SECRET` | Random string (32+ chars) for auth |
| `CORS_ORIGINS` | Comma-separated allowed origins (production domains only) |
| `MONGODB_URI` | Gateway event store |
| `UPSTREAM_URL` | Protected application URL (gateway) |
| `BACKEND_EVENTS_URL` | Gateway → backend events ingest URL |

#### Browser-Reachable (Frontend)

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_API_URL` | Backend API URL (must be reachable from browser) |
| `NEXT_PUBLIC_WS_URL` | WebSocket URL (e.g. `wss://api.example.com/ws/`) |

#### WAF

| Variable | Description |
|----------|-------------|
| `WAF_THRESHOLD` | Anomaly threshold (0.5–0.7) |
| `WAF_REQUIRE_MODEL` | Set to `true` to fail startup if model missing (production) |
| `WAF_FAIL_OPEN` | `true` = allow on WAF error; `false` = block |

### 4. Model Verification

Ensure `models/waf-distilbert` exists with `config.json` and `tokenizer.json`. Docker images include `COPY models/`. Health endpoints expose `model_loaded`:

- Backend: `GET /health` → `{"model_loaded": true|false, ...}`
- Gateway: `GET /gateway/health` → `{"model_loaded": true|false, ...}`

Set `WAF_REQUIRE_MODEL=true` in production to fail startup if the model is missing.

### 5. Monitoring

- Health: `GET /health` (backend), `GET /gateway/health` (gateway)
- WAF metrics: `GET /api/waf/middleware-metrics`
- Dashboard metrics: `GET /api/metrics/realtime`

### 6. Continuous Learning (Optional)

Set in `.env`:

```
LEARNING_ENABLED=true
LOG_PATH=/var/log/nginx/access.log
LEARNING_UPDATE_INTERVAL_HOURS=24
```

The scheduler will collect new benign traffic and fine-tune the model periodically.

## SSL/TLS

- Do **not** expose backend or gateway directly to the internet.
- Use a reverse proxy (Nginx, Traefik, Caddy) for TLS.
- Obtain certs via Let's Encrypt (certbot) or your CA.
