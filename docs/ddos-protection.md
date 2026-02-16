# DDoS Protection

The WAF Gateway includes an L7 DDoS protection layer.

## Mitigations

| Mitigation | Description |
|------------|-------------|
| **Request size** | Rejects requests with `Content-Length` > `DDOS_MAX_BODY_BYTES` (default 10MB) before body is read |
| **Burst detection** | Tracks requests per IP; if exceeding threshold in a short window, temporarily blocks the IP |
| **Block window** | Blocked IPs receive 429 until block expires |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DDOS_ENABLED` | `true` | Enable DDoS protection |
| `DDOS_MAX_BODY_BYTES` | `10485760` | Max request body size (bytes) |
| `DDOS_BURST_THRESHOLD` | `50` | Requests per IP that trigger burst block |
| `DDOS_BURST_WINDOW_SECONDS` | `5` | Window for burst detection |
| `DDOS_BLOCK_DURATION_SECONDS` | `60` | How long to block an IP after burst |
| `DDOS_FAIL_OPEN` | `true` | Allow requests if Redis is unavailable |

## Behavior

- Request size check runs before body is read.
- Burst detection uses Redis sliding window; when an IP exceeds `DDOS_BURST_THRESHOLD` requests in `DDOS_BURST_WINDOW_SECONDS`, it is blocked for `DDOS_BLOCK_DURATION_SECONDS`.
- Blocked IPs receive 429 with `Retry-After` header.

## Gateway Deployment

Requires Redis. Use the provided Compose file to run the gateway with Redis:

```bash
docker compose -f docker-compose.gateway.yml up -d
```

Set `REDIS_URL` (default in compose is `redis://redis:6379`) and, for dashboard metrics, `BACKEND_EVENTS_URL` and `BACKEND_EVENTS_ENABLED=true` (default is true).

If Redis is unavailable, DDoS protection is disabled and requests are allowed (fail-open); the gateway logs a warning at startup.

## Dashboard

DDoS events (burst, blocked, size) are reported to the backend when `BACKEND_EVENTS_URL` is set and `BACKEND_EVENTS_ENABLED=true`, and are shown in "DDoS Blocks" and the "Rate Limit & DDoS Events" chart.

## Block IP (B2B Scalable)

From the DoS/DDoS Protection page you can block abusive IPs. Blocked IPs are:

1. Stored in the backend database (source of truth)
2. Synced to Redis for fast enforcement
3. Enforced at the gateway before rate limit/DDoS checks

**REDIS_URL is required.** The backend MUST have `REDIS_URL` set and Redis must be running for Block IP to work. Without Redis:

- Block IP from the dashboard will return 503 Service Unavailable
- Blocked IPs will not be enforced at the gateway
- Backend startup will fail if existing blacklist entries cannot be synced

Both backend and gateway must share the same Redis (`REDIS_URL`). The gateway checks:

- `blacklist:{tenant_id}:{ip}` for exact IP matches
- `blacklist:{tenant_id}:cidr` (Redis SET) for CIDR ranges; uses `ipaddress` module to test membership

Returns 403 immediately for blacklisted IPs. For multi-tenant B2B, set `BLACKLIST_TENANT_ID` per deployment. CIDR support: blacklist entries with `is_range=True` and `ip_range` (e.g. `192.168.1.0/24`) are synced and enforced at the gateway.
