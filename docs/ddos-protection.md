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

Requires Redis. Use with rate limiting for layered protection:

```bash
REDIS_URL=redis://redis:6379
DDOS_ENABLED=true
DDOS_BURST_THRESHOLD=50
docker compose -f docker-compose.gateway.yml up -d
```

## Dashboard

DDoS events (burst, blocked, size) are reported to the backend and shown in "DDoS Blocks" and the "Rate Limit & DDoS Events" chart.
