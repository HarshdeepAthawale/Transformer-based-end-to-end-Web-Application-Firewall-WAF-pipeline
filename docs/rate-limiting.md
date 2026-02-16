# Rate Limiting

The WAF Gateway includes Redis-backed rate limiting to throttle excessive requests per IP.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_ENABLED` | `true` | Enable rate limiting |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | `120` | Max requests per IP per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Sliding window size (seconds) |
| `RATE_LIMIT_BURST` | `20` | Burst allowance (reserved) |
| `RATE_LIMIT_FAIL_OPEN` | `true` | Allow requests if Redis is unavailable |

## Behavior

- Uses a sliding-window algorithm in Redis for accurate throttling.
- Per-IP: each client IP is tracked independently.
- When exceeded: returns `429 Too Many Requests` with `Retry-After` header.
- Requires Redis; if Redis is unavailable and fail-open is true, requests are allowed.

## Gateway Deployment

Ensure Redis is available and `REDIS_URL` is set:

```bash
REDIS_URL=redis://redis:6379
RATE_LIMIT_ENABLED=true
docker compose -f docker-compose.gateway.yml up -d
```

## Dashboard

Rate limit events are reported to the backend (when `BACKEND_EVENTS_ENABLED=true`) and shown in the dashboard under "Rate Limit Hits" and the "Rate Limit & DDoS Events" chart.
