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

Use the provided Compose file to run the gateway with Redis (rate limiting and DDoS both require Redis):

```bash
docker compose -f docker-compose.gateway.yml up -d
```

Ensure `REDIS_URL` is set (default in the compose is `redis://redis:6379`). For dashboard metrics, set `BACKEND_EVENTS_URL` to your backend ingest endpoint and `BACKEND_EVENTS_ENABLED=true` (default is true).

If Redis is unavailable, rate limiting is disabled and requests are allowed (fail-open); the gateway logs a warning at startup.

To verify rate limit and DDoS behavior, run the stress test against the gateway: `STRESS_TEST_BASE_URL=http://localhost:8080 python scripts/stress_test_rate_limit.py`.

## Dashboard

Rate limit events are reported to the backend when `BACKEND_EVENTS_URL` is set and `BACKEND_EVENTS_ENABLED=true`, and are shown in the dashboard under "Rate Limit Hits" and the "Rate Limit & DDoS Events" chart.
