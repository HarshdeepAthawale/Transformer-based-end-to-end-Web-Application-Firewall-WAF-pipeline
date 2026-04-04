# Week 4 Day 2: Redis-Backed Usage Counting

**Status:** PENDING
**Theme:** Replace per-request DB writes with Redis INCR + batch flush

## Goal

Zero DB writes per request. Usage tracked via Redis INCR (O(1)). Background job flushes to PostgreSQL every 30 seconds.

## Current Bottleneck

- `usage_limit_middleware.py:64` calls `increment_usage()` on EVERY request
- `increment_usage()` does SELECT + UPDATE/INSERT + COMMIT per request
- At 1000 req/sec = 1000 DB writes/sec (SQLite max ~100 writes/sec)

## Files to Modify

1. **`backend/services/usage_service.py`** - Rewrite to use Redis INCR
   - `increment_usage()` -> Redis INCR on `usage:{org_id}:{YYYY-MM}`
   - `flush_to_db()` -> batch upsert Redis counters to PostgreSQL
   - `is_within_limit()` -> read from Redis (fast)
   - `get_usage()` -> read from Redis with DB fallback

2. **`backend/middleware/usage_limit_middleware.py`** - Remove SessionLocal()
   - Call Redis-backed increment_usage() (no DB session needed)
   - Increase cache TTL from 10s to 30s

3. **`backend/tasks/usage_flush_task.py`** (NEW) - Background flush job
   - Every 30 seconds: read all Redis usage keys, batch upsert to DB
   - Register in scheduler.py

4. **`backend/middleware/rate_limit_middleware.py`** - Remove in-memory state
   - Remove _request_tracker and _config_cache dicts
   - Remove SessionLocal() from dispatch

5. **`backend/tasks/scheduler.py`** - Register UsageFlushTask

## Verification

```bash
redis-cli SET "usage:1:2026-04" 500
curl -H "Authorization: Bearer $TOKEN" http://localhost:3001/api/billing/usage
# Should show requests_count: 500
```
