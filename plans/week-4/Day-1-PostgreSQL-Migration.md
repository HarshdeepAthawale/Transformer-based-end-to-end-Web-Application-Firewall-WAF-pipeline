# Week 4 Day 1: PostgreSQL Migration + Connection Pooling

**Status:** PENDING
**Theme:** Switch from SQLite to PostgreSQL. Fix all SQLite-specific code.

## Goal

Production-grade database: PostgreSQL with connection pooling (pool_size=20, max_overflow=10). Fix all 11 func.strftime() calls. Keep SQLite fallback for local dev.

## Files to Modify

1. **`backend/database.py`** - Add pool_size, max_overflow, pool_recycle for PostgreSQL
2. **`alembic.ini`** - Remove hardcoded SQLite URL
3. **`migrations/env.py`** - Resolve DATABASE_URL at runtime
4. **`backend/lib/db_utils.py`** (NEW) - Dialect-aware hour_bucket() helper
5. **`backend/controllers/charts.py`** - Replace func.strftime with hour_bucket
6. **`backend/services/analytics_service.py`** - Replace func.strftime (2 instances)
7. **`backend/services/charts_service.py`** - Replace func.strftime (3 instances)
8. **`backend/routes/events.py`** - Replace func.strftime
9. **`backend/routes/dashboard.py`** - Replace func.strftime

## Verification

```bash
export DATABASE_URL="postgresql://waf:waf@localhost:5432/waf_db"
alembic upgrade head
pytest tests/unit/ -v
bash scripts/pre-commit-check.sh
```
