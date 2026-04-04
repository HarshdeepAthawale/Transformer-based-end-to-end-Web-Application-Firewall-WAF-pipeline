# Week 4 Day 4: Celery Task Queue

**Status:** PENDING
**Theme:** Replace daemon threads with Celery + Redis broker

## Goal

All 6 background tasks migrated to Celery. Proper retries, monitoring, multi-worker support. Graceful degradation to threads if Celery unavailable.

## Current State

All background tasks run as daemon threads via scheduler.py. Single-threaded, no retry logic, no monitoring. Thread crash = task lost silently.

## Tasks to Migrate

| Thread Task | Celery Task | Schedule |
|-------------|-------------|----------|
| AlertEvaluatorTask | evaluate_alerts | Every 60s |
| MetricsAggregator | aggregate_metrics | Every 60s |
| IPReputationUpdater | update_ip_reputation | Every 1h |
| ManagedRulesSyncTask | sync_managed_rules | Every 24h |
| AdaptiveDDoSJob | compute_ddos_threshold | Every 15m |
| UsageFlushTask (Day 2) | flush_usage_counters | Every 30s |

## Files to Create

1. **`backend/celery_app.py`** - Celery app config with Redis broker
2. **`backend/tasks/celery_tasks.py`** - All periodic tasks as Celery tasks

## Files to Modify

3. **`backend/tasks/scheduler.py`** - Dual mode: Celery or threads
4. **`backend/main.py`** - Detect Celery availability at startup
5. **`requirements.txt`** - Add celery[redis]>=5.3.0

## Verification

```bash
celery -A backend.celery_app worker --loglevel=info &
celery -A backend.celery_app beat --loglevel=info &
celery -A backend.celery_app inspect active
```
