"""
Celery application configuration for WAF background task processing.

Usage:
  # Start worker (processes tasks from queue)
  celery -A backend.celery_app worker -Q waf -l info --concurrency 4

  # Start beat scheduler (enqueues periodic tasks)
  celery -A backend.celery_app beat -l info

  # Inspect active tasks
  celery -A backend.celery_app inspect active

Enable in FastAPI by setting WAF_USE_CELERY=true in .env.
Thread-based fallback is used automatically when WAF_USE_CELERY is unset or false.
"""
import os

from celery import Celery

# SECURITY: broker URL contains Redis credentials if AUTH is set — read from env only
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", REDIS_URL)

celery_app = Celery(
    "waf",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["backend.tasks.celery_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Keep results for 1 hour then expire — tasks are fire-and-forget
    result_expires=3600,
    # Route all WAF tasks to the dedicated queue
    task_routes={
        "backend.tasks.celery_tasks.*": {"queue": "waf"},
    },
    # Beat schedule — mirrors the intervals in scheduler.py
    beat_schedule={
        "evaluate-alerts": {
            "task": "backend.tasks.celery_tasks.evaluate_alerts",
            "schedule": 60.0,
        },
        "aggregate-metrics": {
            "task": "backend.tasks.celery_tasks.aggregate_metrics",
            "schedule": 60.0,
        },
        "update-ip-reputation": {
            "task": "backend.tasks.celery_tasks.update_ip_reputation",
            "schedule": 3600.0,  # 1 hour
        },
        "sync-managed-rules": {
            "task": "backend.tasks.celery_tasks.sync_managed_rules",
            "schedule": 86400.0,  # 24 hours
        },
        "compute-ddos-threshold": {
            "task": "backend.tasks.celery_tasks.compute_ddos_threshold",
            "schedule": 900.0,  # 15 minutes
        },
        "flush-usage-counters": {
            "task": "backend.tasks.celery_tasks.flush_usage_counters",
            "schedule": 30.0,
        },
    },
)
