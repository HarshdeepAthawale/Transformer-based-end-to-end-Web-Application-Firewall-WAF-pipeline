"""
Celery task definitions for WAF periodic background processing.

All 6 periodic tasks from scheduler.py are replicated here as Celery shared tasks.
LogProcessor is intentionally excluded — it tracks file position state across
invocations and is not suited to stateless task execution.

Tasks use lazy imports to avoid import-time side effects in worker processes.
"""
import time

from celery import shared_task
from loguru import logger

# Module-level start time — stable across repeated task invocations within a worker
# process. Used by aggregate_metrics to report worker uptime.
_WORKER_START_TIME = time.time()


@shared_task(name="backend.tasks.celery_tasks.evaluate_alerts")
def evaluate_alerts():
    """Run alert rule evaluator for all active organizations."""
    from backend.database import SessionLocal
    from backend.models.organization import Organization
    from backend.services.alert_evaluator import run_evaluator_once

    db = SessionLocal()
    try:
        orgs = db.query(Organization).filter(Organization.is_active).all()
        for org in orgs:
            run_evaluator_once(db, org.id)
        logger.debug("evaluate_alerts: processed %d orgs", len(orgs))
    except Exception as e:
        logger.exception("evaluate_alerts task error: %s", e)
        raise
    finally:
        db.close()


@shared_task(name="backend.tasks.celery_tasks.aggregate_metrics")
def aggregate_metrics():
    """Aggregate WAF metrics snapshot and broadcast to WebSocket clients."""
    from backend.tasks.metrics_aggregator import MetricsAggregator

    agg = MetricsAggregator()
    # Use stable worker start time so uptime reflects process lifetime, not task age
    agg._start_time = _WORKER_START_TIME
    try:
        agg._aggregate_metrics()
        logger.debug("aggregate_metrics: done")
    except Exception as e:
        logger.exception("aggregate_metrics task error: %s", e)
        raise


@shared_task(name="backend.tasks.celery_tasks.update_ip_reputation")
def update_ip_reputation():
    """Update IP reputation scores and auto-block low-reputation IPs."""
    from backend.tasks.ip_reputation_updater import IPReputationUpdater

    updater = IPReputationUpdater()
    try:
        updater._update_reputations()
        logger.debug("update_ip_reputation: done")
    except Exception as e:
        logger.exception("update_ip_reputation task error: %s", e)
        raise


@shared_task(name="backend.tasks.celery_tasks.sync_managed_rules")
def sync_managed_rules():
    """Sync managed security rules from remote feed URL.

    No-ops if MANAGED_RULES_FEED_URL is unset (_run_once handles the check).
    """
    from backend.tasks.managed_rules_sync_task import ManagedRulesSyncTask

    task = ManagedRulesSyncTask()
    try:
        task._run_once()
        logger.debug("sync_managed_rules: done")
    except Exception as e:
        logger.exception("sync_managed_rules task error: %s", e)
        raise


@shared_task(name="backend.tasks.celery_tasks.compute_ddos_threshold")
def compute_ddos_threshold():
    """Recompute adaptive DDoS threshold and publish to Redis.

    No-ops if ADAPTIVE_DDOS_ENABLED is false.
    """
    from backend.config import config
    from backend.services.adaptive_ddos_service import compute_and_publish_threshold

    if not getattr(config, "ADAPTIVE_DDOS_ENABLED", False):
        logger.debug("compute_ddos_threshold: ADAPTIVE_DDOS_ENABLED=false, skipping")
        return

    try:
        compute_and_publish_threshold()
        logger.debug("compute_ddos_threshold: done")
    except Exception as e:
        # Non-critical — DDoS threshold failure should not block other tasks
        logger.warning("compute_ddos_threshold task error: %s", e)


@shared_task(name="backend.tasks.celery_tasks.flush_usage_counters")
def flush_usage_counters():
    """Flush Redis usage counters to PostgreSQL."""
    from backend.database import SessionLocal
    from backend.services.usage_service import UsageService

    db = SessionLocal()
    try:
        UsageService.flush_redis_to_db(db)
        logger.debug("flush_usage_counters: done")
    except Exception as e:
        logger.exception("flush_usage_counters task error: %s", e)
        raise
    finally:
        db.close()
