"""
Background Task Scheduler

Mode selection (WAF_USE_CELERY env var):
  false (default) — all 7 tasks run as daemon threads within the FastAPI process.
  true            — Celery beat handles the 6 stateless periodic tasks; LogProcessor
                    still runs as a thread (it tracks file-position state across calls).
                    Requires a running Celery worker and beat process:
                      celery -A backend.celery_app worker -Q waf -l info
                      celery -A backend.celery_app beat -l info
                    Falls back to threads if the celery package is not installed.
"""
import os

from loguru import logger

from backend.config import config
from backend.tasks.log_processor import LogProcessor
from backend.tasks.metrics_aggregator import MetricsAggregator
from backend.tasks.ip_reputation_updater import IPReputationUpdater
from backend.tasks.managed_rules_sync_task import ManagedRulesSyncTask
from backend.tasks.adaptive_ddos_job import AdaptiveDDoSJob
from backend.tasks.alert_evaluator_task import AlertEvaluatorTask
from backend.tasks.usage_flush_task import UsageFlushTask


# Global instances (thread mode only)
log_processor = None
metrics_aggregator = None
ip_reputation_updater = None
managed_rules_sync_task = None
adaptive_ddos_job = None
alert_evaluator_task = None
usage_flush_task = None

# Tracks which mode is active so stop_background_workers behaves correctly
_celery_mode_active = False


def _celery_available() -> bool:
    """Return True if the celery package is importable."""
    try:
        import celery  # noqa: F401
        return True
    except ImportError:
        return False


def start_background_workers():
    """Start background workers.

    Uses Celery mode when WAF_USE_CELERY=true and celery is installed;
    falls back to thread-based workers otherwise.
    """
    global _celery_mode_active

    use_celery = os.environ.get("WAF_USE_CELERY", "false").lower() == "true"

    if use_celery:
        if _celery_available():
            _start_celery_mode()
            _celery_mode_active = True
            return
        logger.warning(
            "WAF_USE_CELERY=true but celery package is not installed. "
            "Install it with: pip install 'celery[redis]>=5.3.0'. "
            "Falling back to thread-based workers."
        )

    _celery_mode_active = False
    _start_thread_workers()


def _start_celery_mode():
    """Start only LogProcessor as a thread; log instructions for Celery worker."""
    global log_processor

    logger.info(
        "Celery mode active (WAF_USE_CELERY=true). "
        "6 periodic tasks are handled by Celery beat. "
        "Ensure workers are running: "
        "celery -A backend.celery_app worker -Q waf -l info && "
        "celery -A backend.celery_app beat -l info"
    )

    # LogProcessor maintains file-position state across invocations — keep as thread
    try:
        log_processor = LogProcessor()
        log_processor.start()
        logger.info("Log processor started (thread)")
    except Exception as e:
        logger.error(f"Failed to start log processor: {e}")


def _start_thread_workers():
    """Start all 7 background tasks as daemon threads."""
    global log_processor, metrics_aggregator, ip_reputation_updater
    global managed_rules_sync_task, adaptive_ddos_job, alert_evaluator_task
    global usage_flush_task

    logger.info("Starting background workers (thread mode)...")

    try:
        log_processor = LogProcessor()
        log_processor.start()
        logger.info("Log processor started")
    except Exception as e:
        logger.error(f"Failed to start log processor: {e}")

    try:
        metrics_aggregator = MetricsAggregator(interval_seconds=60)
        metrics_aggregator.start()
        logger.info("Metrics aggregator started")
    except Exception as e:
        logger.error(f"Failed to start metrics aggregator: {e}")

    try:
        ip_reputation_updater = IPReputationUpdater(interval_seconds=3600)
        ip_reputation_updater.start()
        logger.info("IP reputation updater started")
    except Exception as e:
        logger.error(f"Failed to start IP reputation updater: {e}")

    try:
        if (config.MANAGED_RULES_FEED_URL or "").strip():
            managed_rules_sync_task = ManagedRulesSyncTask()
            managed_rules_sync_task.start()
            logger.info("Managed rules sync task started")
        else:
            logger.debug("MANAGED_RULES_FEED_URL not set, skipping managed rules sync task")
    except Exception as e:
        logger.error(f"Failed to start managed rules sync task: {e}")

    try:
        adaptive_ddos_job = AdaptiveDDoSJob()
        adaptive_ddos_job.start()
    except Exception as e:
        logger.error(f"Failed to start adaptive DDoS job: {e}")

    try:
        alert_evaluator_task = AlertEvaluatorTask()
        alert_evaluator_task.start()
        logger.info("Alert evaluator task started")
    except Exception as e:
        logger.error(f"Failed to start alert evaluator task: {e}")

    try:
        usage_flush_task = UsageFlushTask()
        usage_flush_task.start()
    except Exception as e:
        logger.error(f"Failed to start usage flush task: {e}")


def stop_background_workers():
    """Stop all running background workers."""
    global log_processor, metrics_aggregator, ip_reputation_updater
    global managed_rules_sync_task, adaptive_ddos_job, alert_evaluator_task
    global usage_flush_task

    logger.info("Stopping background workers...")

    if log_processor:
        log_processor.stop()

    # In Celery mode only LogProcessor was started as a thread
    if not _celery_mode_active:
        if metrics_aggregator:
            metrics_aggregator.stop()
        if ip_reputation_updater:
            ip_reputation_updater.stop()
        if managed_rules_sync_task:
            managed_rules_sync_task.stop()
        if adaptive_ddos_job:
            adaptive_ddos_job.stop()
        if alert_evaluator_task:
            alert_evaluator_task.stop()
        if usage_flush_task:
            usage_flush_task.stop()

    logger.info("Background workers stopped")
