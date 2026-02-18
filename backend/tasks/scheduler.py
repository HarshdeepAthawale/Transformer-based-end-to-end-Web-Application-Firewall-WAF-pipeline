"""
Background Task Scheduler
"""
from loguru import logger
from backend.config import config
from backend.tasks.log_processor import LogProcessor
from backend.tasks.metrics_aggregator import MetricsAggregator
from backend.tasks.ip_reputation_updater import IPReputationUpdater
from backend.tasks.managed_rules_sync_task import ManagedRulesSyncTask


# Global instances
log_processor = None
metrics_aggregator = None
ip_reputation_updater = None
managed_rules_sync_task = None


def start_background_workers():
    """Start all background workers"""
    global log_processor, metrics_aggregator, ip_reputation_updater, managed_rules_sync_task

    logger.info("Starting background workers...")
    
    # Start log processor
    try:
        log_processor = LogProcessor()
        log_processor.start()
        logger.info("Log processor started")
    except Exception as e:
        logger.error(f"Failed to start log processor: {e}")
    
    # Start metrics aggregator
    try:
        metrics_aggregator = MetricsAggregator(interval_seconds=60)
        metrics_aggregator.start()
        logger.info("Metrics aggregator started")
    except Exception as e:
        logger.error(f"Failed to start metrics aggregator: {e}")
    
    # Start IP reputation updater
    try:
        ip_reputation_updater = IPReputationUpdater(interval_seconds=3600)
        ip_reputation_updater.start()
        logger.info("IP reputation updater started")
    except Exception as e:
        logger.error(f"Failed to start IP reputation updater: {e}")

    # Managed rules sync (only if feed URL is set)
    try:
        if (config.MANAGED_RULES_FEED_URL or "").strip():
            managed_rules_sync_task = ManagedRulesSyncTask()
            managed_rules_sync_task.start()
            logger.info("Managed rules sync task started")
        else:
            logger.debug("MANAGED_RULES_FEED_URL not set, skipping managed rules sync task")
    except Exception as e:
        logger.error(f"Failed to start managed rules sync task: {e}")


def stop_background_workers():
    """Stop all background workers"""
    global log_processor, metrics_aggregator, ip_reputation_updater, managed_rules_sync_task

    logger.info("Stopping background workers...")

    if log_processor:
        log_processor.stop()

    if metrics_aggregator:
        metrics_aggregator.stop()

    if ip_reputation_updater:
        ip_reputation_updater.stop()

    if managed_rules_sync_task:
        managed_rules_sync_task.stop()

    logger.info("Background workers stopped")
