"""
Unit tests for Celery task functions.

Tasks use lazy imports so mocks target source modules, not the task module.
"""
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# evaluate_alerts
# ---------------------------------------------------------------------------

def test_evaluate_alerts_calls_evaluator_for_each_org():
    mock_org1 = MagicMock()
    mock_org1.id = 1
    mock_org2 = MagicMock()
    mock_org2.id = 2

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = [mock_org1, mock_org2]

    with patch("backend.database.SessionLocal", return_value=mock_db), \
         patch("backend.services.alert_evaluator.run_evaluator_once") as mock_eval:
        from backend.tasks.celery_tasks import evaluate_alerts
        evaluate_alerts()

    assert mock_eval.call_count == 2
    mock_eval.assert_any_call(mock_db, 1)
    mock_eval.assert_any_call(mock_db, 2)
    mock_db.close.assert_called_once()


def test_evaluate_alerts_closes_db_on_exception():
    mock_db = MagicMock()
    mock_db.query.side_effect = RuntimeError("db error")

    with patch("backend.database.SessionLocal", return_value=mock_db):
        from backend.tasks.celery_tasks import evaluate_alerts
        try:
            evaluate_alerts()
        except RuntimeError:
            pass

    mock_db.close.assert_called_once()


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------

def test_aggregate_metrics_calls_aggregator():
    with patch("backend.tasks.metrics_aggregator.MetricsAggregator") as MockAgg:
        mock_instance = MagicMock()
        MockAgg.return_value = mock_instance

        from backend.tasks.celery_tasks import aggregate_metrics
        aggregate_metrics()

    mock_instance._aggregate_metrics.assert_called_once()


# ---------------------------------------------------------------------------
# update_ip_reputation
# ---------------------------------------------------------------------------

def test_update_ip_reputation_calls_updater():
    with patch("backend.tasks.ip_reputation_updater.IPReputationUpdater") as MockUpdater:
        mock_instance = MagicMock()
        MockUpdater.return_value = mock_instance

        from backend.tasks.celery_tasks import update_ip_reputation
        update_ip_reputation()

    mock_instance._update_reputations.assert_called_once()


# ---------------------------------------------------------------------------
# sync_managed_rules
# ---------------------------------------------------------------------------

def test_sync_managed_rules_calls_run_once():
    with patch("backend.tasks.managed_rules_sync_task.ManagedRulesSyncTask") as MockSync:
        mock_instance = MagicMock()
        MockSync.return_value = mock_instance

        from backend.tasks.celery_tasks import sync_managed_rules
        sync_managed_rules()

    mock_instance._run_once.assert_called_once()


# ---------------------------------------------------------------------------
# compute_ddos_threshold
# ---------------------------------------------------------------------------

def test_compute_ddos_threshold_skips_when_disabled():
    mock_config = MagicMock()
    mock_config.ADAPTIVE_DDOS_ENABLED = False

    with patch("backend.config.config", mock_config), \
         patch("backend.services.adaptive_ddos_service.compute_and_publish_threshold") as mock_fn:
        from backend.tasks.celery_tasks import compute_ddos_threshold
        compute_ddos_threshold()

    mock_fn.assert_not_called()


def test_compute_ddos_threshold_runs_when_enabled():
    mock_config = MagicMock()
    mock_config.ADAPTIVE_DDOS_ENABLED = True

    with patch("backend.config.config", mock_config), \
         patch("backend.services.adaptive_ddos_service.compute_and_publish_threshold") as mock_fn:
        from backend.tasks.celery_tasks import compute_ddos_threshold
        compute_ddos_threshold()

    mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# flush_usage_counters
# ---------------------------------------------------------------------------

def test_flush_usage_counters_calls_service():
    mock_db = MagicMock()

    with patch("backend.database.SessionLocal", return_value=mock_db), \
         patch("backend.services.usage_service.UsageService") as MockUsage:
        from backend.tasks.celery_tasks import flush_usage_counters
        flush_usage_counters()

    MockUsage.flush_redis_to_db.assert_called_once_with(mock_db)
    mock_db.close.assert_called_once()


def test_flush_usage_counters_closes_db_on_exception():
    mock_db = MagicMock()

    with patch("backend.database.SessionLocal", return_value=mock_db), \
         patch("backend.services.usage_service.UsageService") as MockUsage:
        MockUsage.flush_redis_to_db.side_effect = RuntimeError("redis down")
        from backend.tasks.celery_tasks import flush_usage_counters
        try:
            flush_usage_counters()
        except RuntimeError:
            pass

    mock_db.close.assert_called_once()


# ---------------------------------------------------------------------------
# scheduler dual-mode
# ---------------------------------------------------------------------------

def test_scheduler_uses_threads_when_celery_disabled(monkeypatch):
    monkeypatch.setenv("WAF_USE_CELERY", "false")

    with patch("backend.tasks.scheduler._start_thread_workers") as mock_threads, \
         patch("backend.tasks.scheduler._start_celery_mode") as mock_celery:
        # Reset module-level state
        import backend.tasks.scheduler as sched
        sched._celery_mode_active = False

        sched.start_background_workers()

    mock_threads.assert_called_once()
    mock_celery.assert_not_called()


def test_scheduler_uses_celery_when_enabled_and_available(monkeypatch):
    monkeypatch.setenv("WAF_USE_CELERY", "true")

    with patch("backend.tasks.scheduler._celery_available", return_value=True), \
         patch("backend.tasks.scheduler._start_celery_mode") as mock_celery, \
         patch("backend.tasks.scheduler._start_thread_workers") as mock_threads:
        import backend.tasks.scheduler as sched
        sched.start_background_workers()

    mock_celery.assert_called_once()
    mock_threads.assert_not_called()


def test_scheduler_falls_back_to_threads_when_celery_unavailable(monkeypatch):
    monkeypatch.setenv("WAF_USE_CELERY", "true")

    with patch("backend.tasks.scheduler._celery_available", return_value=False), \
         patch("backend.tasks.scheduler._start_thread_workers") as mock_threads, \
         patch("backend.tasks.scheduler._start_celery_mode") as mock_celery:
        import backend.tasks.scheduler as sched
        sched.start_background_workers()

    mock_threads.assert_called_once()
    mock_celery.assert_not_called()
