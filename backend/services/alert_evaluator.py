"""
Alert rule evaluator (Feature 10). Runs periodically; queries security_events,
evaluates block rate and DDoS count rules; creates alerts and sends webhook.
No hardcoded thresholds; from settings or config.
"""
import json
from datetime import timedelta
from backend.lib.datetime_utils import utc_now
from typing import Any, Dict

from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.models.security_event import SecurityEvent
from backend.models.alerts import Alert, AlertType, AlertSeverity
from backend.services.alert_service import AlertService
from backend.services.notification_service import send_alert_webhook
from backend.controllers.settings import get_alerting_settings
from backend.config import config
from loguru import logger


# Block-type event_types (same as dashboard)
BLOCK_EVENT_TYPES = [
    "waf_block", "waf_challenge", "waf",
    "rate_limit",
    "ddos_burst", "ddos_blocked", "ddos_size",
    "bot_block", "bot_challenge",
    "upload_scan_infected",
    "credential_leak_block",
    "firewall_ai_prompt_block", "firewall_ai_pii", "firewall_ai_abuse_rate",
    "blacklist",
]

DDOS_EVENT_TYPES = ["ddos_burst", "ddos_blocked", "ddos_size"]


def _get_alerting_config(db: Session) -> Dict[str, Any]:
    """Alerting config: DB settings override env."""
    cfg = get_alerting_settings(db)
    return {
        "webhook_url": (cfg.get("webhook_url") or "").strip()
                       or (getattr(config, "ALERT_WEBHOOK_URL", "") or "").strip(),
        "webhook_headers": _parse_webhook_headers(
            cfg.get("webhook_headers") or getattr(config, "ALERT_WEBHOOK_HEADERS", "") or ""
        ),
        "block_rate_threshold": float(cfg.get("alert_rule_block_rate_threshold")
                                      or getattr(config, "ALERT_RULE_BLOCK_RATE_THRESHOLD", 0.1)),
        "block_rate_window_minutes": int(cfg.get("alert_rule_block_rate_window_minutes")
                                         or getattr(config, "ALERT_RULE_BLOCK_RATE_WINDOW_MINUTES", 5)),
        "ddos_count_threshold": int(cfg.get("alert_rule_ddos_count_threshold")
                                    or getattr(config, "ALERT_RULE_DDOS_COUNT_THRESHOLD", 100)),
    }


def _parse_webhook_headers(raw: str) -> Dict[str, str]:
    """Parse ALERT_WEBHOOK_HEADERS: JSON object or key:value,key:value."""
    if not (raw or "").strip():
        return {}
    raw = raw.strip()
    try:
        if raw.startswith("{"):
            return json.loads(raw)
        out = {}
        for part in raw.split(","):
            part = part.strip()
            if ":" in part:
                k, v = part.split(":", 1)
                out[k.strip()] = v.strip()
        return out
    except Exception:
        return {}


def evaluate_rules(db: Session) -> None:
    """
    Evaluate alert rules: block rate and DDoS count in window.
    On trigger: create alert, send webhook (and email via maybe_send_notifications if desired).
    """
    cfg = _get_alerting_config(db)
    window_minutes = cfg["block_rate_window_minutes"]
    start_time = utc_now() - timedelta(minutes=window_minutes)

    # Total events in window
    total = db.query(func.count(SecurityEvent.id)).filter(
        SecurityEvent.timestamp >= start_time
    ).scalar() or 0

    # Block count
    block_count = db.query(func.count(SecurityEvent.id)).filter(
        SecurityEvent.event_type.in_(BLOCK_EVENT_TYPES),
        SecurityEvent.timestamp >= start_time,
    ).scalar() or 0

    # DDoS count
    ddos_count = db.query(func.count(SecurityEvent.id)).filter(
        SecurityEvent.event_type.in_(DDOS_EVENT_TYPES),
        SecurityEvent.timestamp >= start_time,
    ).scalar() or 0

    block_rate = (block_count / total) if total > 0 else 0.0
    threshold = cfg["block_rate_threshold"]
    ddos_threshold = cfg["ddos_count_threshold"]

    alert_service = AlertService(db)

    # Block rate rule
    if total > 0 and block_rate >= threshold:
        title = "High block rate detected"
        description = (
            f"Block rate in last {window_minutes} min: {block_rate:.1%} "
            f"({block_count} blocks / {total} requests). Threshold: {threshold:.1%}."
        )
        alert = alert_service.create_alert(
            type=AlertType.WARNING,
            severity=AlertSeverity.HIGH,
            title=title,
            description=description,
            source="alert_rule",
        )
        _notify_webhook(cfg, alert)

    # DDoS count rule
    if ddos_count >= ddos_threshold:
        title = "DDoS event spike"
        description = (
            f"DDoS events in last {window_minutes} min: {ddos_count}. Threshold: {ddos_threshold}."
        )
        alert = alert_service.create_alert(
            type=AlertType.CRITICAL,
            severity=AlertSeverity.HIGH,
            title=title,
            description=description,
            source="alert_rule",
        )
        _notify_webhook(cfg, alert)


def _notify_webhook(cfg: Dict[str, Any], alert: Alert) -> None:
    url = cfg.get("webhook_url")
    if not url:
        return
    headers = cfg.get("webhook_headers") or {}
    send_alert_webhook(url, alert.to_dict(), extra_headers=headers)


def run_evaluator_once(db: Session) -> None:
    """Run evaluator once (for use from background task or cron)."""
    try:
        evaluate_rules(db)
    except Exception as e:
        logger.exception(f"Alert evaluator error: {e}")
