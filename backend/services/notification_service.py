"""
Notification delivery service: email and webhook for WAF alerts.
Respects account settings (notifications, email_alerts, severity filters, webhook_url).
"""
from typing import Any, Dict, List
from loguru import logger
import urllib.request
import urllib.error
import json
import ssl

from backend.services.email_sender import send_alert_email


def _severity_enabled(settings: Dict[str, Any], severity: str) -> bool:
    """Check if this severity is enabled in settings."""
    key = f"alert_severity_{severity}"
    return bool(settings.get(key, False))


def _should_send_email(settings: Dict[str, Any], alert_dict: Dict[str, Any]) -> bool:
    if not settings.get("email_alerts", True):
        return False
    severity = (alert_dict.get("severity") or "high").lower()
    return _severity_enabled(settings, severity)


def _get_email_recipients(settings: Dict[str, Any], db) -> List[str]:
    """Get list of email addresses to notify. Uses alert_emails setting or fallback to admin users."""
    emails_raw = settings.get("alert_emails")
    if emails_raw:
        if isinstance(emails_raw, str):
            return [e.strip() for e in emails_raw.split(",") if e.strip()]
        if isinstance(emails_raw, list):
            return [str(e).strip() for e in emails_raw if e]
    # Fallback: admin users
    try:
        from backend.models.users import User, UserRole
        admins = db.query(User).filter(User.role == UserRole.ADMIN).all()
        return [u.email for u in admins if u.email]
    except Exception:
        return []


def _send_webhook(webhook_url: str, payload: Dict[str, Any], timeout: int = 5) -> bool:
    """POST payload to webhook URL. Returns True if 2xx, else False. One retry on failure."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    ctx = ssl.create_default_context()
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                if 200 <= resp.getcode() < 300:
                    logger.info(f"Webhook delivered to {webhook_url}")
                    return True
                logger.warning(f"Webhook returned {resp.getcode()} from {webhook_url}")
                return False
        except urllib.error.URLError as e:
            logger.warning(f"Webhook attempt {attempt + 1} failed for {webhook_url}: {e}")
        except Exception as e:
            logger.warning(f"Webhook error: {e}")
    return False


def maybe_send_notifications(db, alert, settings: Dict[str, Any]) -> None:
    """
    Send outbound notifications (email, webhook) for the given alert if settings allow.
    Call this from the same place that creates the alert (e.g. WAF middleware background thread).
    alert: Alert model instance with .to_dict()
    settings: result of get_settings(db) or equivalent dict.
    """
    if not settings.get("notifications", True):
        return

    alert_dict = alert.to_dict() if hasattr(alert, "to_dict") else alert
    severity = (alert_dict.get("severity") or "high").lower()
    if not _severity_enabled(settings, severity):
        return

    # Email
    if _should_send_email(settings, alert_dict):
        recipients = _get_email_recipients(settings, db)
        if recipients:
            from backend.config import config
            subject = f"[WAF Alert] {alert_dict.get('title', 'Security alert')}"
            body_text = (
                f"{alert_dict.get('title', 'Alert')}\n\n"
                f"{alert_dict.get('description', '')}\n\n"
                f"Severity: {severity}\n"
                f"Time: {alert_dict.get('timestamp', '')}\n"
            )
            if config.DASHBOARD_BASE_URL:
                body_text += f"\nDashboard: {config.DASHBOARD_BASE_URL}"
            body_html = (
                f"<p><strong>{alert_dict.get('title', 'Alert')}</strong></p>"
                f"<p>{alert_dict.get('description', '')}</p>"
                f"<p>Severity: {severity} | Time: {alert_dict.get('timestamp', '')}</p>"
            )
            if config.DASHBOARD_BASE_URL:
                body_html += f'<p><a href="{config.DASHBOARD_BASE_URL}">Open dashboard</a></p>'
            send_alert_email(recipients, subject, body_text, body_html)

    # Webhook
    webhook_url = (settings.get("webhook_url") or "").strip()
    if webhook_url:
        payload = {
            "event": "waf.alert",
            "alert": alert_dict,
            "timestamp": alert_dict.get("timestamp"),
        }
        _send_webhook(webhook_url, payload)
