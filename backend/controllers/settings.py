"""Settings controller: account preferences and retention."""
import json
from typing import Any, Dict
from sqlalchemy.orm import Session

from backend.models.settings import AccountSetting
from backend.config import config

# Alerting-related keys (Feature 10); used by GET/PUT /settings/alerting
ALERTING_KEYS = {
    "webhook_url": str,
    "webhook_headers": str,  # JSON object string
    "alert_rule_block_rate_threshold": float,
    "alert_rule_block_rate_window_minutes": int,
    "alert_rule_ddos_count_threshold": int,
}

# Keys we allow in settings (and their types for validation)
ALLOWED_KEYS = {
    "theme": str,
    "default_time_range": str,
    "notifications": bool,
    "email_alerts": bool,
    "auto_block_threats": bool,
    "alert_severity_critical": bool,
    "alert_severity_high": bool,
    "alert_severity_medium": bool,
    "webhook_url": str,
    "webhook_headers": str,
    "alert_emails": str,
    "alert_rule_block_rate_threshold": float,
    "alert_rule_block_rate_window_minutes": int,
    "alert_rule_ddos_count_threshold": int,
}

DEFAULTS = {
    "theme": "system",
    "default_time_range": "24h",
    "notifications": True,
    "email_alerts": True,
    "auto_block_threats": True,
    "alert_severity_critical": True,
    "alert_severity_high": True,
    "alert_severity_medium": False,
    "webhook_url": "",
    "webhook_headers": "",
    "alert_emails": "",
    "alert_rule_block_rate_threshold": getattr(config, "ALERT_RULE_BLOCK_RATE_THRESHOLD", 0.1),
    "alert_rule_block_rate_window_minutes": getattr(config, "ALERT_RULE_BLOCK_RATE_WINDOW_MINUTES", 5),
    "alert_rule_ddos_count_threshold": getattr(config, "ALERT_RULE_DDOS_COUNT_THRESHOLD", 100),
}


def get_settings(db: Session) -> Dict[str, Any]:
    """Return all account settings as a flat dict (with defaults for missing keys)."""
    rows = db.query(AccountSetting).all()
    out = dict(DEFAULTS)
    for row in rows:
        if row.key not in ALLOWED_KEYS:
            continue
        try:
            val = json.loads(row.value) if row.value else None
        except (TypeError, json.JSONDecodeError):
            val = row.value
        if val is not None:
            out[row.key] = val
    return out


def update_settings(db: Session, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Update only allowed keys; return full settings after update."""
    for key, value in payload.items():
        if key not in ALLOWED_KEYS:
            continue
        row = db.query(AccountSetting).filter(AccountSetting.key == key).first()
        if value is None:
            if row:
                db.delete(row)
            continue
        serialized = json.dumps(value)
        if row:
            row.value = serialized
        else:
            db.add(AccountSetting(key=key, value=serialized))
    db.commit()
    return get_settings(db)


def get_alerting_settings(db: Session) -> Dict[str, Any]:
    """Return only alerting-related settings (for GET /api/settings/alerting). Fallback to config."""
    full = get_settings(db)
    out = {}
    for k in ALERTING_KEYS:
        if k in full:
            out[k] = full[k]
        else:
            out[k] = DEFAULTS.get(k)
    # Mask webhook URL for display if desired (e.g. show last 4 chars only)
    if out.get("webhook_url") and len(out["webhook_url"]) > 8:
        out["webhook_url_masked"] = out["webhook_url"][:4] + "…" + out["webhook_url"][-4:]
    return out


def update_alerting_settings(db: Session, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Update only alerting keys. Returns updated alerting settings."""
    filtered = {k: v for k, v in payload.items() if k in ALERTING_KEYS}
    if not filtered:
        return get_alerting_settings(db)
    update_settings(db, filtered)
    return get_alerting_settings(db)


def get_retention() -> Dict[str, int]:
    """Return retention days from server config (read-only)."""
    return {
        "metrics_days": getattr(config, "METRICS_RETENTION_DAYS", 30),
        "traffic_days": getattr(config, "TRAFFIC_RETENTION_DAYS", 7),
        "alerts_days": getattr(config, "ALERTS_RETENTION_DAYS", 90),
        "threats_days": getattr(config, "THREATS_RETENTION_DAYS", 90),
    }
