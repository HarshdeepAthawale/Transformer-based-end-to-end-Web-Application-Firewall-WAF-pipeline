"""Settings controller: account preferences and retention."""
import json
from typing import Any, Dict
from sqlalchemy.orm import Session

from backend.models.settings import AccountSetting
from backend.config import config

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
    "alert_emails": str,
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
    "alert_emails": "",
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


def get_retention() -> Dict[str, int]:
    """Return retention days from server config (read-only)."""
    return {
        "metrics_days": getattr(config, "METRICS_RETENTION_DAYS", 30),
        "traffic_days": getattr(config, "TRAFFIC_RETENTION_DAYS", 7),
        "alerts_days": getattr(config, "ALERTS_RETENTION_DAYS", 90),
        "threats_days": getattr(config, "THREATS_RETENTION_DAYS", 90),
    }
