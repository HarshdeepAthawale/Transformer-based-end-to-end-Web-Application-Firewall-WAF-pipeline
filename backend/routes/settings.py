"""Settings API: account preferences, retention, alerting (Feature 10)."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Any, Dict

from backend.database import get_db
from backend.auth import get_current_user, require_waf_api_auth
from backend.models.users import User
from backend.controllers import settings as ctrl
from backend.lib.datetime_utils import utc_now

router = APIRouter()


@router.get("")
async def get_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get all account settings (with defaults for missing keys). Auth required."""
    data = ctrl.get_settings(db)
    return {
        "success": True,
        "data": data,
        "timestamp": utc_now().isoformat(),
    }


@router.put("")
async def update_settings(
    payload: Dict[str, Any],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update one or more settings. Only allowed keys are persisted. Auth required."""
    data = ctrl.update_settings(db, payload)
    return {
        "success": True,
        "data": data,
        "message": "Settings updated",
        "timestamp": utc_now().isoformat(),
    }


@router.get("/retention")
async def get_retention():
    """Get data retention days (read-only from server config)."""
    data = ctrl.get_retention()
    return {
        "success": True,
        "data": data,
        "timestamp": utc_now().isoformat(),
    }


@router.get("/alerting")
async def get_alerting_settings(db: Session = Depends(get_db)):
    """Get alerting settings (webhook URL, rule thresholds). URL may be masked."""
    data = ctrl.get_alerting_settings(db)
    return {
        "success": True,
        "data": data,
        "timestamp": utc_now().isoformat(),
    }


@router.put("/alerting")
async def update_alerting_settings(
    payload: Dict[str, Any],
    db: Session = Depends(get_db),
    _auth=Depends(require_waf_api_auth),
):
    """Update alerting settings (webhook URL, thresholds). Auth required."""
    data = ctrl.update_alerting_settings(db, payload)
    return {
        "success": True,
        "data": data,
        "message": "Alerting settings updated",
        "timestamp": utc_now().isoformat(),
    }


@router.post("/alerting/test-webhook")
async def test_webhook(
    db: Session = Depends(get_db),
    _auth=Depends(require_waf_api_auth),
):
    """Send a test webhook payload to verify configuration."""
    from backend.services.notification_service import send_alert_webhook

    cfg = ctrl.get_alerting_settings_with_secret(db)
    url = (cfg.get("webhook_url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="No webhook_url configured")

    test_payload = {
        "event": "alert",
        "severity": "info",
        "title": "WAF Test Webhook",
        "description": "This is a test delivery from your WAF dashboard.",
        "timestamp": utc_now().isoformat(),
        "source": "test",
    }

    webhook_secret = (cfg.get("webhook_secret") or "").strip() or None
    ok = send_alert_webhook(url, test_payload, webhook_secret=webhook_secret)

    return {
        "success": True,
        "delivered": ok,
        "message": "Test webhook sent" if ok else "Test webhook delivery failed",
        "timestamp": utc_now().isoformat(),
    }
