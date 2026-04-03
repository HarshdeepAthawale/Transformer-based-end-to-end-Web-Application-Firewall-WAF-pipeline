"""Alerts controller."""
from backend.lib.datetime_utils import utc_now
from sqlalchemy.orm import Session

from backend.services.alert_service import AlertService
from backend.core.time_range import parse_time_range


def get_active(db: Session, org_id: int) -> dict:
    service = AlertService(db)
    alerts = service.get_active_alerts(org_id)
    return {
        "success": True,
        "data": [a.to_dict() for a in alerts],
        "timestamp": utc_now().isoformat(),
    }


def get_history(db: Session, org_id: int, range_str: str) -> dict:
    service = AlertService(db)
    start_time, _ = parse_time_range(range_str)
    alerts = service.get_alert_history(org_id, start_time)
    return {
        "success": True,
        "data": [a.to_dict() for a in alerts],
        "timestamp": utc_now().isoformat(),
    }


def dismiss(db: Session, org_id: int, alert_id: int) -> dict:
    service = AlertService(db)
    success = service.dismiss_alert(org_id, alert_id)
    return {
        "success": success,
        "message": "Alert dismissed" if success else "Alert not found",
        "timestamp": utc_now().isoformat(),
    }


def acknowledge(db: Session, org_id: int, alert_id: int) -> dict:
    service = AlertService(db)
    success = service.acknowledge_alert(org_id, alert_id)
    return {
        "success": success,
        "message": "Alert acknowledged" if success else "Alert not found",
        "timestamp": utc_now().isoformat(),
    }
