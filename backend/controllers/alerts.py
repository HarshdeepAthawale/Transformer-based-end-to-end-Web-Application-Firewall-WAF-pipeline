"""Alerts controller."""
from datetime import datetime
from sqlalchemy.orm import Session

from backend.services.alert_service import AlertService
from backend.core.time_range import parse_time_range


def get_active(db: Session) -> dict:
    service = AlertService(db)
    alerts = service.get_active_alerts()
    return {
        "success": True,
        "data": [a.to_dict() for a in alerts],
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_history(db: Session, range_str: str) -> dict:
    service = AlertService(db)
    start_time, _ = parse_time_range(range_str)
    alerts = service.get_alert_history(start_time)
    return {
        "success": True,
        "data": [a.to_dict() for a in alerts],
        "timestamp": datetime.utcnow().isoformat(),
    }


def dismiss(db: Session, alert_id: int) -> dict:
    service = AlertService(db)
    success = service.dismiss_alert(alert_id)
    return {
        "success": success,
        "message": "Alert dismissed" if success else "Alert not found",
        "timestamp": datetime.utcnow().isoformat(),
    }


def acknowledge(db: Session, alert_id: int) -> dict:
    service = AlertService(db)
    success = service.acknowledge_alert(alert_id)
    return {
        "success": success,
        "message": "Alert acknowledged" if success else "Alert not found",
        "timestamp": datetime.utcnow().isoformat(),
    }
