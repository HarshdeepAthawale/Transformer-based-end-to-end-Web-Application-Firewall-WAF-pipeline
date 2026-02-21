"""Audit controller."""
from datetime import datetime

from backend.lib.datetime_utils import utc_now
from sqlalchemy.orm import Session
from typing import Optional

from backend.models.audit_log import AuditLog, AuditAction


def get_logs(
    db: Session,
    limit: int = 100,
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    start_time: Optional[str] = None,
) -> dict:
    query = db.query(AuditLog)
    if action:
        try:
            ae = AuditAction[action.upper()]
            query = query.filter(AuditLog.action == ae)
        except KeyError:
            pass
    if resource_type:
        query = query.filter(AuditLog.resource_type == resource_type)
    if start_time:
        try:
            st = datetime.fromisoformat(start_time)
            query = query.filter(AuditLog.timestamp >= st)
        except ValueError:
            pass
    logs = query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
    return {
        "success": True,
        "data": [log.to_dict() for log in logs],
        "timestamp": utc_now().isoformat(),
    }


def get_log(db: Session, log_id: int) -> dict:
    log = db.query(AuditLog).filter(AuditLog.id == log_id).first()
    if not log:
        return {
            "success": False,
            "message": "Audit log not found",
            "timestamp": utc_now().isoformat(),
        }
    return {"success": True, "data": log.to_dict(), "timestamp": utc_now().isoformat()}
