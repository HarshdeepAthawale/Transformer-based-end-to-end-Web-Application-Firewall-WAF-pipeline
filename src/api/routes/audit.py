"""
Audit Log API endpoints
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional

from src.api.database import get_db
from src.api.models.audit_log import AuditLog, AuditAction
from src.api.auth import get_current_user, require_role
from src.api.models.users import User, UserRole

router = APIRouter()


@router.get("/logs")
async def get_audit_logs(
    limit: int = Query(100, ge=1, le=1000),
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    start_time: Optional[str] = None,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: Session = Depends(get_db)
):
    """Get audit logs"""
    query = db.query(AuditLog)
    
    if action:
        try:
            action_enum = AuditAction[action.upper()]
            query = query.filter(AuditLog.action == action_enum)
        except KeyError:
            pass
    
    if resource_type:
        query = query.filter(AuditLog.resource_type == resource_type)
    
    if start_time:
        try:
            start = datetime.fromisoformat(start_time)
            query = query.filter(AuditLog.timestamp >= start)
        except ValueError:
            pass
    
    logs = query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
    
    return {
        "success": True,
        "data": [log.to_dict() for log in logs],
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/logs/{log_id}")
async def get_audit_log(
    log_id: int,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: Session = Depends(get_db)
):
    """Get specific audit log"""
    log = db.query(AuditLog).filter(AuditLog.id == log_id).first()
    
    if not log:
        return {
            "success": False,
            "message": "Audit log not found",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    return {
        "success": True,
        "data": log.to_dict(),
        "timestamp": datetime.utcnow().isoformat()
    }
