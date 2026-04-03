"""Audit Log API endpoints."""
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.auth import get_current_tenant
from backend.controllers import audit as ctrl

router = APIRouter()


@router.get("/logs")
async def get_audit_logs(
    limit: int = Query(100, ge=1, le=1000),
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    start_time: Optional[str] = None,
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    return ctrl.get_logs(db, org_id, limit=limit, action=action, resource_type=resource_type, start_time=start_time)


@router.get("/logs/{log_id}")
async def get_audit_log(
    log_id: int,
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    return ctrl.get_log(db, org_id, log_id)
