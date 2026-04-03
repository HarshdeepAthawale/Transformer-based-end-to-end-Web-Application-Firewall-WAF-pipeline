"""Threat Intelligence API endpoints."""
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.auth import get_current_tenant
from backend.schemas.threat_intel import ThreatIntelRequest
from backend.controllers import threat_intel as ctrl

router = APIRouter()


@router.get("/feeds")
async def get_threat_feeds(
    threat_type: Optional[str] = None,
    active_only: bool = Query(True),
    limit: int = Query(100, ge=1, le=1000),
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    return ctrl.get_feeds(db, org_id=org_id, threat_type=threat_type, active_only=active_only, limit=limit)


@router.post("/feeds")
async def add_threat_intel(
    request: ThreatIntelRequest,
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    expires_at = None
    if request.expires_at:
        try:
            expires_at = datetime.fromisoformat(request.expires_at)
        except ValueError:
            pass
    return ctrl.add_threat(
        db,
        org_id=org_id,
        threat_type=request.threat_type,
        value=request.value,
        severity=request.severity,
        category=request.category,
        source=request.source,
        description=request.description,
        expires_at=expires_at,
    )


@router.get("/check/{ip}")
async def check_ip_threat(
    ip: str,
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    return ctrl.check_ip(db, org_id=org_id, ip=ip)
