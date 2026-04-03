"""Threats API endpoints."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.auth import get_current_tenant
from backend.controllers import threats as ctrl

router = APIRouter()


@router.get("/recent")
async def get_recent_threats(limit: int = Query(20, ge=1, le=500), org_id: int = Depends(get_current_tenant), db: Session = Depends(get_db)):
    return ctrl.get_recent(db, org_id, limit)


@router.get("")
async def get_threats_by_range(range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"), org_id: int = Depends(get_current_tenant), db: Session = Depends(get_db)):
    return ctrl.get_by_range(db, org_id, range)


@router.get("/type/{threat_type}")
async def get_threats_by_type(threat_type: str, range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"), org_id: int = Depends(get_current_tenant), db: Session = Depends(get_db)):
    return ctrl.get_by_type(db, org_id, threat_type, range)


@router.get("/stats")
async def get_threat_stats(range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"), org_id: int = Depends(get_current_tenant), db: Session = Depends(get_db)):
    return ctrl.get_stats(db, org_id, range)
