"""Traffic API endpoints."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.auth import get_current_tenant
from backend.controllers import traffic as ctrl

router = APIRouter()


@router.get("/recent")
async def get_recent_traffic(limit: int = Query(50, ge=1, le=2000), org_id: int = Depends(get_current_tenant), db: Session = Depends(get_db)):
    return ctrl.get_recent(db, org_id, limit)


@router.get("")
async def get_traffic_by_range(range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"), org_id: int = Depends(get_current_tenant), db: Session = Depends(get_db)):
    return ctrl.get_by_range(db, org_id, range)


@router.get("/endpoint/{endpoint}")
async def get_traffic_by_endpoint(endpoint: str, range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"), org_id: int = Depends(get_current_tenant), db: Session = Depends(get_db)):
    return ctrl.get_by_endpoint(db, org_id, endpoint, range)
