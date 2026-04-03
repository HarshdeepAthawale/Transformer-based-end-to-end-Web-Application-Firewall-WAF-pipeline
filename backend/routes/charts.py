"""Charts API endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.auth import get_current_tenant
from backend.controllers import charts as ctrl

router = APIRouter()


@router.get("/requests")
async def get_requests_chart(
    range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d, 30d, 90d"),
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    return ctrl.get_requests(db, org_id, range)


@router.get("/threats")
async def get_threats_chart(
    range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d, 30d, 90d"),
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    return ctrl.get_threats(db, org_id, range)


@router.get("/rate-limit")
async def get_rate_limit_chart(
    range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d, 30d, 90d"),
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    return ctrl.get_rate_limit_chart(db, org_id, range)


@router.get("/ddos")
async def get_ddos_chart(
    range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d, 30d, 90d"),
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    return ctrl.get_ddos_chart(db, org_id, range)
