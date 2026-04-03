"""Metrics API endpoints."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.auth import get_current_tenant
from backend.controllers import metrics as ctrl

router = APIRouter()


@router.get("/realtime")
async def get_realtime_metrics(org_id: int = Depends(get_current_tenant), db: Session = Depends(get_db)):
    return ctrl.get_realtime(db, org_id)


@router.get("/historical")
async def get_historical_metrics(range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"), org_id: int = Depends(get_current_tenant), db: Session = Depends(get_db)):
    return ctrl.get_historical(db, org_id, range)
