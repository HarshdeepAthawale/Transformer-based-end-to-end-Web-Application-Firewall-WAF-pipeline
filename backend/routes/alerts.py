"""Alerts API endpoints."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.auth import get_current_tenant
from backend.controllers import alerts as ctrl

router = APIRouter()


@router.get("/active")
async def get_active_alerts(org_id: int = Depends(get_current_tenant), db: Session = Depends(get_db)):
    return ctrl.get_active(db, org_id)


@router.get("/history")
async def get_alert_history(range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"), org_id: int = Depends(get_current_tenant), db: Session = Depends(get_db)):
    return ctrl.get_history(db, org_id, range)


@router.post("/{alert_id}/dismiss")
async def dismiss_alert(alert_id: int, org_id: int = Depends(get_current_tenant), db: Session = Depends(get_db)):
    return ctrl.dismiss(db, org_id, alert_id)


@router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int, org_id: int = Depends(get_current_tenant), db: Session = Depends(get_db)):
    return ctrl.acknowledge(db, org_id, alert_id)
