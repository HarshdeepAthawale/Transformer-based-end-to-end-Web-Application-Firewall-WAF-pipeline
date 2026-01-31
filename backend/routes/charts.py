"""Charts API endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.controllers import charts as ctrl

router = APIRouter()


@router.get("/requests")
async def get_requests_chart(
    range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    db: Session = Depends(get_db),
):
    return ctrl.get_requests(db, range)


@router.get("/threats")
async def get_threats_chart(
    range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    db: Session = Depends(get_db),
):
    return ctrl.get_threats(db, range)
