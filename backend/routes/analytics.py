"""Analytics API endpoints."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.controllers import analytics as ctrl

router = APIRouter()


@router.get("/overview")
async def get_analytics_overview(range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"), db: Session = Depends(get_db)):
    return ctrl.get_overview(db, range)


@router.get("/trends/{metric}")
async def get_analytics_trends(metric: str, range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"), db: Session = Depends(get_db)):
    return ctrl.get_trends(db, metric, range)


@router.get("/summary")
async def get_analytics_summary(range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"), db: Session = Depends(get_db)):
    return ctrl.get_summary(db, range)
