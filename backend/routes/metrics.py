"""Metrics API endpoints."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.controllers import metrics as ctrl

router = APIRouter()


@router.get("/realtime")
async def get_realtime_metrics(db: Session = Depends(get_db)):
    return ctrl.get_realtime(db)


@router.get("/historical")
async def get_historical_metrics(range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"), db: Session = Depends(get_db)):
    return ctrl.get_historical(db, range)
