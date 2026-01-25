"""Activities API endpoints."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.controllers import activities as ctrl

router = APIRouter()


@router.get("/recent")
async def get_recent_activities(limit: int = Query(10, ge=1, le=100), db: Session = Depends(get_db)):
    return ctrl.get_recent(db, limit)


@router.get("")
async def get_activities_by_range(range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"), db: Session = Depends(get_db)):
    return ctrl.get_by_range(db, range)
