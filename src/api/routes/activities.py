"""
Activities API endpoints
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from src.api.database import get_db
from src.api.services.activity_service import ActivityService

router = APIRouter()


@router.get("/recent")
async def get_recent_activities(
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get recent activities"""
    service = ActivityService(db)
    activities = service.get_recent_activities(limit)
    
    return {
        "success": True,
        "data": [activity.to_dict() for activity in activities],
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("")
async def get_activities_by_range(
    range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    db: Session = Depends(get_db)
):
    """Get activities by time range"""
    service = ActivityService(db)
    
    # Parse time range
    hours = 24
    if range.endswith("h"):
        hours = int(range[:-1])
    elif range.endswith("d"):
        hours = int(range[:-1]) * 24
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    activities = service.get_activities_by_range(start_time)
    
    return {
        "success": True,
        "data": [activity.to_dict() for activity in activities],
        "timestamp": datetime.utcnow().isoformat()
    }
