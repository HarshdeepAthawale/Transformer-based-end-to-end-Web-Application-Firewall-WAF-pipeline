"""
Analytics API endpoints
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from src.api.database import get_db
from src.api.services.analytics_service import AnalyticsService

router = APIRouter()


@router.get("/overview")
async def get_analytics_overview(
    range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    db: Session = Depends(get_db)
):
    """Get analytics overview"""
    service = AnalyticsService(db)
    
    # Parse time range
    hours = 24
    if range.endswith("h"):
        hours = int(range[:-1])
    elif range.endswith("d"):
        hours = int(range[:-1]) * 24
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    overview = service.get_overview(start_time)
    
    return {
        "success": True,
        "data": overview,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/trends/{metric}")
async def get_analytics_trends(
    metric: str,
    range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    db: Session = Depends(get_db)
):
    """Get analytics trends for a specific metric"""
    service = AnalyticsService(db)
    
    # Parse time range
    hours = 24
    if range.endswith("h"):
        hours = int(range[:-1])
    elif range.endswith("d"):
        hours = int(range[:-1]) * 24
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    trends = service.get_trends(metric, start_time)
    
    return {
        "success": True,
        "data": trends,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/summary")
async def get_analytics_summary(
    range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    db: Session = Depends(get_db)
):
    """Get analytics summary"""
    service = AnalyticsService(db)
    
    # Parse time range
    hours = 24
    if range.endswith("h"):
        hours = int(range[:-1])
    elif range.endswith("d"):
        hours = int(range[:-1]) * 24
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    summary = service.get_summary(start_time)
    
    return {
        "success": True,
        "data": summary,
        "timestamp": datetime.utcnow().isoformat()
    }
