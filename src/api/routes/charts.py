"""
Charts API endpoints
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from src.api.database import get_db
from src.api.services.charts_service import ChartsService

router = APIRouter()


@router.get("/requests")
async def get_requests_chart(
    range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    db: Session = Depends(get_db)
):
    """Get requests chart data"""
    service = ChartsService(db)
    
    # Parse time range
    hours = 24
    if range.endswith("h"):
        hours = int(range[:-1])
    elif range.endswith("d"):
        hours = int(range[:-1]) * 24
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    data = service.get_requests_chart_data(start_time)
    
    return {
        "success": True,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/threats")
async def get_threats_chart(
    range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    db: Session = Depends(get_db)
):
    """Get threats chart data"""
    service = ChartsService(db)
    
    # Parse time range
    hours = 24
    if range.endswith("h"):
        hours = int(range[:-1])
    elif range.endswith("d"):
        hours = int(range[:-1]) * 24
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    data = service.get_threats_chart_data(start_time)
    
    return {
        "success": True,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/performance")
async def get_performance_chart(
    range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    db: Session = Depends(get_db)
):
    """Get performance chart data"""
    service = ChartsService(db)
    
    # Parse time range
    hours = 24
    if range.endswith("h"):
        hours = int(range[:-1])
    elif range.endswith("d"):
        hours = int(range[:-1]) * 24
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    data = service.get_performance_chart_data(start_time)
    
    return {
        "success": True,
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    }
