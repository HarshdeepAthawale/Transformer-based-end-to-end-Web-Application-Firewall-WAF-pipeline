"""
Traffic API endpoints
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from src.api.database import get_db
from src.api.services.traffic_service import TrafficService

router = APIRouter()


@router.get("/recent")
async def get_recent_traffic(
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Get recent traffic logs"""
    service = TrafficService(db)
    traffic_logs = service.get_recent_traffic(limit)
    
    return {
        "success": True,
        "data": [log.to_dict() for log in traffic_logs],
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("")
async def get_traffic_by_range(
    range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    db: Session = Depends(get_db)
):
    """Get traffic logs by time range"""
    service = TrafficService(db)
    
    # Parse time range
    hours = 24
    if range.endswith("h"):
        hours = int(range[:-1])
    elif range.endswith("d"):
        hours = int(range[:-1]) * 24
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    traffic_logs = service.get_traffic_by_range(start_time)
    
    return {
        "success": True,
        "data": [log.to_dict() for log in traffic_logs],
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/endpoint/{endpoint}")
async def get_traffic_by_endpoint(
    endpoint: str,
    range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    db: Session = Depends(get_db)
):
    """Get traffic logs for specific endpoint"""
    service = TrafficService(db)
    
    # Parse time range
    hours = 24
    if range.endswith("h"):
        hours = int(range[:-1])
    elif range.endswith("d"):
        hours = int(range[:-1]) * 24
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    traffic_logs = service.get_traffic_by_endpoint(endpoint, start_time)
    
    return {
        "success": True,
        "data": [log.to_dict() for log in traffic_logs],
        "timestamp": datetime.utcnow().isoformat()
    }
