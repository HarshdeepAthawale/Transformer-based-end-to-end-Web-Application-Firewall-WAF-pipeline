"""
Threats API endpoints
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from src.api.database import get_db
from src.api.services.threat_service import ThreatService

router = APIRouter()


@router.get("/recent")
async def get_recent_threats(
    limit: int = Query(20, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Get recent threats"""
    service = ThreatService(db)
    threats = service.get_recent_threats(limit)
    
    return {
        "success": True,
        "data": [threat.to_dict() for threat in threats],
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("")
async def get_threats_by_range(
    range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    db: Session = Depends(get_db)
):
    """Get threats by time range"""
    service = ThreatService(db)
    
    # Parse time range
    hours = 24
    if range.endswith("h"):
        hours = int(range[:-1])
    elif range.endswith("d"):
        hours = int(range[:-1]) * 24
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    threats = service.get_threats_by_range(start_time)
    
    return {
        "success": True,
        "data": [threat.to_dict() for threat in threats],
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/type/{threat_type}")
async def get_threats_by_type(
    threat_type: str,
    range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    db: Session = Depends(get_db)
):
    """Get threats by type"""
    service = ThreatService(db)
    
    # Parse time range
    hours = 24
    if range.endswith("h"):
        hours = int(range[:-1])
    elif range.endswith("d"):
        hours = int(range[:-1]) * 24
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    threats = service.get_threats_by_type(threat_type, start_time)
    
    return {
        "success": True,
        "data": [threat.to_dict() for threat in threats],
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/stats")
async def get_threat_stats(
    range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    db: Session = Depends(get_db)
):
    """Get threat statistics"""
    service = ThreatService(db)
    
    # Parse time range
    hours = 24
    if range.endswith("h"):
        hours = int(range[:-1])
    elif range.endswith("d"):
        hours = int(range[:-1]) * 24
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    stats = service.get_threat_stats(start_time)
    
    return {
        "success": True,
        "data": stats,
        "timestamp": datetime.utcnow().isoformat()
    }
