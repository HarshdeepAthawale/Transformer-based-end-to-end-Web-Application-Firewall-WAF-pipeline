"""
Threat Intelligence API endpoints
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from src.api.database import get_db
from src.api.services.threat_intel_service import ThreatIntelService

router = APIRouter()


class ThreatIntelRequest(BaseModel):
    threat_type: str  # ip, domain, signature
    value: str
    severity: str  # critical, high, medium, low
    category: str
    source: str
    description: Optional[str] = None
    expires_at: Optional[str] = None


@router.get("/feeds")
async def get_threat_feeds(
    threat_type: Optional[str] = None,
    active_only: bool = Query(True),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get threat intelligence feeds"""
    service = ThreatIntelService(db)
    threats = service.get_threats(threat_type, active_only, limit)
    
    return {
        "success": True,
        "data": [threat.to_dict() for threat in threats],
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/feeds")
async def add_threat_intel(
    request: ThreatIntelRequest,
    db: Session = Depends(get_db)
):
    """Add threat intelligence entry"""
    service = ThreatIntelService(db)
    
    expires_at = None
    if request.expires_at:
        expires_at = datetime.fromisoformat(request.expires_at)
    
    threat = service.add_threat(
        threat_type=request.threat_type,
        value=request.value,
        severity=request.severity,
        category=request.category,
        source=request.source,
        description=request.description,
        expires_at=expires_at
    )
    
    return {
        "success": True,
        "data": threat.to_dict(),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/check/{ip}")
async def check_ip_threat(
    ip: str,
    db: Session = Depends(get_db)
):
    """Check IP against threat intelligence"""
    service = ThreatIntelService(db)
    result = service.check_threat(ip)
    
    return {
        "success": True,
        "data": result,
        "timestamp": datetime.utcnow().isoformat()
    }
