"""
Geo Rules API endpoints
"""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from src.api.database import get_db
from src.api.services.geo_fencing import GeoFencingService
from src.api.models.geo_rules import GeoRuleType

router = APIRouter()


class GeoRuleRequest(BaseModel):
    rule_type: str  # allow or deny
    country_code: str
    country_name: str
    priority: int = 0
    exception_ips: Optional[List[str]] = None
    reason: Optional[str] = None


@router.get("/rules")
async def get_geo_rules(
    active_only: bool = Query(True),
    db: Session = Depends(get_db)
):
    """Get geo rules"""
    service = GeoFencingService(db)
    rules = service.get_rules(active_only)
    
    return {
        "success": True,
        "data": [rule.to_dict() for rule in rules],
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/rules")
async def create_geo_rule(
    request: GeoRuleRequest,
    db: Session = Depends(get_db)
):
    """Create geo rule"""
    service = GeoFencingService(db)
    
    rule_type = GeoRuleType.ALLOW if request.rule_type == "allow" else GeoRuleType.DENY
    
    rule = service.create_rule(
        rule_type=rule_type,
        country_code=request.country_code,
        country_name=request.country_name,
        priority=request.priority,
        exception_ips=request.exception_ips,
        reason=request.reason
    )
    
    return {
        "success": True,
        "data": rule.to_dict(),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/stats")
async def get_geographic_stats(
    range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    db: Session = Depends(get_db)
):
    """Get geographic threat statistics"""
    from datetime import timedelta
    
    service = GeoFencingService(db)
    
    # Parse time range
    hours = 24
    if range.endswith("h"):
        hours = int(range[:-1])
    elif range.endswith("d"):
        hours = int(range[:-1]) * 24
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    stats = service.get_geographic_stats(start_time)
    
    return {
        "success": True,
        "data": stats,
        "timestamp": datetime.utcnow().isoformat()
    }
