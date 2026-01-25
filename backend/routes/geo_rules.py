"""Geo Rules API endpoints."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas.geo_rules import GeoRuleRequest
from backend.controllers import geo_rules as ctrl

router = APIRouter()


@router.get("/rules")
async def get_geo_rules(active_only: bool = Query(True), db: Session = Depends(get_db)):
    return ctrl.get_rules(db, active_only)


@router.post("/rules")
async def create_geo_rule(request: GeoRuleRequest, db: Session = Depends(get_db)):
    return ctrl.create_rule(
        db,
        rule_type=request.rule_type,
        country_code=request.country_code,
        country_name=request.country_name,
        priority=request.priority,
        exception_ips=request.exception_ips,
        reason=request.reason,
    )


@router.get("/stats")
async def get_geographic_stats(range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"), db: Session = Depends(get_db)):
    return ctrl.get_geographic_stats(db, range)
