"""Geo rules controller."""
from datetime import datetime
from sqlalchemy.orm import Session
from typing import List, Optional

from backend.services.geo_fencing import GeoFencingService
from backend.models.geo_rules import GeoRuleType
from backend.core.time_range import parse_time_range


def get_rules(db: Session, active_only: bool) -> dict:
    service = GeoFencingService(db)
    rules = service.get_rules(active_only)
    return {
        "success": True,
        "data": [r.to_dict() for r in rules],
        "timestamp": datetime.utcnow().isoformat(),
    }


def create_rule(
    db: Session,
    *,
    rule_type: str,
    country_code: str,
    country_name: str,
    priority: int = 0,
    exception_ips: Optional[List[str]] = None,
    reason: Optional[str] = None,
) -> dict:
    service = GeoFencingService(db)
    rt = GeoRuleType.ALLOW if rule_type == "allow" else GeoRuleType.DENY
    rule = service.create_rule(
        rule_type=rt,
        country_code=country_code,
        country_name=country_name,
        priority=priority,
        exception_ips=exception_ips,
        reason=reason,
    )
    return {"success": True, "data": rule.to_dict(), "timestamp": datetime.utcnow().isoformat()}


def get_geographic_stats(db: Session, range_str: str) -> dict:
    service = GeoFencingService(db)
    start_time, _ = parse_time_range(range_str)
    stats = service.get_geographic_stats(start_time)
    return {"success": True, "data": stats, "timestamp": datetime.utcnow().isoformat()}
