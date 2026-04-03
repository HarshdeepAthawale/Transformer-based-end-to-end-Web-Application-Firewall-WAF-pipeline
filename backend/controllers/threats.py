"""Threats controller."""
from backend.lib.datetime_utils import utc_now
from sqlalchemy.orm import Session

from backend.services.threat_service import ThreatService
from backend.core.time_range import parse_time_range


def get_recent(db: Session, org_id: int, limit: int) -> dict:
    service = ThreatService(db)
    threats = service.get_recent_threats(org_id, limit)
    return {
        "success": True,
        "data": [t.to_dict() for t in threats],
        "timestamp": utc_now().isoformat(),
    }


def get_by_range(db: Session, org_id: int, range_str: str) -> dict:
    service = ThreatService(db)
    start_time, _ = parse_time_range(range_str)
    threats = service.get_threats_by_range(org_id, start_time)
    return {
        "success": True,
        "data": [t.to_dict() for t in threats],
        "timestamp": utc_now().isoformat(),
    }


def get_by_type(db: Session, org_id: int, threat_type: str, range_str: str) -> dict:
    service = ThreatService(db)
    start_time, _ = parse_time_range(range_str)
    threats = service.get_threats_by_type(org_id, threat_type, start_time)
    return {
        "success": True,
        "data": [t.to_dict() for t in threats],
        "timestamp": utc_now().isoformat(),
    }


def get_stats(db: Session, org_id: int, range_str: str) -> dict:
    service = ThreatService(db)
    start_time, _ = parse_time_range(range_str)
    stats = service.get_threat_stats(org_id, start_time)
    return {"success": True, "data": stats, "timestamp": utc_now().isoformat()}
