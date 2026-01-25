"""Threats controller."""
from datetime import datetime
from sqlalchemy.orm import Session

from backend.services.threat_service import ThreatService
from backend.core.time_range import parse_time_range


def get_recent(db: Session, limit: int) -> dict:
    service = ThreatService(db)
    threats = service.get_recent_threats(limit)
    return {
        "success": True,
        "data": [t.to_dict() for t in threats],
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_by_range(db: Session, range_str: str) -> dict:
    service = ThreatService(db)
    start_time, _ = parse_time_range(range_str)
    threats = service.get_threats_by_range(start_time)
    return {
        "success": True,
        "data": [t.to_dict() for t in threats],
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_by_type(db: Session, threat_type: str, range_str: str) -> dict:
    service = ThreatService(db)
    start_time, _ = parse_time_range(range_str)
    threats = service.get_threats_by_type(threat_type, start_time)
    return {
        "success": True,
        "data": [t.to_dict() for t in threats],
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_stats(db: Session, range_str: str) -> dict:
    service = ThreatService(db)
    start_time, _ = parse_time_range(range_str)
    stats = service.get_threat_stats(start_time)
    return {"success": True, "data": stats, "timestamp": datetime.utcnow().isoformat()}
