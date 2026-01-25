"""Traffic controller."""
from datetime import datetime
from sqlalchemy.orm import Session

from backend.services.traffic_service import TrafficService
from backend.core.time_range import parse_time_range


def get_recent(db: Session, limit: int) -> dict:
    service = TrafficService(db)
    logs = service.get_recent_traffic(limit)
    return {
        "success": True,
        "data": [log.to_dict() for log in logs],
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_by_range(db: Session, range_str: str) -> dict:
    service = TrafficService(db)
    start_time, _ = parse_time_range(range_str)
    logs = service.get_traffic_by_range(start_time)
    return {
        "success": True,
        "data": [log.to_dict() for log in logs],
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_by_endpoint(db: Session, endpoint: str, range_str: str) -> dict:
    service = TrafficService(db)
    start_time, _ = parse_time_range(range_str)
    logs = service.get_traffic_by_endpoint(endpoint, start_time)
    return {
        "success": True,
        "data": [log.to_dict() for log in logs],
        "timestamp": datetime.utcnow().isoformat(),
    }
