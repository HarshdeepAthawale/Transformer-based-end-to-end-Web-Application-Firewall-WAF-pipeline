"""Traffic controller."""
from backend.lib.datetime_utils import utc_now
from sqlalchemy.orm import Session

from backend.services.traffic_service import TrafficService
from backend.core.time_range import parse_time_range


def get_recent(db: Session, org_id: int, limit: int) -> dict:
    service = TrafficService(db)
    logs = service.get_recent_traffic(org_id, limit)
    return {
        "success": True,
        "data": [log.to_dict() for log in logs],
        "timestamp": utc_now().isoformat(),
    }


def get_by_range(db: Session, org_id: int, range_str: str) -> dict:
    service = TrafficService(db)
    start_time, _ = parse_time_range(range_str)
    logs = service.get_traffic_by_range(org_id, start_time)
    return {
        "success": True,
        "data": [log.to_dict() for log in logs],
        "timestamp": utc_now().isoformat(),
    }


def get_by_endpoint(db: Session, org_id: int, endpoint: str, range_str: str) -> dict:
    service = TrafficService(db)
    start_time, _ = parse_time_range(range_str)
    logs = service.get_traffic_by_endpoint(org_id, endpoint, start_time)
    return {
        "success": True,
        "data": [log.to_dict() for log in logs],
        "timestamp": utc_now().isoformat(),
    }
