"""Activities controller."""
from datetime import datetime
from sqlalchemy.orm import Session

from backend.services.activity_service import ActivityService
from backend.core.time_range import parse_time_range


def get_recent(db: Session, limit: int) -> dict:
    service = ActivityService(db)
    activities = service.get_recent_activities(limit)
    return {
        "success": True,
        "data": [a.to_dict() for a in activities],
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_by_range(db: Session, range_str: str) -> dict:
    service = ActivityService(db)
    start_time, _ = parse_time_range(range_str)
    activities = service.get_activities_by_range(start_time)
    return {
        "success": True,
        "data": [a.to_dict() for a in activities],
        "timestamp": datetime.utcnow().isoformat(),
    }
