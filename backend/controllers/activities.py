"""Activities controller."""
from backend.lib.datetime_utils import utc_now
from sqlalchemy.orm import Session

from backend.services.activity_service import ActivityService
from backend.core.time_range import parse_time_range


def get_recent(db: Session, org_id: int, limit: int) -> dict:
    service = ActivityService(db)
    activities = service.get_recent_activities(org_id, limit)
    return {
        "success": True,
        "data": [a.to_dict() for a in activities],
        "timestamp": utc_now().isoformat(),
    }


def get_by_range(db: Session, org_id: int, range_str: str) -> dict:
    service = ActivityService(db)
    start_time, _ = parse_time_range(range_str)
    activities = service.get_activities_by_range(org_id, start_time)
    return {
        "success": True,
        "data": [a.to_dict() for a in activities],
        "timestamp": utc_now().isoformat(),
    }
