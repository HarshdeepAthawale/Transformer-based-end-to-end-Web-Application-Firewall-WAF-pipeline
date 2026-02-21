"""Analytics controller."""
from backend.lib.datetime_utils import utc_now
from sqlalchemy.orm import Session

from backend.services.analytics_service import AnalyticsService
from backend.core.time_range import parse_time_range


def get_overview(db: Session, range_str: str) -> dict:
    service = AnalyticsService(db)
    start_time, _ = parse_time_range(range_str)
    overview = service.get_overview(start_time)
    return {"success": True, "data": overview, "timestamp": utc_now().isoformat()}


def get_trends(db: Session, metric: str, range_str: str) -> dict:
    service = AnalyticsService(db)
    start_time, _ = parse_time_range(range_str)
    trends = service.get_trends(metric, start_time)
    return {"success": True, "data": trends, "timestamp": utc_now().isoformat()}


def get_summary(db: Session, range_str: str) -> dict:
    service = AnalyticsService(db)
    start_time, _ = parse_time_range(range_str)
    summary = service.get_summary(start_time)
    return {"success": True, "data": summary, "timestamp": utc_now().isoformat()}
