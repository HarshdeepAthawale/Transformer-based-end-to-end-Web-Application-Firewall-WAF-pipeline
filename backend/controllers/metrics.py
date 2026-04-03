"""Metrics controller."""
from backend.lib.datetime_utils import utc_now
from sqlalchemy.orm import Session

from backend.services.metrics_service import MetricsService
from backend.core.time_range import parse_time_range


def get_realtime(db: Session, org_id: int) -> dict:
    service = MetricsService(db)
    metrics = service.get_realtime_metrics(org_id)
    return {
        "success": True,
        "data": metrics,
        "timestamp": utc_now().isoformat(),
    }


def get_historical(db: Session, org_id: int, range_str: str) -> dict:
    service = MetricsService(db)
    start_time, _ = parse_time_range(range_str)
    metrics = service.get_historical_metrics(org_id, start_time)
    return {
        "success": True,
        "data": [m.to_dict() for m in metrics],
        "timestamp": utc_now().isoformat(),
    }
