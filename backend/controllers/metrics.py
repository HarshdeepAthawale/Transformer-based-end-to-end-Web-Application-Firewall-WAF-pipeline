"""Metrics controller."""
from datetime import datetime
from sqlalchemy.orm import Session

from backend.services.metrics_service import MetricsService
from backend.core.time_range import parse_time_range


def get_realtime(db: Session) -> dict:
    service = MetricsService(db)
    metrics = service.get_realtime_metrics()
    return {
        "success": True,
        "data": metrics,
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_historical(db: Session, range_str: str) -> dict:
    service = MetricsService(db)
    start_time, _ = parse_time_range(range_str)
    metrics = service.get_historical_metrics(start_time)
    return {
        "success": True,
        "data": [m.to_dict() for m in metrics],
        "timestamp": datetime.utcnow().isoformat(),
    }
