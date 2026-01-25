"""Charts controller."""
from datetime import datetime
from sqlalchemy.orm import Session

from backend.services.charts_service import ChartsService
from backend.core.time_range import parse_time_range


def get_requests(db: Session, range_str: str) -> dict:
    service = ChartsService(db)
    start_time, _ = parse_time_range(range_str)
    data = service.get_requests_chart_data(start_time)
    return {"success": True, "data": data, "timestamp": datetime.utcnow().isoformat()}


def get_threats(db: Session, range_str: str) -> dict:
    service = ChartsService(db)
    start_time, _ = parse_time_range(range_str)
    data = service.get_threats_chart_data(start_time)
    return {"success": True, "data": data, "timestamp": datetime.utcnow().isoformat()}


def get_performance(db: Session, range_str: str) -> dict:
    service = ChartsService(db)
    start_time, _ = parse_time_range(range_str)
    data = service.get_performance_chart_data(start_time)
    return {"success": True, "data": data, "timestamp": datetime.utcnow().isoformat()}
