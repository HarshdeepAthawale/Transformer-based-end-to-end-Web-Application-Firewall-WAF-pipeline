"""Charts controller."""

from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.services.charts_service import ChartsService
from backend.core.time_range import parse_time_range
from backend.models.security_event import SecurityEvent


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


def _aggregate_security_events(db: Session, start_time, event_types: list[str]) -> list:
    """Aggregate security events by hour for chart data."""
    results = (
        db.query(
            func.strftime("%Y-%m-%d %H:00:00", SecurityEvent.timestamp).label("time"),
            func.count(SecurityEvent.id).label("count"),
        )
        .filter(
            SecurityEvent.event_type.in_(event_types),
            SecurityEvent.timestamp >= start_time,
        )
        .group_by("time")
        .order_by("time")
        .all()
    )
    return [{"time": row.time, "count": int(row.count)} for row in results]


def get_rate_limit_chart(db: Session, range_str: str) -> dict:
    start_time, _ = parse_time_range(range_str)
    data = _aggregate_security_events(db, start_time, ["rate_limit"])
    return {"success": True, "data": data, "timestamp": datetime.utcnow().isoformat()}


def get_ddos_chart(db: Session, range_str: str) -> dict:
    start_time, _ = parse_time_range(range_str)
    data = _aggregate_security_events(
        db, start_time, ["ddos_burst", "ddos_blocked", "ddos_size"]
    )
    return {"success": True, "data": data, "timestamp": datetime.utcnow().isoformat()}
