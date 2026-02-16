"""Events API - ingest and query security events (rate limit, DDoS)."""

import json
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import get_db
from backend.models.security_event import SecurityEvent

router = APIRouter()


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


class IngestEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_type: str
    ip: str
    method: Optional[str] = None
    path: Optional[str] = None
    retry_after: Optional[int] = None
    content_length: Optional[int] = None
    max_bytes: Optional[int] = None
    block_ttl_seconds: Optional[int] = None
    block_duration_seconds: Optional[int] = None


class IngestRequest(BaseModel):
    events: list[IngestEvent]


@router.post("/ingest")
async def ingest_events(body: IngestRequest, db: Session = Depends(get_db)):
    """Accept batch of events from gateway. Fire-and-forget from gateway."""
    for e in body.events:
        details = {}
        block_sec = None
        if e.retry_after is not None:
            details["retry_after"] = e.retry_after
        if e.content_length is not None:
            details["content_length"] = e.content_length
        if e.max_bytes is not None:
            details["max_bytes"] = e.max_bytes
        if e.block_ttl_seconds is not None:
            details["block_ttl_seconds"] = e.block_ttl_seconds
        if e.block_duration_seconds is not None:
            block_sec = e.block_duration_seconds
            details["block_duration_seconds"] = e.block_duration_seconds

        ev = SecurityEvent(
            event_type=e.event_type,
            ip=e.ip,
            method=e.method,
            path=e.path,
            details=json.dumps(details) if details else None,
            block_duration_seconds=block_sec,
        )
        db.add(ev)
    db.commit()
    return {"success": True, "ingested": len(body.events)}


@router.get("/stats")
async def get_events_stats(
    range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d"),
    db: Session = Depends(get_db),
):
    """Get aggregate counts of rate limit and DDoS events for dashboard metrics."""
    from backend.core.time_range import parse_time_range

    start_time, _ = parse_time_range(range)
    rate_limit_count = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type == "rate_limit",
            SecurityEvent.timestamp >= start_time,
        )
        .count()
    )
    ddos_count = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type.in_(("ddos_burst", "ddos_blocked", "ddos_size")),
            SecurityEvent.timestamp >= start_time,
        )
        .count()
    )
    return {
        "success": True,
        "data": {
            "rate_limit_count": rate_limit_count,
            "ddos_count": ddos_count,
        },
    }


@router.get("/rate-limit")
async def list_rate_limit_events(
    range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d"),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
):
    """List rate limit events for dashboard."""
    from backend.core.time_range import parse_time_range

    start_time, _ = parse_time_range(range)
    rows = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type == "rate_limit",
            SecurityEvent.timestamp >= start_time,
        )
        .order_by(SecurityEvent.timestamp.desc())
        .limit(limit)
        .all()
    )
    return {"success": True, "data": [r.to_dict() for r in rows]}


@router.get("/ddos")
async def list_ddos_events(
    range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d"),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
):
    """List DDoS events (burst, blocked, size) for dashboard."""
    from backend.core.time_range import parse_time_range

    start_time, _ = parse_time_range(range)
    rows = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type.in_(
                ("ddos_burst", "ddos_blocked", "ddos_size")
            ),
            SecurityEvent.timestamp >= start_time,
        )
        .order_by(SecurityEvent.timestamp.desc())
        .limit(limit)
        .all()
    )
    return {"success": True, "data": [r.to_dict() for r in rows]}


@router.get("/dos-overview")
async def get_dos_overview(
    range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d, 30d, 90d"),
    limit: int = Query(100, le=500, description="Max events per list"),
    db: Session = Depends(get_db),
):
    """Combined endpoint: stats, chart data, and recent events for DoS/DDoS protection page."""
    from backend.core.time_range import parse_time_range

    start_time, _ = parse_time_range(range)

    # Stats
    rate_limit_count = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type == "rate_limit",
            SecurityEvent.timestamp >= start_time,
        )
        .count()
    )
    ddos_count = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type.in_(("ddos_burst", "ddos_blocked", "ddos_size")),
            SecurityEvent.timestamp >= start_time,
        )
        .count()
    )
    blacklist_count = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type == "blacklist",
            SecurityEvent.timestamp >= start_time,
        )
        .count()
    )

    # Chart data
    chart_rate_limit = _aggregate_security_events(db, start_time, ["rate_limit"])
    chart_ddos = _aggregate_security_events(
        db, start_time, ["ddos_burst", "ddos_blocked", "ddos_size"]
    )
    chart_blacklist = _aggregate_security_events(db, start_time, ["blacklist"])

    # Recent events
    recent_rate_limit = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type == "rate_limit",
            SecurityEvent.timestamp >= start_time,
        )
        .order_by(SecurityEvent.timestamp.desc())
        .limit(limit)
        .all()
    )
    recent_ddos = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type.in_(("ddos_burst", "ddos_blocked", "ddos_size")),
            SecurityEvent.timestamp >= start_time,
        )
        .order_by(SecurityEvent.timestamp.desc())
        .limit(limit)
        .all()
    )
    recent_blacklist = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type == "blacklist",
            SecurityEvent.timestamp >= start_time,
        )
        .order_by(SecurityEvent.timestamp.desc())
        .limit(limit)
        .all()
    )

    return {
        "success": True,
        "data": {
            "stats": {
                "rate_limit_count": rate_limit_count,
                "ddos_count": ddos_count,
                "blacklist_count": blacklist_count,
            },
            "chart_rate_limit": chart_rate_limit,
            "chart_ddos": chart_ddos,
            "chart_blacklist": chart_blacklist,
            "recent_rate_limit": [r.to_dict() for r in recent_rate_limit],
            "recent_ddos": [r.to_dict() for r in recent_ddos],
            "recent_blacklist": [r.to_dict() for r in recent_blacklist],
        },
    }
