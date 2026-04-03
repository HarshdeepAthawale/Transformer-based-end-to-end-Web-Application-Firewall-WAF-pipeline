"""Unified security dashboard: overview, charts, recent events from security_events."""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database import get_db
from backend.auth import get_current_tenant
from backend.models.security_event import SecurityEvent

router = APIRouter()


def _aggregate_events(db: Session, start_time, event_types: list[str]) -> list[dict]:
    """Aggregate security events by hour for chart series."""
    try:
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
        return [{"time": row.time, "count": int(row.count)} for row in results if row.time]
    except Exception:
        return []


@router.get("/overview")
async def get_dashboard_overview(
    range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d, 30d"),
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Aggregate counts and averages for dashboard cards. No mocks."""
    from backend.core.time_range import parse_time_range

    start_time, _ = parse_time_range(range)

    waf_types = ["waf_block", "waf_challenge", "waf"]
    waf_block_count = (
        db.query(SecurityEvent)
        .filter(SecurityEvent.event_type.in_(waf_types), SecurityEvent.timestamp >= start_time)
        .count()
    )
    rate_limit_count = (
        db.query(SecurityEvent)
        .filter(SecurityEvent.event_type == "rate_limit", SecurityEvent.timestamp >= start_time)
        .count()
    )
    ddos_types = ["ddos_burst", "ddos_blocked", "ddos_size"]
    ddos_count = (
        db.query(SecurityEvent)
        .filter(SecurityEvent.event_type.in_(ddos_types), SecurityEvent.timestamp >= start_time)
        .count()
    )
    bot_types = ["bot_block", "bot_challenge"]
    bot_block_count = (
        db.query(SecurityEvent)
        .filter(SecurityEvent.event_type.in_(bot_types), SecurityEvent.timestamp >= start_time)
        .count()
    )
    upload_scan_infected_count = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type == "upload_scan_infected",
            SecurityEvent.timestamp >= start_time,
        )
        .count()
    )
    credential_leak_block_count = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type == "credential_leak_block",
            SecurityEvent.timestamp >= start_time,
        )
        .count()
    )
    firewall_ai_types = ["firewall_ai_prompt_block", "firewall_ai_pii", "firewall_ai_abuse_rate"]
    firewall_ai_block_count = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type.in_(firewall_ai_types),
            SecurityEvent.timestamp >= start_time,
        )
        .count()
    )
    avg_attack = (
        db.query(func.avg(SecurityEvent.attack_score))
        .filter(
            SecurityEvent.attack_score.isnot(None),
            SecurityEvent.timestamp >= start_time,
        )
        .scalar()
    )
    avg_bot = (
        db.query(func.avg(SecurityEvent.bot_score))
        .filter(
            SecurityEvent.bot_score.isnot(None),
            SecurityEvent.timestamp >= start_time,
        )
        .scalar()
    )

    return {
        "success": True,
        "data": {
            "waf_block_count": waf_block_count,
            "rate_limit_count": rate_limit_count,
            "ddos_count": ddos_count,
            "bot_block_count": bot_block_count,
            "upload_scan_infected_count": upload_scan_infected_count,
            "credential_leak_block_count": credential_leak_block_count,
            "firewall_ai_block_count": firewall_ai_block_count,
            "avg_attack_score": round(avg_attack, 1) if avg_attack is not None else None,
            "avg_bot_score": round(avg_bot, 1) if avg_bot is not None else None,
        },
    }


@router.get("/charts")
async def get_dashboard_charts(
    range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d, 30d"),
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Time-series series per event type (hourly buckets). Data from DB."""
    from backend.core.time_range import parse_time_range

    start_time, _ = parse_time_range(range)

    series = [
        {"name": "waf_block", "data": _aggregate_events(db, start_time, ["waf_block", "waf_challenge", "waf"])},
        {"name": "rate_limit", "data": _aggregate_events(db, start_time, ["rate_limit"])},
        {"name": "ddos", "data": _aggregate_events(db, start_time, ["ddos_burst", "ddos_blocked", "ddos_size"])},
        {"name": "bot_block", "data": _aggregate_events(db, start_time, ["bot_block", "bot_challenge"])},
        {"name": "upload_scan_infected", "data": _aggregate_events(db, start_time, ["upload_scan_infected"])},
        {"name": "credential_leak_block", "data": _aggregate_events(db, start_time, ["credential_leak_block"])},
        {
            "name": "firewall_ai",
            "data": _aggregate_events(
                db,
                start_time,
                ["firewall_ai_prompt_block", "firewall_ai_pii", "firewall_ai_abuse_rate"],
            ),
        },
    ]
    return {"success": True, "data": {"series": series}}


@router.get("/events")
async def get_dashboard_events(
    range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d, 30d"),
    limit: int = Query(50, ge=1, le=200),
    event_type: Optional[str] = Query(None, description="Filter by event_type"),
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Recent security events for dashboard table. Optional event_type filter."""
    from backend.core.time_range import parse_time_range

    start_time, _ = parse_time_range(range)
    q = (
        db.query(SecurityEvent)
        .filter(SecurityEvent.timestamp >= start_time)
        .order_by(SecurityEvent.timestamp.desc())
        .limit(limit)
    )
    if event_type and event_type.strip():
        q = q.filter(SecurityEvent.event_type == event_type.strip())
    rows = q.all()
    return {"success": True, "data": [r.to_dict() for r in rows]}


@router.get("/unified")
async def get_dashboard_unified(
    range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d, 30d"),
    limit: int = Query(50, ge=1, le=200),
    event_type: Optional[str] = Query(None),
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Single response: overview + charts + recent_events. Same data as separate endpoints."""
    from backend.core.time_range import parse_time_range

    start_time, _ = parse_time_range(range)

    # Overview
    waf_types = ["waf_block", "waf_challenge", "waf"]
    waf_block_count = (
        db.query(SecurityEvent)
        .filter(SecurityEvent.event_type.in_(waf_types), SecurityEvent.timestamp >= start_time)
        .count()
    )
    rate_limit_count = (
        db.query(SecurityEvent)
        .filter(SecurityEvent.event_type == "rate_limit", SecurityEvent.timestamp >= start_time)
        .count()
    )
    ddos_count = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type.in_(["ddos_burst", "ddos_blocked", "ddos_size"]),
            SecurityEvent.timestamp >= start_time,
        )
        .count()
    )
    bot_block_count = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type.in_(["bot_block", "bot_challenge"]),
            SecurityEvent.timestamp >= start_time,
        )
        .count()
    )
    upload_scan_infected_count = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type == "upload_scan_infected",
            SecurityEvent.timestamp >= start_time,
        )
        .count()
    )
    credential_leak_block_count = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type == "credential_leak_block",
            SecurityEvent.timestamp >= start_time,
        )
        .count()
    )
    firewall_ai_block_count = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type.in_(
                ["firewall_ai_prompt_block", "firewall_ai_pii", "firewall_ai_abuse_rate"]
            ),
            SecurityEvent.timestamp >= start_time,
        )
        .count()
    )
    avg_attack = (
        db.query(func.avg(SecurityEvent.attack_score))
        .filter(
            SecurityEvent.attack_score.isnot(None),
            SecurityEvent.timestamp >= start_time,
        )
        .scalar()
    )
    avg_bot = (
        db.query(func.avg(SecurityEvent.bot_score))
        .filter(
            SecurityEvent.bot_score.isnot(None),
            SecurityEvent.timestamp >= start_time,
        )
        .scalar()
    )
    overview = {
        "waf_block_count": waf_block_count,
        "rate_limit_count": rate_limit_count,
        "ddos_count": ddos_count,
        "bot_block_count": bot_block_count,
        "upload_scan_infected_count": upload_scan_infected_count,
        "credential_leak_block_count": credential_leak_block_count,
        "firewall_ai_block_count": firewall_ai_block_count,
        "avg_attack_score": round(avg_attack, 1) if avg_attack is not None else None,
        "avg_bot_score": round(avg_bot, 1) if avg_bot is not None else None,
    }

    # Charts
    series = [
        {"name": "waf_block", "data": _aggregate_events(db, start_time, waf_types)},
        {"name": "rate_limit", "data": _aggregate_events(db, start_time, ["rate_limit"])},
        {"name": "ddos", "data": _aggregate_events(db, start_time, ["ddos_burst", "ddos_blocked", "ddos_size"])},
        {"name": "bot_block", "data": _aggregate_events(db, start_time, ["bot_block", "bot_challenge"])},
        {"name": "upload_scan_infected", "data": _aggregate_events(db, start_time, ["upload_scan_infected"])},
        {"name": "credential_leak_block", "data": _aggregate_events(db, start_time, ["credential_leak_block"])},
        {
            "name": "firewall_ai",
            "data": _aggregate_events(
                db,
                start_time,
                ["firewall_ai_prompt_block", "firewall_ai_pii", "firewall_ai_abuse_rate"],
            ),
        },
    ]

    # Recent events
    q = (
        db.query(SecurityEvent)
        .filter(SecurityEvent.timestamp >= start_time)
        .order_by(SecurityEvent.timestamp.desc())
        .limit(limit)
    )
    if event_type and event_type.strip():
        q = q.filter(SecurityEvent.event_type == event_type.strip())
    recent_events = [r.to_dict() for r in q.all()]

    return {
        "success": True,
        "data": {
            "overview": overview,
            "charts": {"series": series},
            "recent_events": recent_events,
        },
    }
