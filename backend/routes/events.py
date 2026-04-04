"""Events API - ingest and query security events (rate limit, DDoS)."""

import json
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session
from sqlalchemy import func
from loguru import logger
from backend.lib.db_utils import hour_bucket

from backend.database import get_db
from backend.models.security_event import SecurityEvent
from backend.config import config
from backend.services.traffic_service import TrafficService
from backend.services.websocket_service import broadcast_update_sync

router = APIRouter()


def _aggregate_security_events(db: Session, start_time, event_types: list[str]) -> list:
    """Aggregate security events by hour for chart data. Returns empty list on DB/strftime errors."""
    try:
        results = (
            db.query(
                hour_bucket(SecurityEvent.timestamp).label("time"),
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
    except Exception as e:
        logger.warning(f"Events aggregation failed: {e}")
        return []


class IngestEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    event_type: str
    ip: str
    method: Optional[str] = None
    path: Optional[str] = None
    request_id: Optional[str] = None
    retry_after: Optional[int] = None
    content_length: Optional[int] = None
    max_bytes: Optional[int] = None
    block_ttl_seconds: Optional[int] = None
    block_duration_seconds: Optional[int] = None
    attack_score: Optional[int] = None
    bot_score: Optional[int] = None
    bot_action: Optional[str] = None
    rule_id: Optional[int] = None
    pack_id: Optional[str] = None
    # Upload scan
    upload_scan_result: Optional[str] = None
    upload_scan_signature: Optional[str] = None
    upload_filename: Optional[str] = None
    upload_size_bytes: Optional[int] = None
    upload_scan_engine: Optional[str] = None
    upload_content_type: Optional[str] = None
    # Firewall for AI
    firewall_ai_reason: Optional[str] = None
    firewall_ai_pattern: Optional[str] = None
    firewall_ai_action: Optional[str] = None
    # Credential leak (no password; optional hash prefix for debugging)
    credential_leak_username: Optional[str] = None
    credential_leak_hash_prefix: Optional[str] = None


class IngestRequest(BaseModel):
    events: list[IngestEvent]


@router.post("/ingest")
async def ingest_events(body: IngestRequest, db: Session = Depends(get_db)):
    """Accept batch of events from gateway. Fire-and-forget from gateway.
    When EVENTS_INGEST_WRITE_TRAFFIC_LOG is true, also creates TrafficLog rows
    so Request Volume & Threats chart shows gateway traffic."""
    traffic_svc = TrafficService(db) if config.EVENTS_INGEST_WRITE_TRAFFIC_LOG else None
    traffic_logs_created: list = []

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
        if e.attack_score is not None:
            details["attack_score"] = e.attack_score
        if e.bot_action is not None:
            details["bot_action"] = e.bot_action
        if e.rule_id is not None:
            details["rule_id"] = e.rule_id
        if e.pack_id is not None:
            details["pack_id"] = e.pack_id
        if e.upload_scan_result is not None:
            details["upload_scan_result"] = e.upload_scan_result
        if e.upload_scan_signature is not None:
            details["upload_scan_signature"] = e.upload_scan_signature
        if e.upload_filename is not None:
            details["upload_filename"] = e.upload_filename
        if e.upload_size_bytes is not None:
            details["upload_size_bytes"] = e.upload_size_bytes
        if e.upload_scan_engine is not None:
            details["upload_scan_engine"] = e.upload_scan_engine
        if e.upload_content_type is not None:
            details["upload_content_type"] = e.upload_content_type
        if e.firewall_ai_reason is not None:
            details["firewall_ai_reason"] = e.firewall_ai_reason
        if e.firewall_ai_pattern is not None:
            details["firewall_ai_pattern"] = e.firewall_ai_pattern
        if e.firewall_ai_action is not None:
            details["firewall_ai_action"] = e.firewall_ai_action
        if e.credential_leak_username is not None:
            details["credential_leak_username"] = e.credential_leak_username
        if e.credential_leak_hash_prefix is not None:
            details["credential_leak_hash_prefix"] = e.credential_leak_hash_prefix

        ev = SecurityEvent(
            event_type=e.event_type,
            ip=e.ip,
            method=e.method,
            path=e.path,
            details=json.dumps(details) if details else None,
            attack_score=e.attack_score,
            block_duration_seconds=block_sec,
            bot_score=e.bot_score,
        )
        db.add(ev)

        if traffic_svc:
            log = traffic_svc.add_traffic_log_from_ingest_event(
                event_type=e.event_type,
                ip=e.ip,
                method=e.method,
                path=e.path,
                attack_score=e.attack_score,
                bot_score=e.bot_score,
                bot_action=e.bot_action,
            )
            traffic_logs_created.append(log)

    db.commit()
    for log in traffic_logs_created:
        db.refresh(log)
        broadcast_update_sync("traffic", log.to_dict())

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
    waf_block_count = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type.in_(("waf_block", "waf_challenge", "waf")),
            SecurityEvent.timestamp >= start_time,
        )
        .count()
    )
    bot_block_count = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type.in_(("bot_block", "bot_challenge")),
            SecurityEvent.timestamp >= start_time,
        )
        .count()
    )
    avg_attack_score_row = (
        db.query(func.avg(SecurityEvent.attack_score))
        .filter(
            SecurityEvent.attack_score.isnot(None),
            SecurityEvent.timestamp >= start_time,
        )
        .scalar()
    )
    return {
        "success": True,
        "data": {
            "rate_limit_count": rate_limit_count,
            "ddos_count": ddos_count,
            "waf_block_count": waf_block_count,
            "bot_block_count": bot_block_count,
            "avg_attack_score": round(avg_attack_score_row, 1) if avg_attack_score_row is not None else None,
        },
    }


@router.get("/waf")
async def list_waf_events(
    range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d, 30d, 90d"),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
):
    """List WAF block/challenge events with attack scores."""
    from backend.core.time_range import parse_time_range

    start_time, _ = parse_time_range(range)
    rows = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type.in_(("waf_block", "waf_challenge", "waf")),
            SecurityEvent.timestamp >= start_time,
        )
        .order_by(SecurityEvent.timestamp.desc())
        .limit(limit)
        .all()
    )
    return {"success": True, "data": [r.to_dict() for r in rows]}


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


@router.get("/bot")
async def list_bot_events(
    range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d"),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
):
    """List bot block/challenge events with bot_score and action."""
    from backend.core.time_range import parse_time_range

    start_time, _ = parse_time_range(range)
    rows = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type.in_(("bot_block", "bot_challenge")),
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


@router.get("/upload-scans")
async def list_upload_scan_events(
    range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d"),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
):
    """List upload scan events (infected and clean) with filename, size, result, signature."""
    from backend.core.time_range import parse_time_range

    start_time, _ = parse_time_range(range)
    rows = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type.in_(("upload_scan_infected", "upload_scan_clean")),
            SecurityEvent.timestamp >= start_time,
        )
        .order_by(SecurityEvent.timestamp.desc())
        .limit(limit)
        .all()
    )
    return {"success": True, "data": [r.to_dict() for r in rows]}


@router.get("/credential-leak")
async def list_credential_leak_events(
    range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d"),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
):
    """List credential leak events (block/flag). No password or sensitive data."""
    from backend.core.time_range import parse_time_range

    start_time, _ = parse_time_range(range)
    rows = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type.in_(("credential_leak_block", "credential_leak_flag")),
            SecurityEvent.timestamp >= start_time,
        )
        .order_by(SecurityEvent.timestamp.desc())
        .limit(limit)
        .all()
    )
    return {"success": True, "data": [r.to_dict() for r in rows]}


@router.get("/firewall-ai")
async def list_firewall_ai_events(
    range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d"),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
):
    """List Firewall-for-AI events (prompt_block, pii, abuse_rate)."""
    from backend.core.time_range import parse_time_range

    start_time, _ = parse_time_range(range)
    rows = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type.in_(
                ("firewall_ai_prompt_block", "firewall_ai_pii", "firewall_ai_abuse_rate")
            ),
            SecurityEvent.timestamp >= start_time,
        )
        .order_by(SecurityEvent.timestamp.desc())
        .limit(limit)
        .all()
    )
    return {"success": True, "data": [r.to_dict() for r in rows]}


@router.get("/upload-scan-stats")
async def get_upload_scan_stats(
    range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d"),
    db: Session = Depends(get_db),
):
    """Get upload scan counts: infected_count, scanned_count."""
    from backend.core.time_range import parse_time_range

    start_time, _ = parse_time_range(range)
    infected_count = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type == "upload_scan_infected",
            SecurityEvent.timestamp >= start_time,
        )
        .count()
    )
    scanned_count = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type.in_(("upload_scan_infected", "upload_scan_clean")),
            SecurityEvent.timestamp >= start_time,
        )
        .count()
    )
    return {
        "success": True,
        "data": {
            "infected_count": infected_count,
            "scanned_count": scanned_count,
        },
    }


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
