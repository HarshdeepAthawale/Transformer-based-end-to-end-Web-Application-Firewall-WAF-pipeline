"""Stats API - aggregate endpoints. Spec path: GET /api/stats/upload-scans."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.security_event import SecurityEvent

router = APIRouter()


@router.get("/upload-scans")
async def get_upload_scan_stats(
    range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d"),
    db: Session = Depends(get_db),
):
    """Get upload scan counts: infected_count, scanned_count. Spec: GET /api/stats/upload-scans."""
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


@router.get("/credential-leak")
async def get_credential_leak_stats(
    range: str = Query("24h", description="Time range: 1h, 6h, 24h, 7d"),
    db: Session = Depends(get_db),
):
    """Get credential leak counts: blocked_count, flagged_count."""
    from backend.core.time_range import parse_time_range

    start_time, _ = parse_time_range(range)
    blocked_count = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type == "credential_leak_block",
            SecurityEvent.timestamp >= start_time,
        )
        .count()
    )
    flagged_count = (
        db.query(SecurityEvent)
        .filter(
            SecurityEvent.event_type == "credential_leak_flag",
            SecurityEvent.timestamp >= start_time,
        )
        .count()
    )
    return {
        "success": True,
        "data": {
            "blocked_count": blocked_count,
            "flagged_count": flagged_count,
        },
    }
