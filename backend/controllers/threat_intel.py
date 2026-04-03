"""Threat intelligence controller."""
from datetime import datetime

from backend.lib.datetime_utils import utc_now
from sqlalchemy.orm import Session
from typing import Optional

from backend.services.threat_intel_service import ThreatIntelService


def get_feeds(
    db: Session,
    org_id: int,
    threat_type: Optional[str] = None,
    active_only: bool = True,
    limit: int = 100,
) -> dict:
    service = ThreatIntelService(db)
    threats = service.get_threats(threat_type, active_only, limit, org_id=org_id)
    return {
        "success": True,
        "data": [t.to_dict() for t in threats],
        "timestamp": utc_now().isoformat(),
    }


def add_threat(
    db: Session,
    org_id: int,
    *,
    threat_type: str,
    value: str,
    severity: str,
    category: str,
    source: str,
    description: Optional[str] = None,
    expires_at: Optional[datetime] = None,
) -> dict:
    service = ThreatIntelService(db)
    threat = service.add_threat(
        threat_type=threat_type,
        value=value,
        severity=severity,
        category=category,
        source=source,
        org_id=org_id,
        description=description,
        expires_at=expires_at,
    )
    return {"success": True, "data": threat.to_dict(), "timestamp": utc_now().isoformat()}


def check_ip(db: Session, org_id: int, ip: str) -> dict:
    service = ThreatIntelService(db)
    result = service.check_threat(ip, org_id=org_id)
    return {"success": True, "data": result, "timestamp": utc_now().isoformat()}
