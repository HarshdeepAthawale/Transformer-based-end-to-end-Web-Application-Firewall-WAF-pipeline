"""IP management controller."""
from datetime import datetime
from sqlalchemy.orm import Session

from backend.services.ip_fencing import IPFencingService
from backend.models.ip_blacklist import IPListType


def get_blacklist(db: Session, limit: int) -> dict:
    service = IPFencingService(db)
    entries = service.get_blacklist(limit)
    return {
        "success": True,
        "data": [e.to_dict() for e in entries],
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_whitelist(db: Session, limit: int) -> dict:
    service = IPFencingService(db)
    entries = service.get_whitelist(limit)
    return {
        "success": True,
        "data": [e.to_dict() for e in entries],
        "timestamp": datetime.utcnow().isoformat(),
    }


def add_to_blacklist(
    db: Session,
    *,
    ip: str,
    reason: str | None = None,
    source: str = "manual",
    duration_hours: int | None = None,
) -> dict:
    service = IPFencingService(db)
    entry = service.add_to_blacklist(ip=ip, reason=reason, source=source, duration_hours=duration_hours)
    return {"success": True, "data": entry.to_dict(), "timestamp": datetime.utcnow().isoformat()}


def add_to_whitelist(db: Session, *, ip: str, reason: str | None = None) -> dict:
    service = IPFencingService(db)
    entry = service.add_to_whitelist(ip=ip, reason=reason)
    return {"success": True, "data": entry.to_dict(), "timestamp": datetime.utcnow().isoformat()}


def remove_from_list(db: Session, ip: str, list_type: str) -> dict:
    service = IPFencingService(db)
    if list_type == "blacklist":
        success = service.remove_from_list(ip, IPListType.BLACKLIST)
    elif list_type == "whitelist":
        success = service.remove_from_list(ip, IPListType.WHITELIST)
    else:
        raise ValueError("Invalid list_type. Use 'blacklist' or 'whitelist'")
    return {
        "success": success,
        "message": f"IP {ip} removed from {list_type}" if success else f"IP {ip} not found in {list_type}",
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_ip_reputation(db: Session, ip: str) -> dict:
    service = IPFencingService(db)
    rep = service.get_ip_reputation(ip)
    if not rep:
        return {"success": False, "message": "IP reputation not found", "timestamp": datetime.utcnow().isoformat()}
    return {"success": True, "data": rep.to_dict(), "timestamp": datetime.utcnow().isoformat()}
