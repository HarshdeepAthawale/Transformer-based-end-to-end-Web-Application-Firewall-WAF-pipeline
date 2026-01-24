"""
IP Management API endpoints
"""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from src.api.database import get_db
from src.api.services.ip_fencing import IPFencingService
from src.api.models.ip_blacklist import IPListType, IPBlockType

router = APIRouter()


class IPBlacklistRequest(BaseModel):
    ip: str
    reason: Optional[str] = None
    duration_hours: Optional[int] = None
    source: str = "manual"


class IPWhitelistRequest(BaseModel):
    ip: str
    reason: Optional[str] = None


@router.get("/blacklist")
async def get_blacklist(
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get blacklisted IPs"""
    service = IPFencingService(db)
    blacklist = service.get_blacklist(limit)
    
    return {
        "success": True,
        "data": [entry.to_dict() for entry in blacklist],
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/whitelist")
async def get_whitelist(
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """Get whitelisted IPs"""
    service = IPFencingService(db)
    whitelist = service.get_whitelist(limit)
    
    return {
        "success": True,
        "data": [entry.to_dict() for entry in whitelist],
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/blacklist")
async def add_to_blacklist(
    request: IPBlacklistRequest,
    db: Session = Depends(get_db)
):
    """Add IP to blacklist"""
    service = IPFencingService(db)
    entry = service.add_to_blacklist(
        ip=request.ip,
        reason=request.reason,
        source=request.source,
        duration_hours=request.duration_hours
    )
    
    return {
        "success": True,
        "data": entry.to_dict(),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/whitelist")
async def add_to_whitelist(
    request: IPWhitelistRequest,
    db: Session = Depends(get_db)
):
    """Add IP to whitelist"""
    service = IPFencingService(db)
    entry = service.add_to_whitelist(
        ip=request.ip,
        reason=request.reason
    )
    
    return {
        "success": True,
        "data": entry.to_dict(),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.delete("/{ip}")
async def remove_from_list(
    ip: str,
    list_type: str = Query(..., description="blacklist or whitelist"),
    db: Session = Depends(get_db)
):
    """Remove IP from blacklist or whitelist"""
    service = IPFencingService(db)
    
    if list_type == "blacklist":
        success = service.remove_from_list(ip, IPListType.BLACKLIST)
    elif list_type == "whitelist":
        success = service.remove_from_list(ip, IPListType.WHITELIST)
    else:
        raise HTTPException(status_code=400, detail="Invalid list_type. Use 'blacklist' or 'whitelist'")
    
    return {
        "success": success,
        "message": f"IP {ip} removed from {list_type}" if success else f"IP {ip} not found in {list_type}",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/{ip}/reputation")
async def get_ip_reputation(
    ip: str,
    db: Session = Depends(get_db)
):
    """Get IP reputation"""
    service = IPFencingService(db)
    reputation = service.get_ip_reputation(ip)
    
    if not reputation:
        return {
            "success": False,
            "message": "IP reputation not found",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    return {
        "success": True,
        "data": reputation.to_dict(),
        "timestamp": datetime.utcnow().isoformat()
    }
