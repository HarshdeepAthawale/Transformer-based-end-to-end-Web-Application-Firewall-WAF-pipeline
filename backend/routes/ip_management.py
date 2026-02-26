"""IP Management API endpoints."""
import ipaddress

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas.ip_management import IPBlacklistRequest, IPWhitelistRequest
from backend.controllers import ip_management as ctrl
from backend.auth import get_current_user, require_waf_api_auth
from backend.models.users import User

router = APIRouter()


def _validate_ip(ip: str) -> str:
    """Validate and normalize an IP address or CIDR block."""
    try:
        # Try as single IP
        return str(ipaddress.ip_address(ip))
    except ValueError:
        pass
    try:
        # Try as CIDR network
        return str(ipaddress.ip_network(ip, strict=False))
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid IP address or CIDR: {ip}")


@router.get("/blacklist")
async def get_blacklist(
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return ctrl.get_blacklist(db, limit)


@router.get("/whitelist")
async def get_whitelist(
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return ctrl.get_whitelist(db, limit)


@router.post("/blacklist")
async def add_to_blacklist(
    request: IPBlacklistRequest,
    auth=Depends(require_waf_api_auth),
    db: Session = Depends(get_db),
):
    validated_ip = _validate_ip(request.ip)
    return ctrl.add_to_blacklist(
        db, ip=validated_ip, reason=request.reason, source=request.source, duration_hours=request.duration_hours
    )


@router.post("/whitelist")
async def add_to_whitelist(
    request: IPWhitelistRequest,
    auth=Depends(require_waf_api_auth),
    db: Session = Depends(get_db),
):
    validated_ip = _validate_ip(request.ip)
    return ctrl.add_to_whitelist(db, ip=validated_ip, reason=request.reason)


@router.delete("/{ip}")
async def remove_from_list(
    ip: str,
    list_type: str = Query(..., description="blacklist or whitelist"),
    auth=Depends(require_waf_api_auth),
    db: Session = Depends(get_db),
):
    if list_type not in ("blacklist", "whitelist"):
        raise HTTPException(status_code=422, detail="list_type must be 'blacklist' or 'whitelist'")
    validated_ip = _validate_ip(ip)
    try:
        return ctrl.remove_from_list(db, validated_ip, list_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{ip}/reputation")
async def get_ip_reputation(
    ip: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    validated_ip = _validate_ip(ip)
    return ctrl.get_ip_reputation(db, validated_ip)
