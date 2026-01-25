"""IP Management API endpoints."""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas.ip_management import IPBlacklistRequest, IPWhitelistRequest
from backend.controllers import ip_management as ctrl

router = APIRouter()


@router.get("/blacklist")
async def get_blacklist(limit: int = Query(100, ge=1, le=1000), db: Session = Depends(get_db)):
    return ctrl.get_blacklist(db, limit)


@router.get("/whitelist")
async def get_whitelist(limit: int = Query(100, ge=1, le=1000), db: Session = Depends(get_db)):
    return ctrl.get_whitelist(db, limit)


@router.post("/blacklist")
async def add_to_blacklist(request: IPBlacklistRequest, db: Session = Depends(get_db)):
    return ctrl.add_to_blacklist(
        db, ip=request.ip, reason=request.reason, source=request.source, duration_hours=request.duration_hours
    )


@router.post("/whitelist")
async def add_to_whitelist(request: IPWhitelistRequest, db: Session = Depends(get_db)):
    return ctrl.add_to_whitelist(db, ip=request.ip, reason=request.reason)


@router.delete("/{ip}")
async def remove_from_list(ip: str, list_type: str = Query(..., description="blacklist or whitelist"), db: Session = Depends(get_db)):
    try:
        return ctrl.remove_from_list(db, ip, list_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{ip}/reputation")
async def get_ip_reputation(ip: str, db: Session = Depends(get_db)):
    return ctrl.get_ip_reputation(db, ip)
