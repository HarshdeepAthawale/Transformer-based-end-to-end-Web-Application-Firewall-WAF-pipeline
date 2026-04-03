"""Rate limit configuration API (Feature 9)."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.auth import require_waf_api_auth, get_current_tenant
from backend.services.rate_limit_config_service import RateLimitConfigService

router = APIRouter()


class RateLimitConfigCreate(BaseModel):
    path_prefix: str
    requests_per_minute: int
    window_seconds: int = 60
    burst: int | None = None
    zone_id: str | None = "default"
    is_active: bool = True


class RateLimitConfigUpdate(BaseModel):
    path_prefix: str | None = None
    requests_per_minute: int | None = None
    window_seconds: int | None = None
    burst: int | None = None
    zone_id: str | None = None
    is_active: bool | None = None


@router.get("/")
async def list_rate_limits(
    zone_id: str | None = Query(None, description="Filter by zone_id"),
    active_only: bool = Query(True, description="Filter to active only"),
    db: Session = Depends(get_db),
):
    """List rate limit configs (e.g. for gateway to fetch and apply)."""
    svc = RateLimitConfigService(db)
    configs = svc.list_all(zone_id=zone_id, active_only=active_only)
    return {"success": True, "data": [c.to_dict() for c in configs]}


@router.post("/")
async def create_rate_limit(
    body: RateLimitConfigCreate,
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _auth=Depends(require_waf_api_auth),
):
    """Create a rate limit config. Requires auth."""
    svc = RateLimitConfigService(db)
    r = svc.create(
        org_id=org_id,
        path_prefix=body.path_prefix,
        requests_per_minute=body.requests_per_minute,
        window_seconds=body.window_seconds,
        burst=body.burst,
        zone_id=body.zone_id,
        is_active=body.is_active,
    )
    return {"success": True, "data": r.to_dict()}


@router.get("/{config_id}")
async def get_rate_limit(config_id: int, db: Session = Depends(get_db)):
    """Get one rate limit config by id."""
    svc = RateLimitConfigService(db)
    r = svc.get_by_id(config_id)
    if not r:
        raise HTTPException(status_code=404, detail="Rate limit config not found")
    return {"success": True, "data": r.to_dict()}


@router.put("/{config_id}")
async def update_rate_limit(
    config_id: int,
    body: RateLimitConfigUpdate,
    db: Session = Depends(get_db),
    _auth=Depends(require_waf_api_auth),
):
    """Update a rate limit config. Requires auth."""
    svc = RateLimitConfigService(db)
    kwargs = body.model_dump(exclude_unset=True)
    r = svc.update(config_id, **kwargs)
    if not r:
        raise HTTPException(status_code=404, detail="Rate limit config not found")
    return {"success": True, "data": r.to_dict()}


@router.delete("/{config_id}", status_code=204)
async def delete_rate_limit(
    config_id: int,
    db: Session = Depends(get_db),
    _auth=Depends(require_waf_api_auth),
):
    """Delete a rate limit config. Requires auth."""
    from fastapi.responses import Response

    svc = RateLimitConfigService(db)
    ok = svc.delete(config_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Rate limit config not found")
    return Response(status_code=204)


@router.get("/effective")
async def get_effective_limit(
    path: str = Query(..., description="The API path to check"),
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Get effective rate limit for org on a given path (returns DB config or default)."""
    svc = RateLimitConfigService(db)
    config = svc.get_by_org_and_path(org_id, path)

    if config:
        return {
            "success": True,
            "data": {
                "path": path,
                "requests_per_minute": config.requests_per_minute,
                "window_seconds": config.window_seconds,
                "from_db": True,
            },
        }
    else:
        return {
            "success": True,
            "data": {
                "path": path,
                "requests_per_minute": 300,
                "window_seconds": 60,
                "from_db": False,
            },
        }
