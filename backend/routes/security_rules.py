"""Security Rules API endpoints (Feature 9: API-first WAF rules)."""
from backend.lib.datetime_utils import utc_now
from pathlib import Path

from fastapi import APIRouter, Depends, Query, Body, HTTPException
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from backend.config import config
from backend.database import get_db
from backend.auth import require_waf_api_auth, get_current_tenant
from backend.schemas.security_rules import SecurityRuleRequest, SecurityRuleUpdate
from backend.controllers import security_rules as ctrl
from backend.controllers import managed_rules as managed_ctrl

router = APIRouter()


@router.get("/")
async def list_security_rules(
    active_only: bool = Query(True, description="Filter to active rules only"),
    pack_id: int | None = Query(None, description="Filter by rule pack id"),
    limit: int | None = Query(None, ge=1, le=500),
    offset: int = Query(0, ge=0),
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """List rules. Response: { data: rule_dict[], total: number }."""
    return ctrl.get_rules(db, org_id, active_only=active_only, pack_id=pack_id, limit=limit, offset=offset)


@router.post("/")
async def create_security_rule(
    request: SecurityRuleRequest,
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _auth=Depends(require_waf_api_auth),
):
    """Create a rule. Requires auth (JWT or API key)."""
    return ctrl.create_rule(
        db,
        org_id,
        name=request.name,
        rule_type=request.rule_type,
        pattern=request.pattern,
        applies_to=request.applies_to,
        action=request.action,
        priority=request.priority,
        description=request.description,
        owasp_category=request.owasp_category,
        match_conditions=request.match_conditions,
        is_active=request.is_active if request.is_active is not None else True,
    )


@router.get("/owasp")
async def get_owasp_rules(db: Session = Depends(get_db)):
    return ctrl.get_owasp_rules(db)


# --- Managed rules (rule packs) ---

# Fixed paths first so they are not matched as pack_id
# Project root data/ (local); backend/data/ (Docker image copy)
_resolve_dir = Path(__file__).resolve().parent.parent
FEED_FILE = _resolve_dir.parent / "data" / "managed-rules-feed.json"
FEED_FILE_FALLBACK = _resolve_dir / "data" / "managed-rules-feed.json"


@router.get("/managed/config")
async def get_managed_rules_config():
    """Return whether feed URL is configured (no URL exposed)."""
    feed_configured = bool((config.MANAGED_RULES_FEED_URL or "").strip())
    return {"success": True, "data": {"feed_url_configured": feed_configured}, "timestamp": utc_now().isoformat() + "Z"}


@router.get("/managed/feed")
async def get_managed_rules_feed():
    """Serve built-in sample rule feed for sync (JSON format)."""
    from fastapi import HTTPException
    path = FEED_FILE if FEED_FILE.exists() else (FEED_FILE_FALLBACK if FEED_FILE_FALLBACK.exists() else None)
    if not path:
        raise HTTPException(status_code=404, detail="Managed rules feed file not found")
    return FileResponse(path=str(path), media_type="application/json")


@router.get("/managed")
async def get_managed_rules(
    enabled_only: bool = Query(True, description="Return only enabled packs"),
    db: Session = Depends(get_db),
):
    """List enabled rule packs with their rules (for gateway and frontend)."""
    return managed_ctrl.get_managed_rules_for_gateway(db, enabled_only=enabled_only)


@router.get("/managed/packs")
async def get_managed_packs(
    enabled_only: bool = Query(False, description="Filter to enabled only"),
    db: Session = Depends(get_db),
):
    """List rule packs with metadata (version, last_synced_at, rule_count) for dashboard."""
    return managed_ctrl.get_managed_packs(db, enabled_only=enabled_only)


@router.patch("/managed/packs/{pack_id}")
async def toggle_rule_pack(
    pack_id: str,
    enabled: bool = Body(..., embed=True),
    db: Session = Depends(get_db),
    _auth=Depends(require_waf_api_auth),
):
    """Enable or disable a rule pack."""
    return managed_ctrl.toggle_pack(db, pack_id, enabled)


@router.post("/managed/sync")
async def sync_managed_rules(
    pack_id: str | None = Body(None, embed=True),
    db: Session = Depends(get_db),
    _auth=Depends(require_waf_api_auth),
):
    """Trigger sync for managed rules (default pack from config or specified pack_id)."""
    return managed_ctrl.sync_managed_rules(db, pack_id=pack_id)


@router.post("/evaluate")
async def evaluate_rules(
    body: dict = Body(...),
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Evaluate request snapshot against managed rules (for gateway or sidecar). Returns first match."""
    from backend.services.rules_service import RulesService
    method = body.get("method", "GET")
    path = body.get("path", "")
    headers = body.get("headers") or {}
    query_params = body.get("query_params") or body.get("query") or {}
    req_body = body.get("body") or body.get("raw_body") or ""
    if isinstance(req_body, bytes):
        req_body = req_body.decode("utf-8", errors="replace")
    service = RulesService(db)
    result = service.evaluate_managed_rules(org_id, method, path, headers, query_params, req_body)
    return result


# By-id routes last so /owasp, /managed/* are not matched as rule_id
@router.get("/{rule_id}")
async def get_security_rule(rule_id: int, org_id: int = Depends(get_current_tenant), db: Session = Depends(get_db)):
    """Get one rule by id."""
    out = ctrl.get_rule_by_id(db, org_id, rule_id)
    if out is None:
        raise HTTPException(status_code=404, detail="Rule not found")
    return out


@router.put("/{rule_id}")
async def update_security_rule(
    rule_id: int,
    request: SecurityRuleUpdate,
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _auth=Depends(require_waf_api_auth),
):
    """Update a rule (partial). System rules cannot be updated. Requires auth."""
    body = request.model_dump(exclude_unset=True)
    out = ctrl.update_rule(db, org_id, rule_id, **body)
    if out is None:
        raise HTTPException(status_code=404, detail="Rule not found or system rule cannot be updated")
    return out


@router.delete("/{rule_id}", status_code=204)
async def delete_security_rule(
    rule_id: int,
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
    _auth=Depends(require_waf_api_auth),
):
    """Soft-deactivate a rule. Returns 204. System rules cannot be deleted. Requires auth."""
    ok = ctrl.delete_rule(db, org_id, rule_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Rule not found or system rule cannot be deleted")
    return Response(status_code=204)
