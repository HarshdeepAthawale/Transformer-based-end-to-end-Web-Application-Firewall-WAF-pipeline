"""Security Rules API endpoints."""
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, Query, Body
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.config import config
from backend.database import get_db
from backend.schemas.security_rules import SecurityRuleRequest
from backend.controllers import security_rules as ctrl
from backend.controllers import managed_rules as managed_ctrl

router = APIRouter()


@router.get("/")
async def get_security_rules(active_only: bool = Query(True), db: Session = Depends(get_db)):
    return ctrl.get_rules(db, active_only)


@router.post("/")
async def create_security_rule(request: SecurityRuleRequest, db: Session = Depends(get_db)):
    return ctrl.create_rule(
        db,
        name=request.name,
        rule_type=request.rule_type,
        pattern=request.pattern,
        applies_to=request.applies_to,
        action=request.action,
        priority=request.priority,
        description=request.description,
        owasp_category=request.owasp_category,
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
    return {"success": True, "data": {"feed_url_configured": feed_configured}, "timestamp": datetime.utcnow().isoformat() + "Z"}


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
):
    """Enable or disable a rule pack."""
    return managed_ctrl.toggle_pack(db, pack_id, enabled)


@router.post("/managed/sync")
async def sync_managed_rules(
    pack_id: str | None = Body(None, embed=True),
    db: Session = Depends(get_db),
):
    """Trigger sync for managed rules (default pack from config or specified pack_id)."""
    return managed_ctrl.sync_managed_rules(db, pack_id=pack_id)


@router.post("/evaluate")
async def evaluate_rules(
    body: dict = Body(...),
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
    result = service.evaluate_managed_rules(method, path, headers, query_params, req_body)
    return result
