"""Bot Management API - score, signatures, verified bots, score bands (Feature 9)."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy import inspect
from sqlalchemy.orm import Session
from loguru import logger

from backend.database import get_db, engine, Base
from backend.auth import require_waf_api_auth, get_current_tenant
from backend.services.bot_detection import BotDetectionService
from backend.services.verified_bots_service import VerifiedBotsService
from backend.services.bot_score_bands_service import BotScoreBandsService
from backend.models.bot_signatures import BotCategory
from backend.schemas.bot_detection import BotSignatureRequest

router = APIRouter()

_bot_tables_ensured = False


def _ensure_bot_tables():
    """One-time lazy check: create verified_bots and bot_score_bands if missing (e.g. old DB)."""
    global _bot_tables_ensured
    if _bot_tables_ensured:
        return
    try:
        insp = inspect(engine)
        if not insp.has_table("verified_bots") or not insp.has_table("bot_score_bands"):
            import backend.models  # noqa: F401
            Base.metadata.create_all(bind=engine)
            logger.info("Bot tables created (lazy migration)")
    except Exception as e:
        logger.warning(f"Bot tables check failed: {e}")
    _bot_tables_ensured = True


def get_db_with_bot_tables():
    """Dependency: ensure bot tables exist, then yield db session."""
    _ensure_bot_tables()
    yield from get_db()


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_agent: str = ""
    ip: str = ""
    headers: dict = {}


class ScoreResponse(BaseModel):
    bot_score: int
    action: str
    is_verified_bot: bool
    matched_signature: str | None


class VerifiedBotCreate(BaseModel):
    name: str
    user_agent_pattern: str


class ScoreBandItem(BaseModel):
    min_score: int
    max_score: int
    action: str


class ScoreBandsUpdate(BaseModel):
    bands: list[ScoreBandItem]


@router.post("/score", response_model=ScoreResponse)
async def post_bot_score(req: ScoreRequest, db: Session = Depends(get_db_with_bot_tables)):
    """
    Gateway calls this to get bot score and action. No mocks.
    """
    service = BotDetectionService(db)
    result = service.detect_bot(
        user_agent=req.user_agent or "",
        ip=req.ip or "",
        headers=req.headers or {},
    )
    return ScoreResponse(
        bot_score=result["bot_score"],
        action=result["action"],
        is_verified_bot=result.get("is_verified_bot", False),
        matched_signature=result.get("matched_signature"),
    )


@router.get("/verified")
async def get_verified_bots(db: Session = Depends(get_db_with_bot_tables)):
    """List all verified bots."""
    svc = VerifiedBotsService(db)
    bots = svc.list_all()
    return {"success": True, "data": [b.to_dict() for b in bots]}


@router.post("/verified")
async def add_verified_bot(
    body: VerifiedBotCreate,
    db: Session = Depends(get_db_with_bot_tables),
    _auth=Depends(require_waf_api_auth),
):
    """Add a verified bot. Requires auth."""
    svc = VerifiedBotsService(db)
    bot = svc.add(body.name, body.user_agent_pattern, source="manual")
    return {"success": True, "data": bot.to_dict()}


@router.delete("/verified/{bot_id}")
async def delete_verified_bot(
    bot_id: int,
    db: Session = Depends(get_db_with_bot_tables),
    _auth=Depends(require_waf_api_auth),
):
    """Delete a verified bot. Requires auth."""
    svc = VerifiedBotsService(db)
    ok = svc.delete(bot_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Verified bot not found")
    return {"success": True}


@router.post("/verified/sync")
async def sync_verified_bots(db: Session = Depends(get_db_with_bot_tables)):
    """Trigger sync from BOT_VERIFIED_SYNC_URL. Returns 503 if URL not configured."""
    from backend.config import config

    url = getattr(config, "BOT_VERIFIED_SYNC_URL", "") or ""
    if not url or not url.strip():
        raise HTTPException(
            status_code=503,
            detail="BOT_VERIFIED_SYNC_URL not configured",
        )
    headers = {}
    sync_header = getattr(config, "BOT_VERIFIED_SYNC_HEADER", None)
    if sync_header:
        # Format: "Header-Name: value"
        if ":" in sync_header:
            k, v = sync_header.split(":", 1)
            headers[k.strip()] = v.strip()
    svc = VerifiedBotsService(db)
    count = svc.sync_from_url(url, headers=headers if headers else None)
    return {"success": True, "synced": count}


@router.get("/score-bands")
async def get_score_bands(db: Session = Depends(get_db_with_bot_tables)):
    """List score bands."""
    svc = BotScoreBandsService(db)
    bands = svc.get_bands()
    return {"success": True, "data": [b.to_dict() for b in bands]}


@router.put("/score-bands")
async def update_score_bands(
    body: ScoreBandsUpdate,
    db: Session = Depends(get_db_with_bot_tables),
    _auth=Depends(require_waf_api_auth),
):
    """Replace score bands with the given array. Requires auth."""
    svc = BotScoreBandsService(db)
    bands = svc.update_bands([b.model_dump() for b in body.bands])
    return {"success": True, "data": [b.to_dict() for b in bands]}


# --- Signatures CRUD (Feature 9) ---

class BotSignatureUpdate(BaseModel):
    user_agent_pattern: str | None = None
    name: str | None = None
    category: str | None = None
    action: str | None = None
    is_whitelisted: bool | None = None
    is_active: bool | None = None


@router.get("/stats")
async def get_bot_stats(
    range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db_with_bot_tables),
):
    """Per-org bot score distribution and stats. Requires auth."""
    svc = BotDetectionService(db)
    stats = svc.get_bot_stats(org_id, db, range)
    return {"success": True, "data": stats}


@router.get("/signatures")
async def list_bot_signatures(
    active_only: bool = Query(True, description="Filter to active only"),
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db_with_bot_tables),
):
    """List bot signatures."""
    svc = BotDetectionService(db)
    sigs = svc.get_signatures(org_id, active_only=active_only)
    return {"success": True, "data": [s.to_dict() for s in sigs]}


@router.post("/signatures")
async def create_bot_signature(
    request: BotSignatureRequest,
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db_with_bot_tables),
    _auth=Depends(require_waf_api_auth),
):
    """Create a bot signature. Requires auth."""
    svc = BotDetectionService(db)
    cat = BotCategory[request.category.upper()] if hasattr(BotCategory, request.category.upper()) else BotCategory.UNKNOWN
    sig = svc.add_signature(
        org_id,
        user_agent_pattern=request.user_agent_pattern,
        name=request.name,
        category=cat,
        action=request.action,
        is_whitelisted=request.is_whitelisted,
    )
    return {"success": True, "data": sig.to_dict()}


@router.get("/signatures/{sig_id}")
async def get_bot_signature(sig_id: int, db: Session = Depends(get_db_with_bot_tables)):
    """Get one bot signature by id."""
    svc = BotDetectionService(db)
    sig = svc.get_signature_by_id(sig_id)
    if not sig:
        raise HTTPException(status_code=404, detail="Signature not found")
    return {"success": True, "data": sig.to_dict()}


@router.put("/signatures/{sig_id}")
async def update_bot_signature(
    sig_id: int,
    body: BotSignatureUpdate,
    db: Session = Depends(get_db_with_bot_tables),
    _auth=Depends(require_waf_api_auth),
):
    """Update a bot signature. Requires auth."""
    svc = BotDetectionService(db)
    kwargs = body.model_dump(exclude_unset=True)
    if "category" in kwargs and kwargs["category"] is not None:
        c = kwargs["category"].upper()
        kwargs["category"] = BotCategory[c] if hasattr(BotCategory, c) else None
    sig = svc.update_signature(sig_id, **kwargs)
    if not sig:
        raise HTTPException(status_code=404, detail="Signature not found")
    return {"success": True, "data": sig.to_dict()}


@router.delete("/signatures/{sig_id}")
async def delete_bot_signature(
    sig_id: int,
    db: Session = Depends(get_db_with_bot_tables),
    _auth=Depends(require_waf_api_auth),
):
    """Delete a bot signature. Requires auth."""
    svc = BotDetectionService(db)
    ok = svc.delete_signature(sig_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Signature not found")
    return {"success": True}
