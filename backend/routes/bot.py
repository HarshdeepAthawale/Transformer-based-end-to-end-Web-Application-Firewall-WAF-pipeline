"""Bot Management API - score endpoint, verified bots, score bands."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import inspect
from sqlalchemy.orm import Session
from loguru import logger

from backend.database import get_db, engine, Base
from backend.services.bot_detection import BotDetectionService
from backend.services.verified_bots_service import VerifiedBotsService
from backend.services.bot_score_bands_service import BotScoreBandsService

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
async def add_verified_bot(body: VerifiedBotCreate, db: Session = Depends(get_db_with_bot_tables)):
    """Add a verified bot."""
    svc = VerifiedBotsService(db)
    bot = svc.add(body.name, body.user_agent_pattern, source="manual")
    return {"success": True, "data": bot.to_dict()}


@router.delete("/verified/{bot_id}")
async def delete_verified_bot(bot_id: int, db: Session = Depends(get_db_with_bot_tables)):
    """Delete a verified bot."""
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
async def update_score_bands(body: ScoreBandsUpdate, db: Session = Depends(get_db_with_bot_tables)):
    """Replace score bands with the given array."""
    svc = BotScoreBandsService(db)
    bands = svc.update_bands([b.model_dump() for b in body.bands])
    return {"success": True, "data": [b.to_dict() for b in bands]}
