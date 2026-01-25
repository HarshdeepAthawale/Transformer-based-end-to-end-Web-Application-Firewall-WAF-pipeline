"""Bot Detection API endpoints."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas.bot_detection import BotSignatureRequest
from backend.controllers import bot_detection as ctrl

router = APIRouter()


@router.get("/signatures")
async def get_bot_signatures(active_only: bool = Query(True), db: Session = Depends(get_db)):
    return ctrl.get_signatures(db, active_only)


@router.post("/signatures")
async def add_bot_signature(request: BotSignatureRequest, db: Session = Depends(get_db)):
    return ctrl.add_signature(
        db,
        user_agent_pattern=request.user_agent_pattern,
        name=request.name,
        category=request.category,
        action=request.action,
        is_whitelisted=request.is_whitelisted,
    )
