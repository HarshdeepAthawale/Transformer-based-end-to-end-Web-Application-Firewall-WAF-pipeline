"""
Bot Detection API endpoints
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from src.api.database import get_db
from src.api.services.bot_detection import BotDetectionService
from src.api.models.bot_signatures import BotCategory

router = APIRouter()


class BotSignatureRequest(BaseModel):
    user_agent_pattern: str
    name: str
    category: str  # search_engine, scraper, malicious, monitoring, unknown
    action: str = "block"  # block, allow, challenge, monitor
    is_whitelisted: bool = False


@router.get("/signatures")
async def get_bot_signatures(
    active_only: bool = Query(True),
    db: Session = Depends(get_db)
):
    """Get bot signatures"""
    service = BotDetectionService(db)
    signatures = service.get_signatures(active_only)
    
    return {
        "success": True,
        "data": [sig.to_dict() for sig in signatures],
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/signatures")
async def add_bot_signature(
    request: BotSignatureRequest,
    db: Session = Depends(get_db)
):
    """Add bot signature"""
    service = BotDetectionService(db)
    
    category = BotCategory[request.category.upper()] if hasattr(BotCategory, request.category.upper()) else BotCategory.UNKNOWN
    
    signature = service.add_signature(
        user_agent_pattern=request.user_agent_pattern,
        name=request.name,
        category=category,
        action=request.action,
        is_whitelisted=request.is_whitelisted
    )
    
    return {
        "success": True,
        "data": signature.to_dict(),
        "timestamp": datetime.utcnow().isoformat()
    }
