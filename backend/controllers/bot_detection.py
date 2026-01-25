"""Bot detection controller."""
from datetime import datetime
from sqlalchemy.orm import Session

from backend.services.bot_detection import BotDetectionService
from backend.models.bot_signatures import BotCategory


def get_signatures(db: Session, active_only: bool) -> dict:
    service = BotDetectionService(db)
    sigs = service.get_signatures(active_only)
    return {
        "success": True,
        "data": [s.to_dict() for s in sigs],
        "timestamp": datetime.utcnow().isoformat(),
    }


def add_signature(
    db: Session,
    *,
    user_agent_pattern: str,
    name: str,
    category: str,
    action: str = "block",
    is_whitelisted: bool = False,
) -> dict:
    service = BotDetectionService(db)
    cat = BotCategory[category.upper()] if hasattr(BotCategory, category.upper()) else BotCategory.UNKNOWN
    sig = service.add_signature(
        user_agent_pattern=user_agent_pattern,
        name=name,
        category=cat,
        action=action,
        is_whitelisted=is_whitelisted,
    )
    return {"success": True, "data": sig.to_dict(), "timestamp": datetime.utcnow().isoformat()}
