"""
Bot scoring for gateway - calls backend POST /api/bot/score and enforces action.
"""

from typing import Any

import httpx
from loguru import logger

from gateway.config import gateway_config


async def get_bot_score(
    user_agent: str,
    ip: str,
    headers: dict,
) -> dict[str, Any] | None:
    """
    Call backend POST /api/bot/score. Returns {bot_score, action, is_verified_bot} or None on failure.
    """
    url = (gateway_config.BOT_BACKEND_URL or "").rstrip("/")
    if not url:
        return None

    score_url = f"{url}/api/bot/score"
    timeout = gateway_config.BOT_TIMEOUT_SECONDS

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                score_url,
                json={
                    "user_agent": user_agent or "",
                    "ip": ip or "",
                    "headers": dict(headers) if headers else {},
                },
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            bot_score = data.get("bot_score")
            action = data.get("action")
            if bot_score is None or action is None:
                logger.warning("Bot score response missing bot_score or action")
                return None
            return {
                "bot_score": int(bot_score),
                "action": str(action),
                "is_verified_bot": bool(data.get("is_verified_bot", False)),
                "matched_signature": data.get("matched_signature"),
            }
    except Exception as e:
        logger.debug(f"Bot score request failed: {e}")
        return None
