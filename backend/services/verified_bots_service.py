"""
Verified Bots Service - CRUD and sync from URL.
"""

import re
from datetime import datetime
from typing import List, Optional

import httpx
from sqlalchemy.orm import Session

from backend.models.verified_bots import VerifiedBot
from loguru import logger


class VerifiedBotsService:
    """Service for managing verified bots allowlist."""

    def __init__(self, db: Session):
        self.db = db

    def list_all(self) -> List[VerifiedBot]:
        """List all verified bots."""
        return self.db.query(VerifiedBot).order_by(VerifiedBot.name).all()

    def add(self, name: str, user_agent_pattern: str, source: str = "manual") -> VerifiedBot:
        """Add a verified bot."""
        bot = VerifiedBot(
            name=name,
            user_agent_pattern=user_agent_pattern,
            source=source,
        )
        self.db.add(bot)
        self.db.commit()
        self.db.refresh(bot)
        return bot

    def delete(self, bot_id: int) -> bool:
        """Delete a verified bot by id."""
        bot = self.db.query(VerifiedBot).filter(VerifiedBot.id == bot_id).first()
        if not bot:
            return False
        self.db.delete(bot)
        self.db.commit()
        return True

    def is_verified(self, user_agent: str) -> tuple[bool, Optional[str]]:
        """
        Check if user_agent matches any verified bot.
        Returns (is_verified, matched_name or None).
        """
        if not user_agent:
            return False, None
        bots = self.db.query(VerifiedBot).all()
        for bot in bots:
            try:
                if re.search(bot.user_agent_pattern, user_agent, re.IGNORECASE):
                    return True, bot.name
            except re.error:
                logger.warning(f"Invalid regex in verified bot {bot.id}: {bot.user_agent_pattern}")
        return False, None

    def sync_from_url(self, url: str, headers: Optional[dict] = None) -> int:
        """
        Fetch JSON from URL (array of {name, pattern} or {name, user_agent_pattern}).
        Upsert into verified_bots with source=remote, synced_at=now.
        Returns count of records synced.
        """
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(url, headers=headers or {})
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.error(f"Verified bots sync failed: {e}")
            raise

        if not isinstance(data, list):
            raise ValueError("Expected JSON array")

        now = datetime.utcnow()
        count = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("bot_name")
            pattern = item.get("pattern") or item.get("user_agent_pattern")
            if not name or not pattern:
                continue

            # Upsert: find by name or add new
            existing = self.db.query(VerifiedBot).filter(VerifiedBot.name == name).first()
            if existing:
                existing.user_agent_pattern = pattern
                existing.source = "remote"
                existing.synced_at = now
            else:
                self.db.add(
                    VerifiedBot(
                        name=name,
                        user_agent_pattern=pattern,
                        source="remote",
                        synced_at=now,
                    )
                )
            count += 1

        self.db.commit()
        logger.info(f"Verified bots sync: {count} records from {url}")
        return count
