"""
Bot Score Bands Service - get action for a given score.
"""

from typing import List

from sqlalchemy.orm import Session

from backend.models.bot_score_bands import BotScoreBand


class BotScoreBandsService:
    """Service for score band configuration."""

    def __init__(self, db: Session):
        self.db = db

    def get_bands(self) -> List[BotScoreBand]:
        """Get all bands ordered by priority."""
        return self.db.query(BotScoreBand).order_by(BotScoreBand.priority).all()

    def get_action_for_score(self, score: int) -> str:
        """
        Return action (allow|challenge|block) for the given score.
        Score must be 1-99. Default to allow if no band matches.
        """
        score = max(1, min(99, score))
        bands = self.get_bands()
        for band in bands:
            if band.min_score <= score <= band.max_score:
                return band.action
        return "allow"

    def update_bands(self, bands_data: List[dict]) -> List[BotScoreBand]:
        """
        Replace all bands with the given array of {min_score, max_score, action}.
        Priority is assigned by order (0, 1, 2, ...).
        """
        self.db.query(BotScoreBand).delete()
        result = []
        for i, b in enumerate(bands_data):
            band = BotScoreBand(
                min_score=int(b["min_score"]),
                max_score=int(b["max_score"]),
                action=b["action"],
                priority=i,
            )
            self.db.add(band)
            result.append(band)
        self.db.commit()
        for b in result:
            self.db.refresh(b)
        return result
