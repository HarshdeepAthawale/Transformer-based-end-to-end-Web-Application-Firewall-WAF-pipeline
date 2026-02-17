"""
Bot score bands - map score ranges to actions (allow, challenge, block).
"""

from sqlalchemy import Column, Integer, String, UniqueConstraint
from backend.database import Base


class BotScoreBand(Base):
    """Score bands: min_score, max_score -> action. Lower priority = evaluated first."""

    __tablename__ = "bot_score_bands"
    __table_args__ = (UniqueConstraint("min_score", "max_score", name="uq_bot_score_bands_range"),)

    id = Column(Integer, primary_key=True, index=True)
    min_score = Column(Integer, nullable=False)
    max_score = Column(Integer, nullable=False)
    action = Column(String(20), nullable=False)  # allow | challenge | block
    priority = Column(Integer, nullable=False, default=0)

    def to_dict(self):
        return {
            "id": self.id,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "action": self.action,
            "priority": self.priority,
        }
