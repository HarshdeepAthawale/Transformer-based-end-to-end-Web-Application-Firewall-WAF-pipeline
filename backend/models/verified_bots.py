"""
Verified bots database model.
Separate from bot_signatures for clear separation and bulk sync from URL.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text
from backend.database import Base
from datetime import datetime


class VerifiedBot(Base):
    """Verified bots table - allowlist of known-good bots."""

    __tablename__ = "verified_bots"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    user_agent_pattern = Column(String(500), nullable=False, index=True)
    source = Column(String(50), nullable=False, default="manual")  # manual | remote
    synced_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "user_agent_pattern": self.user_agent_pattern,
            "source": self.source,
            "synced_at": self.synced_at.isoformat() if self.synced_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
