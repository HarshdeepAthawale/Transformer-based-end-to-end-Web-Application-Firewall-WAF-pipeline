"""
Rate limit configuration - API-driven config for gateway (Feature 9).
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean
from backend.database import Base
from datetime import datetime


class RateLimitConfig(Base):
    """Rate limit config: path_prefix, requests_per_minute, window_seconds, burst, optional zone_id."""

    __tablename__ = "rate_limit_config"

    id = Column(Integer, primary_key=True, index=True)
    path_prefix = Column(String(500), nullable=False, index=True)  # e.g. /api/
    requests_per_minute = Column(Integer, nullable=False)
    window_seconds = Column(Integer, nullable=False, default=60)
    burst = Column(Integer, nullable=True)  # optional burst allowance
    zone_id = Column(String(100), nullable=True, index=True, default="default")
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "path_prefix": self.path_prefix,
            "requests_per_minute": self.requests_per_minute,
            "window_seconds": self.window_seconds,
            "burst": self.burst,
            "zone_id": self.zone_id,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
