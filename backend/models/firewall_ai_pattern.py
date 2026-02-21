"""Firewall-for-AI patterns: prompt-injection and PII (from DB or remote URL)."""

from sqlalchemy import Column, Integer, String, Boolean, DateTime
from backend.database import Base
from datetime import datetime


class FirewallAIPattern(Base):
    """Single pattern: prompt_injection or pii; value is regex or substring."""

    __tablename__ = "firewall_ai_patterns"

    id = Column(Integer, primary_key=True, index=True)
    pattern_type = Column(String(50), nullable=False)   # prompt_injection | pii
    pattern_value = Column(String(1000), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    source = Column(String(50), default="manual")       # manual | remote
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "pattern_type": self.pattern_type,
            "pattern_value": self.pattern_value[:100] + "..." if len(self.pattern_value) > 100 else self.pattern_value,
            "is_active": self.is_active,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
