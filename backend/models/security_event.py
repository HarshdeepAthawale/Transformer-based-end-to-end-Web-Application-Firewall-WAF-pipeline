"""
Security events: rate limit hits and DDoS blocks.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text
from backend.database import Base
from datetime import datetime, timezone


class SecurityEvent(Base):
    """Security events (rate limit, DDoS) from gateway."""

    __tablename__ = "security_events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)

    event_type = Column(String(50), nullable=False, index=True)
    ip = Column(String(45), nullable=False, index=True)
    method = Column(String(10), nullable=True)
    path = Column(String(500), nullable=True)

    details = Column(Text, nullable=True)
    attack_score = Column(Integer, nullable=True, index=True)
    block_duration_seconds = Column(Integer, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.replace(tzinfo=timezone.utc).isoformat() if self.timestamp else None,
            "event_type": self.event_type,
            "ip": self.ip,
            "method": self.method,
            "path": self.path,
            "details": self.details,
            "attack_score": self.attack_score,
            "block_duration_seconds": self.block_duration_seconds,
        }
