"""
Security events: rate limit hits and DDoS blocks.
"""

from sqlalchemy import ForeignKey, Column, Integer, String, DateTime, Text
from backend.database import Base
from datetime import datetime, timezone
from backend.lib.datetime_utils import utc_now


class SecurityEvent(Base):
    """Security events (rate limit, DDoS) from gateway."""

    __tablename__ = "security_events"

    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime, default=utc_now, index=True, nullable=False)

    event_type = Column(String(50), nullable=False, index=True)
    ip = Column(String(45), nullable=False, index=True)
    method = Column(String(10), nullable=True)
    path = Column(String(500), nullable=True)

    details = Column(Text, nullable=True)
    attack_score = Column(Integer, nullable=True, index=True)
    block_duration_seconds = Column(Integer, nullable=True)
    bot_score = Column(Integer, nullable=True, index=True)

    # WAF Attack Score sub-categories (industry-standard, 1-99: lower = more malicious)
    waf_sqli_score = Column(Integer, nullable=True)
    waf_xss_score = Column(Integer, nullable=True)
    waf_rce_score = Column(Integer, nullable=True)
    attack_class = Column(String(30), nullable=True, index=True)  # sqli, xss, rce, path_traversal, etc.

    def _serialize_timestamp(self, value):
        """Normalize timestamp for JSON: handle None, string (e.g. from SQLite), or datetime."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, datetime):
            return value.replace(tzinfo=timezone.utc).isoformat()
        return str(value)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self._serialize_timestamp(self.timestamp),
            "event_type": self.event_type,
            "ip": self.ip,
            "method": self.method,
            "path": self.path,
            "details": self.details,
            "attack_score": self.attack_score,
            "block_duration_seconds": self.block_duration_seconds,
            "bot_score": self.bot_score,
            "waf_sqli_score": self.waf_sqli_score,
            "waf_xss_score": self.waf_xss_score,
            "waf_rce_score": self.waf_rce_score,
            "attack_class": self.attack_class,
        }
