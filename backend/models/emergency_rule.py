"""
Emergency Rule model: rapid-deploy rules for zero-day threats.

These rules are checked BEFORE ML inference for fast blocking of
known exploit patterns (like our Emergency Rules).
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from backend.database import Base
from backend.lib.datetime_utils import utc_now


class EmergencyRule(Base):
    """A rapid-deploy security rule for zero-day threat response."""

    __tablename__ = "emergency_rules"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # CVE references (JSON array)
    cves = Column(Text, nullable=True)  # e.g. '["CVE-2023-46805", "CVE-2024-21887"]'

    # Pattern matching rules (JSON array of pattern objects)
    # Each pattern: {"field": "path"|"query"|"body"|"header", "op": "contains"|"regex"|"equals", "value": "..."}
    patterns = Column(Text, nullable=False)

    # Action to take
    action = Column(String(20), nullable=False, default="block")  # block, challenge, log

    # Status
    enabled = Column(Boolean, default=True, index=True)
    hit_count = Column(Integer, default=0)

    # Metadata
    severity = Column(String(20), default="critical")  # critical, high, medium
    source = Column(String(100), nullable=True)  # "manual", "threat-intel", "auto"

    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)
    expires_at = Column(DateTime, nullable=True)

    def to_dict(self):
        import json
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "cves": json.loads(self.cves) if self.cves else [],
            "patterns": json.loads(self.patterns) if self.patterns else [],
            "action": self.action,
            "enabled": self.enabled,
            "hit_count": self.hit_count,
            "severity": self.severity,
            "source": self.source,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }
