"""
Toxic Combination model: multi-signal security incident detection.

Based on our "Toxic Combinations" blog: detects when small signals
(bot activity, path anomalies, misconfigurations) converge into incidents.
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from backend.database import Base
from backend.lib.datetime_utils import utc_now


class ToxicCombination(Base):
    """A detected toxic combination of security signals."""

    __tablename__ = "toxic_combinations"

    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)

    pattern_name = Column(String(100), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)  # critical, high, medium, low
    description = Column(Text, nullable=True)

    # Affected resources
    affected_host = Column(String(255), nullable=True)
    affected_path = Column(String(500), nullable=True)
    source_ips = Column(Text, nullable=True)  # JSON array of IPs

    # Signal details
    signals = Column(Text, nullable=True)  # JSON array of contributing signals

    # Counters
    event_count = Column(Integer, default=1)
    first_seen = Column(DateTime, nullable=False, default=utc_now)
    last_seen = Column(DateTime, nullable=False, default=utc_now)

    # Status tracking
    status = Column(String(20), default="active", index=True)  # active, investigating, resolved, dismissed
    resolved_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=utc_now, nullable=False)

    def to_dict(self):
        import json
        return {
            "id": self.id,
            "pattern_name": self.pattern_name,
            "severity": self.severity,
            "description": self.description,
            "affected_host": self.affected_host,
            "affected_path": self.affected_path,
            "source_ips": json.loads(self.source_ips) if self.source_ips else [],
            "signals": json.loads(self.signals) if self.signals else [],
            "event_count": self.event_count,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "status": self.status,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
