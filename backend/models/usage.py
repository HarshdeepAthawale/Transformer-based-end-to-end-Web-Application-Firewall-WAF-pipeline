"""
Usage tracking model - per-org monthly request counts.
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, UniqueConstraint
from backend.database import Base
from backend.lib.datetime_utils import utc_now


class UsageRecord(Base):
    """Monthly usage tracking per organization."""
    __tablename__ = "usage_records"
    __table_args__ = (
        UniqueConstraint("org_id", "month", name="uq_usage_org_month"),
    )

    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    month = Column(String(7), nullable=False, index=True)  # YYYY-MM format
    requests_count = Column(Integer, nullable=False, default=0)
    last_updated = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "org_id": self.org_id,
            "month": self.month,
            "requests_count": self.requests_count,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }
