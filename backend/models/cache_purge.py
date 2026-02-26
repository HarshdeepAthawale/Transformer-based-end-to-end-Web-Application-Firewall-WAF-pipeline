"""
Cache purge log: tracks all cache purge operations.
"""

from sqlalchemy import Column, Integer, String, DateTime
from backend.database import Base
from backend.lib.datetime_utils import utc_now


class CachePurgeLog(Base):
    """Log of cache purge operations."""

    __tablename__ = "cache_purge_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=utc_now, nullable=False, index=True)
    purge_type = Column(String(20), nullable=False)  # url, tag, prefix, wildcard, all
    purge_value = Column(String(500), nullable=True)
    keys_purged = Column(Integer, default=0)
    requested_by = Column(String(100), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "purge_type": self.purge_type,
            "purge_value": self.purge_value,
            "keys_purged": self.keys_purged,
            "requested_by": self.requested_by,
        }
