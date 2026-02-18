"""
Rule pack (managed ruleset) model for OWASP CRS and other feed-based rule packs.
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from backend.database import Base
from datetime import datetime


class RulePack(Base):
    """Rule pack table: one row per managed ruleset (e.g. OWASP CRS)."""
    __tablename__ = "rule_packs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    pack_id = Column(String(100), unique=True, nullable=False, index=True)  # e.g. owasp_crs
    source_url = Column(String(2000), nullable=True)
    version = Column(String(100), nullable=True)
    enabled = Column(Boolean, default=True, nullable=False, index=True)
    last_synced_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "pack_id": self.pack_id,
            "source_url": self.source_url,
            "version": self.version,
            "enabled": self.enabled,
            "last_synced_at": self.last_synced_at.isoformat() if self.last_synced_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
