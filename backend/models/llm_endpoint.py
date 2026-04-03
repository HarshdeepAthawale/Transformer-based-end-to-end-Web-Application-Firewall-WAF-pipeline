"""LLM endpoints: path patterns and methods for Firewall-for-AI."""

from sqlalchemy import ForeignKey, Column, Integer, String, Boolean, DateTime
from backend.database import Base
from backend.lib.datetime_utils import utc_now


class LLMEndpoint(Base):
    """Labeled LLM endpoint: path_pattern (prefix or regex), methods, label."""

    __tablename__ = "llm_endpoints"

    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    path_pattern = Column(String(500), nullable=False)  # e.g. /api/chat, /v1/completions (prefix match) or regex
    methods = Column(String(100), nullable=False)       # e.g. POST or POST,PUT
    label = Column(String(100), nullable=False)         # e.g. chat
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "path_pattern": self.path_pattern,
            "methods": self.methods,
            "label": self.label,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
