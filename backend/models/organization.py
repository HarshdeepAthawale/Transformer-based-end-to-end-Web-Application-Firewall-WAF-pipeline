from sqlalchemy import Column, Integer, String, Boolean, DateTime
from backend.database import Base
from datetime import datetime


class Organization(Base):
    """Organization/Tenant model for multi-tenant WAF system.

    Each organization represents a customer account.
    All tenant-scoped data (users, threats, alerts, etc.) belongs to exactly one organization.
    """
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    slug = Column(String(100), unique=True, nullable=False, index=True)  # URL-safe: "acme-corp"
    plan = Column(String(50), default="free", nullable=False)  # free, starter, pro, enterprise
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    owner_email = Column(String(255), nullable=True)

    def __repr__(self):
        return f"<Organization(id={self.id}, name='{self.name}', slug='{self.slug}')>"
