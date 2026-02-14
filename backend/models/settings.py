"""
Account settings key-value store for B2B SaaS preferences.
"""
from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from backend.database import Base
from datetime import datetime


class AccountSetting(Base):
    """Key-value store for account-level settings (single-tenant)."""
    __tablename__ = "account_settings"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=True)  # JSON string

    def to_dict(self):
        return {
            "key": self.key,
            "value": self.value,
        }
