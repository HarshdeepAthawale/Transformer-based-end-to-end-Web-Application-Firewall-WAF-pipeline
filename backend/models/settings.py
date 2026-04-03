"""
Account settings key-value store for B2B SaaS preferences.
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, UniqueConstraint
from backend.database import Base
from backend.lib.datetime_utils import utc_now


class AccountSetting(Base):
    """Key-value store for account-level settings (multi-tenant)."""
    __tablename__ = "account_settings"
    __table_args__ = (
        UniqueConstraint("org_id", "key", name="uq_account_settings_org_key"),
    )

    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    key = Column(String(100), nullable=False, index=True)
    value = Column(Text, nullable=True)  # JSON string

    def to_dict(self):
        return {
            "org_id": self.org_id,
            "key": self.key,
            "value": self.value,
        }
