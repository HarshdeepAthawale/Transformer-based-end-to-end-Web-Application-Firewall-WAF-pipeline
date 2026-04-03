"""
Subscription model - tracks org billing state with Razorpay.
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from backend.database import Base
from backend.lib.datetime_utils import utc_now


class Subscription(Base):
    """Organization subscription linked to Razorpay."""
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    org_id = Column(Integer, ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    plan_id = Column(Integer, ForeignKey("billing_plans.id"), nullable=False)
    razorpay_subscription_id = Column(String(100), nullable=True, unique=True, index=True)
    razorpay_customer_id = Column(String(100), nullable=True, index=True)
    status = Column(String(30), nullable=False, default="active", index=True)
    # Status values: active, cancelled, paused, halted, pending, expired
    current_period_start = Column(DateTime, nullable=True)
    current_period_end = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "org_id": self.org_id,
            "plan_id": self.plan_id,
            "razorpay_subscription_id": self.razorpay_subscription_id,
            "status": self.status,
            "current_period_start": self.current_period_start.isoformat() if self.current_period_start else None,
            "current_period_end": self.current_period_end.isoformat() if self.current_period_end else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
