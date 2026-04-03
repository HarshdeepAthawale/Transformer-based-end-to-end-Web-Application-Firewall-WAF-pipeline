"""
Billing plan model - WAF pricing tiers.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from backend.database import Base
from backend.lib.datetime_utils import utc_now


class BillingPlan(Base):
    """WAF pricing plan (Free, Pro, Enterprise)."""
    __tablename__ = "billing_plans"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True)
    slug = Column(String(50), nullable=False, unique=True, index=True)
    razorpay_plan_id = Column(String(100), nullable=True, index=True)
    price_monthly = Column(Integer, nullable=False, default=0)  # in paise (INR smallest unit)
    requests_limit = Column(Integer, nullable=False, default=10000)  # monthly request quota; -1 = unlimited
    features = Column(Text, nullable=True)  # JSON array of feature strings
    max_domains = Column(Integer, nullable=False, default=1)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    def to_dict(self):
        import json
        features_list = json.loads(self.features) if self.features else []
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "razorpay_plan_id": self.razorpay_plan_id,
            "price_monthly": self.price_monthly,
            "price_display": f"{self.price_monthly / 100:.0f}" if self.price_monthly else "0",
            "requests_limit": self.requests_limit,
            "max_domains": self.max_domains,
            "features": features_list,
            "is_active": self.is_active,
        }
