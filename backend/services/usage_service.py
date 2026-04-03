"""
Usage tracking service - per-org request counting and plan limit enforcement.
"""
from typing import Optional

from loguru import logger
from sqlalchemy.orm import Session

from backend.models.usage import UsageRecord
from backend.models.subscription import Subscription
from backend.models.billing_plan import BillingPlan
from backend.lib.datetime_utils import utc_now


def _current_month() -> str:
    """Return current month as YYYY-MM string."""
    return utc_now().strftime("%Y-%m")


class UsageService:
    """Track and enforce per-org request usage against plan limits."""

    def __init__(self, db: Session):
        self.db = db

    def increment_usage(self, org_id: int, count: int = 1) -> int:
        """Increment request count for org in current month. Returns new total."""
        month = _current_month()
        record = self.db.query(UsageRecord).filter(
            UsageRecord.org_id == org_id,
            UsageRecord.month == month,
        ).first()

        if record:
            record.requests_count += count
            record.last_updated = utc_now()
        else:
            record = UsageRecord(
                org_id=org_id,
                month=month,
                requests_count=count,
            )
            self.db.add(record)

        self.db.commit()
        return record.requests_count

    def get_usage(self, org_id: int, month: Optional[str] = None) -> dict:
        """Get usage stats for org in a given month (defaults to current)."""
        month = month or _current_month()
        record = self.db.query(UsageRecord).filter(
            UsageRecord.org_id == org_id,
            UsageRecord.month == month,
        ).first()

        requests_count = record.requests_count if record else 0

        # Get plan limit
        limit = self._get_plan_limit(org_id)

        return {
            "org_id": org_id,
            "month": month,
            "requests_count": requests_count,
            "requests_limit": limit,
            "usage_percent": round((requests_count / limit) * 100, 1) if limit > 0 else 0,
            "within_limit": limit < 0 or requests_count <= limit,  # -1 = unlimited
        }

    def is_within_limit(self, org_id: int) -> bool:
        """Check if org is within their plan's request limit."""
        limit = self._get_plan_limit(org_id)
        if limit < 0:  # Unlimited
            return True

        month = _current_month()
        record = self.db.query(UsageRecord).filter(
            UsageRecord.org_id == org_id,
            UsageRecord.month == month,
        ).first()

        current = record.requests_count if record else 0
        return current <= limit

    def _get_plan_limit(self, org_id: int) -> int:
        """Get the request limit for org's current plan. Returns -1 for unlimited."""
        sub = self.db.query(Subscription).filter(
            Subscription.org_id == org_id,
            Subscription.status.in_(["active", "pending"]),
        ).first()

        if not sub:
            # No subscription = free plan default
            free_plan = self.db.query(BillingPlan).filter(BillingPlan.slug == "free").first()
            return free_plan.requests_limit if free_plan else 10000

        plan = self.db.query(BillingPlan).filter(BillingPlan.id == sub.plan_id).first()
        return plan.requests_limit if plan else 10000

    def reset_monthly_usage(self) -> int:
        """Reset usage for previous months (cleanup). Returns count of records cleaned."""
        current = _current_month()
        old_records = self.db.query(UsageRecord).filter(
            UsageRecord.month != current
        ).all()
        count = len(old_records)
        for record in old_records:
            self.db.delete(record)
        if count > 0:
            self.db.commit()
            logger.info(f"Cleaned {count} old usage records")
        return count
