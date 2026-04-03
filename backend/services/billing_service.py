"""
Billing service - Razorpay integration for WAF SaaS subscriptions.
Handles plan management, subscription CRUD, and checkout sessions.
"""
import json
import os
from typing import Optional

from loguru import logger
from sqlalchemy.orm import Session

from backend.models.billing_plan import BillingPlan
from backend.models.subscription import Subscription
from backend.models.organization import Organization
from backend.lib.datetime_utils import utc_now


# Razorpay client (lazy-loaded)
_razorpay_client = None


def _get_razorpay_client():
    """Get or create Razorpay client. Returns None if keys not configured."""
    global _razorpay_client
    if _razorpay_client is not None:
        return _razorpay_client

    key_id = os.getenv("RAZORPAY_KEY_ID", "")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")

    if not key_id or not key_secret:
        logger.warning("Razorpay API keys not configured (RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET)")
        return None

    try:
        import razorpay
        _razorpay_client = razorpay.Client(auth=(key_id, key_secret))
        logger.info("Razorpay client initialized")
        return _razorpay_client
    except ImportError:
        logger.warning("razorpay package not installed; billing features disabled")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Razorpay client: {e}")
        return None


class BillingService:
    """Service for managing billing plans and subscriptions."""

    def __init__(self, db: Session):
        self.db = db

    # -- Plan management --

    def list_plans(self, active_only: bool = True) -> list[BillingPlan]:
        """List all billing plans."""
        query = self.db.query(BillingPlan)
        if active_only:
            query = query.filter(BillingPlan.is_active)
        return query.order_by(BillingPlan.price_monthly).all()

    def get_plan_by_slug(self, slug: str) -> Optional[BillingPlan]:
        """Get a billing plan by slug."""
        return self.db.query(BillingPlan).filter(BillingPlan.slug == slug).first()

    def seed_default_plans(self) -> list[BillingPlan]:
        """Seed default WAF pricing plans if they don't exist."""
        defaults = [
            {
                "name": "Free",
                "slug": "free",
                "price_monthly": 0,
                "requests_limit": 10000,
                "max_domains": 1,
                "features": json.dumps([
                    "Basic WAF protection",
                    "1 domain",
                    "10K requests/month",
                    "Community support",
                ]),
            },
            {
                "name": "Pro",
                "slug": "pro",
                "price_monthly": 299900,  # 2999 INR in paise
                "requests_limit": 500000,
                "max_domains": 5,
                "features": json.dumps([
                    "Full WAF protection",
                    "Bot detection + management",
                    "5 domains",
                    "500K requests/month",
                    "DDoS protection",
                    "Custom security rules",
                    "Email support",
                ]),
            },
            {
                "name": "Enterprise",
                "slug": "enterprise",
                "price_monthly": 999900,  # 9999 INR in paise
                "requests_limit": -1,  # Unlimited
                "max_domains": -1,  # Unlimited
                "features": json.dumps([
                    "Everything in Pro",
                    "Unlimited domains",
                    "Unlimited requests",
                    "Priority support",
                    "Custom managed rules",
                    "Dedicated account manager",
                    "SLA guarantee",
                    "ONNX ML optimization",
                ]),
            },
        ]

        created = []
        for plan_data in defaults:
            existing = self.db.query(BillingPlan).filter(
                BillingPlan.slug == plan_data["slug"]
            ).first()
            if not existing:
                plan = BillingPlan(**plan_data)
                self.db.add(plan)
                created.append(plan)

        if created:
            self.db.commit()
            for p in created:
                self.db.refresh(p)
        return created

    def create_razorpay_plan(self, plan: BillingPlan) -> Optional[str]:
        """Create a plan in Razorpay and store the razorpay_plan_id."""
        client = _get_razorpay_client()
        if not client:
            return None

        if plan.price_monthly <= 0:
            return None  # Free plans don't need Razorpay plans

        try:
            rz_plan = client.plan.create({
                "period": "monthly",
                "interval": 1,
                "item": {
                    "name": f"WAF {plan.name} Plan",
                    "amount": plan.price_monthly,
                    "currency": "INR",
                    "description": f"WAF {plan.name} - {plan.requests_limit} requests/month",
                },
            })
            plan.razorpay_plan_id = rz_plan["id"]
            self.db.commit()
            logger.info(f"Created Razorpay plan: {rz_plan['id']} for {plan.name}")
            return rz_plan["id"]
        except Exception as e:
            logger.error(f"Failed to create Razorpay plan for {plan.name}: {e}")
            return None

    # -- Subscription management --

    def get_subscription(self, org_id: int) -> Optional[Subscription]:
        """Get active subscription for an organization."""
        return self.db.query(Subscription).filter(
            Subscription.org_id == org_id,
            Subscription.status.in_(["active", "pending"]),
        ).first()

    def create_subscription(self, org_id: int, plan_slug: str) -> dict:
        """Create a Razorpay subscription for an org. Returns checkout data."""
        plan = self.get_plan_by_slug(plan_slug)
        if not plan:
            return {"error": "Plan not found"}

        # Free plan: create subscription record directly (no Razorpay)
        if plan.price_monthly <= 0:
            sub = Subscription(
                org_id=org_id,
                plan_id=plan.id,
                status="active",
                current_period_start=utc_now(),
            )
            self.db.add(sub)

            # Update org plan
            org = self.db.query(Organization).filter(Organization.id == org_id).first()
            if org:
                org.plan = plan.slug
            self.db.commit()
            self.db.refresh(sub)
            return {"subscription": sub.to_dict(), "checkout_url": None}

        # Paid plan: create Razorpay subscription
        client = _get_razorpay_client()
        if not client:
            return {"error": "Razorpay not configured. Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET."}

        if not plan.razorpay_plan_id:
            return {"error": f"Plan {plan.name} has no Razorpay plan ID. Run plan sync first."}

        try:
            rz_sub = client.subscription.create({
                "plan_id": plan.razorpay_plan_id,
                "total_count": 12,  # 12 months billing cycle
                "quantity": 1,
            })

            sub = Subscription(
                org_id=org_id,
                plan_id=plan.id,
                razorpay_subscription_id=rz_sub["id"],
                status="pending",
            )
            self.db.add(sub)
            self.db.commit()
            self.db.refresh(sub)

            return {
                "subscription": sub.to_dict(),
                "razorpay_subscription_id": rz_sub["id"],
                "razorpay_key_id": os.getenv("RAZORPAY_KEY_ID", ""),
                "short_url": rz_sub.get("short_url"),
            }
        except Exception as e:
            logger.error(f"Failed to create Razorpay subscription: {e}")
            return {"error": str(e)}

    def activate_subscription(self, razorpay_subscription_id: str) -> Optional[Subscription]:
        """Activate a subscription after successful payment."""
        sub = self.db.query(Subscription).filter(
            Subscription.razorpay_subscription_id == razorpay_subscription_id
        ).first()
        if not sub:
            return None

        sub.status = "active"
        sub.current_period_start = utc_now()
        sub.updated_at = utc_now()

        # Update org plan
        plan = self.db.query(BillingPlan).filter(BillingPlan.id == sub.plan_id).first()
        if plan:
            org = self.db.query(Organization).filter(Organization.id == sub.org_id).first()
            if org:
                org.plan = plan.slug

        self.db.commit()
        self.db.refresh(sub)
        logger.info(f"Activated subscription {razorpay_subscription_id} for org {sub.org_id}")
        return sub

    def cancel_subscription(self, org_id: int) -> Optional[Subscription]:
        """Cancel an active subscription."""
        sub = self.get_subscription(org_id)
        if not sub:
            return None

        # Cancel in Razorpay if applicable
        if sub.razorpay_subscription_id:
            client = _get_razorpay_client()
            if client:
                try:
                    client.subscription.cancel(sub.razorpay_subscription_id)
                except Exception as e:
                    logger.error(f"Failed to cancel Razorpay subscription: {e}")

        sub.status = "cancelled"
        sub.cancelled_at = utc_now()
        sub.updated_at = utc_now()

        # Downgrade org to free plan
        org = self.db.query(Organization).filter(Organization.id == org_id).first()
        if org:
            org.plan = "free"

        self.db.commit()
        self.db.refresh(sub)
        logger.info(f"Cancelled subscription for org {org_id}")
        return sub

    def upgrade_plan(self, org_id: int, new_plan_slug: str) -> dict:
        """Upgrade or downgrade subscription to a new plan."""
        current_sub = self.get_subscription(org_id)

        # Cancel current if exists
        if current_sub:
            self.cancel_subscription(org_id)

        # Create new subscription
        return self.create_subscription(org_id, new_plan_slug)
