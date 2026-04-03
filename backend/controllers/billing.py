"""Billing controller - thin adapter between routes and billing service."""
from sqlalchemy.orm import Session

from backend.lib.datetime_utils import utc_now
from backend.services.billing_service import BillingService


def list_plans(db: Session) -> dict:
    """List all active billing plans."""
    svc = BillingService(db)
    plans = svc.list_plans()
    return {
        "success": True,
        "data": [p.to_dict() for p in plans],
        "timestamp": utc_now().isoformat(),
    }


def get_subscription(db: Session, org_id: int) -> dict:
    """Get current subscription for org."""
    svc = BillingService(db)
    sub = svc.get_subscription(org_id)
    if not sub:
        return {
            "success": True,
            "data": {"plan": "free", "status": "active", "subscription": None},
            "timestamp": utc_now().isoformat(),
        }

    plan = svc.get_plan_by_slug("free")  # fallback
    from backend.models.billing_plan import BillingPlan
    plan_obj = db.query(BillingPlan).filter(BillingPlan.id == sub.plan_id).first()

    return {
        "success": True,
        "data": {
            "subscription": sub.to_dict(),
            "plan": plan_obj.to_dict() if plan_obj else None,
        },
        "timestamp": utc_now().isoformat(),
    }


def create_checkout(db: Session, org_id: int, plan_slug: str) -> dict:
    """Create checkout session for a plan."""
    svc = BillingService(db)
    result = svc.create_subscription(org_id, plan_slug)
    if "error" in result:
        return {"success": False, "message": result["error"], "timestamp": utc_now().isoformat()}
    return {"success": True, "data": result, "timestamp": utc_now().isoformat()}


def upgrade_plan(db: Session, org_id: int, new_plan_slug: str) -> dict:
    """Upgrade or downgrade plan."""
    svc = BillingService(db)
    result = svc.upgrade_plan(org_id, new_plan_slug)
    if "error" in result:
        return {"success": False, "message": result["error"], "timestamp": utc_now().isoformat()}
    return {"success": True, "data": result, "timestamp": utc_now().isoformat()}


def cancel_subscription(db: Session, org_id: int) -> dict:
    """Cancel current subscription."""
    svc = BillingService(db)
    sub = svc.cancel_subscription(org_id)
    if not sub:
        return {"success": False, "message": "No active subscription found", "timestamp": utc_now().isoformat()}
    return {
        "success": True,
        "data": sub.to_dict(),
        "message": "Subscription cancelled. Downgraded to free plan.",
        "timestamp": utc_now().isoformat(),
    }


def seed_plans(db: Session) -> dict:
    """Seed default plans (admin operation)."""
    svc = BillingService(db)
    created = svc.seed_default_plans()
    return {
        "success": True,
        "data": [p.to_dict() for p in created],
        "message": f"Seeded {len(created)} new plans",
        "timestamp": utc_now().isoformat(),
    }
