"""Billing API - plans, subscriptions, checkout."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.auth import get_current_tenant, require_waf_api_auth
from backend.controllers import billing as ctrl

router = APIRouter()


class CheckoutRequest(BaseModel):
    plan_slug: str


class UpgradeRequest(BaseModel):
    new_plan_slug: str


@router.get("/plans")
async def list_plans(db: Session = Depends(get_db)):
    """List all available billing plans. Public endpoint."""
    return ctrl.list_plans(db)


@router.get("/subscription")
async def get_subscription(
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Get current subscription for the authenticated org."""
    return ctrl.get_subscription(db, org_id)


@router.post("/checkout")
async def create_checkout(
    body: CheckoutRequest,
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Create a Razorpay checkout session for a plan."""
    return ctrl.create_checkout(db, org_id, body.plan_slug)


@router.post("/upgrade")
async def upgrade_plan(
    body: UpgradeRequest,
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Upgrade or downgrade to a different plan."""
    return ctrl.upgrade_plan(db, org_id, body.new_plan_slug)


@router.post("/cancel")
async def cancel_subscription(
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Cancel current subscription. Downgrades to free plan."""
    return ctrl.cancel_subscription(db, org_id)


@router.get("/usage")
async def get_usage(
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Get current month's usage stats for the authenticated org."""
    from backend.services.usage_service import UsageService
    from backend.lib.datetime_utils import utc_now as _utc_now
    svc = UsageService(db)
    data = svc.get_usage(org_id)
    return {"success": True, "data": data, "timestamp": _utc_now().isoformat()}


@router.post("/seed-plans")
async def seed_plans(
    db: Session = Depends(get_db),
    _auth=Depends(require_waf_api_auth),
):
    """Seed default billing plans (admin only)."""
    return ctrl.seed_plans(db)
