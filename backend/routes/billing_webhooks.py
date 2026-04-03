"""
Razorpay webhook handler - processes payment events.
Verifies webhook signature and updates subscription status.
"""
import os
import json

from fastapi import APIRouter, Request, HTTPException
from loguru import logger

from backend.database import SessionLocal
from backend.services.billing_service import BillingService
from backend.lib.datetime_utils import utc_now

router = APIRouter()


def _verify_razorpay_signature(body: bytes, signature: str) -> bool:
    """Verify Razorpay webhook signature using webhook secret."""
    webhook_secret = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")
    if not webhook_secret:
        logger.warning("RAZORPAY_WEBHOOK_SECRET not set; skipping signature verification")
        return True  # Allow in dev mode

    try:
        import hmac
        import hashlib
        expected = hmac.new(
            webhook_secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, signature)
    except Exception as e:
        logger.error(f"Webhook signature verification failed: {e}")
        return False


@router.post("/razorpay")
async def razorpay_webhook(request: Request):
    """
    Handle Razorpay webhook events.
    Events handled:
    - subscription.activated: Mark subscription as active
    - subscription.charged: Record successful payment
    - subscription.cancelled: Mark subscription as cancelled
    - payment.captured: Confirm payment capture
    - payment.failed: Log payment failure
    """
    body = await request.body()

    # Verify signature
    signature = request.headers.get("x-razorpay-signature", "")
    if not _verify_razorpay_signature(body, signature):
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    event = payload.get("event", "")
    event_payload = payload.get("payload", {})

    logger.info(f"Razorpay webhook received: {event}")

    db = SessionLocal()
    try:
        svc = BillingService(db)

        if event == "subscription.activated":
            sub_entity = event_payload.get("subscription", {}).get("entity", {})
            rz_sub_id = sub_entity.get("id")
            if rz_sub_id:
                svc.activate_subscription(rz_sub_id)
                logger.info(f"Subscription activated via webhook: {rz_sub_id}")

        elif event == "subscription.charged":
            sub_entity = event_payload.get("subscription", {}).get("entity", {})
            rz_sub_id = sub_entity.get("id")
            logger.info(f"Subscription charged: {rz_sub_id}")

        elif event == "subscription.cancelled":
            sub_entity = event_payload.get("subscription", {}).get("entity", {})
            rz_sub_id = sub_entity.get("id")
            if rz_sub_id:
                from backend.models.subscription import Subscription
                sub = db.query(Subscription).filter(
                    Subscription.razorpay_subscription_id == rz_sub_id
                ).first()
                if sub:
                    sub.status = "cancelled"
                    sub.cancelled_at = utc_now()
                    db.commit()
                    logger.info(f"Subscription cancelled via webhook: {rz_sub_id}")

        elif event == "payment.captured":
            payment_entity = event_payload.get("payment", {}).get("entity", {})
            logger.info(
                f"Payment captured: {payment_entity.get('id')} "
                f"amount={payment_entity.get('amount')} "
                f"currency={payment_entity.get('currency')}"
            )

        elif event == "payment.failed":
            payment_entity = event_payload.get("payment", {}).get("entity", {})
            logger.warning(
                f"Payment failed: {payment_entity.get('id')} "
                f"reason={payment_entity.get('error_description', 'unknown')}"
            )

        else:
            logger.info(f"Unhandled Razorpay event: {event}")

    except Exception as e:
        logger.error(f"Error processing Razorpay webhook: {e}")
        raise HTTPException(status_code=500, detail="Internal error processing webhook")
    finally:
        db.close()

    return {"status": "ok", "event": event}
