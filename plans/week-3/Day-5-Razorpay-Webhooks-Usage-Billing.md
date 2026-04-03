# Week 3 Day 5: Razorpay Webhooks + Usage-Based Billing

**Status:** PENDING
**Theme:** Webhook handler for payment events + usage tracking + enforcement

## Goal

Handle Razorpay webhook events (payment.captured, subscription.activated/cancelled/charged). Track per-org request usage. Enforce plan limits (block requests when quota exceeded).

## Implementation

### 1. Razorpay Webhook Handler: `backend/routes/billing_webhooks.py`
- POST /api/billing/webhooks/razorpay
- Verify webhook signature using Razorpay utility
- Handle events: subscription.activated, subscription.charged, subscription.cancelled, payment.captured, payment.failed

### 2. Usage Tracking: `backend/services/usage_service.py`
- increment_usage(org_id) - Called per request from gateway
- get_usage(org_id) - Get current month usage
- reset_monthly_usage() - Cron job to reset at month start
- is_within_limit(org_id) - Check if org is within plan limit

### 3. Usage Enforcement Middleware: `backend/middleware/usage_limit_middleware.py`
- Check org's usage against plan limit on each request
- Return 402 Payment Required if quota exceeded
- Skip for Enterprise (unlimited) plans

### 4. Usage Model: `backend/models/usage.py`
- UsageRecord: org_id, month (YYYY-MM), requests_count, last_updated

## Files to Create/Modify
1. `backend/routes/billing_webhooks.py` - NEW
2. `backend/services/usage_service.py` - NEW
3. `backend/middleware/usage_limit_middleware.py` - NEW
4. `backend/models/usage.py` - NEW
5. `backend/main.py` - Register webhook routes + usage middleware
