# Week 3 Day 4: Razorpay Subscriptions + Payment Checkout

**Status:** PENDING
**Theme:** SaaS monetization - subscription plans + checkout flow

## Goal

Build Razorpay billing infrastructure: pricing tiers (Free/Pro/Enterprise), Razorpay Subscriptions API integration, checkout flow for tenant signup and plan upgrades.

## Pricing Tiers

| Plan | Price | Requests/month | Features |
|------|-------|----------------|----------|
| Free | 0 INR | 10,000 | Basic WAF, 1 domain |
| Pro | 2,999 INR/mo | 500,000 | Full WAF, bot protection, 5 domains |
| Enterprise | 9,999 INR/mo | Unlimited | Everything + priority support, custom rules |

## Implementation

### 1. New Model: `backend/models/subscription.py`
- Subscription model: org_id, plan_name, razorpay_subscription_id, razorpay_plan_id, status, current_period_start, current_period_end, requests_limit, requests_used

### 2. New Model: `backend/models/billing_plan.py`
- BillingPlan model: name, slug, razorpay_plan_id, price_monthly, requests_limit, features (JSON), is_active

### 3. New Service: `backend/services/billing_service.py`
- create_razorpay_plan() - Create plan in Razorpay
- create_subscription() - Create subscription for org
- get_subscription() - Get current subscription for org
- upgrade_plan() - Change subscription plan
- cancel_subscription() - Cancel subscription
- check_usage_limit() - Check if org has exceeded request quota

### 4. New Routes: `backend/routes/billing.py`
- GET /api/billing/plans - List available plans
- GET /api/billing/subscription - Get current org subscription
- POST /api/billing/checkout - Create Razorpay checkout session
- POST /api/billing/upgrade - Upgrade/downgrade plan
- POST /api/billing/cancel - Cancel subscription

### 5. Config: Environment variables
- RAZORPAY_KEY_ID
- RAZORPAY_KEY_SECRET
- RAZORPAY_WEBHOOK_SECRET

## Files to Create/Modify
1. `backend/models/subscription.py` - NEW
2. `backend/models/billing_plan.py` - NEW
3. `backend/services/billing_service.py` - NEW
4. `backend/controllers/billing.py` - NEW
5. `backend/routes/billing.py` - NEW
6. `backend/main.py` - Register billing routes
7. `requirements.txt` - Add razorpay package
