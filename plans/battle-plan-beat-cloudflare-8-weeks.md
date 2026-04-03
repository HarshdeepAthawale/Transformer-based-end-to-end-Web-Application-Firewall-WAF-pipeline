# Battle Plan: Transformer WAF Startup — Beat Cloudflare in 2 Months

## Context

You have a substantial WAF prototype: fine-tuned DistilBERT (99.48% accuracy), FastAPI gateway with 10 security layers, 30+ API routes, Next.js dashboard. But it's fundamentally **single-tenant**, has **no billing**, uses **SHA-256 for passwords**, regenerates JWT secrets on restart, and runs only via Docker Compose. The gap to "sellable product" is significant but achievable in 8 weeks with ruthless prioritization.

**The core insight**: You don't beat Cloudflare on infrastructure (300+ PoPs, $1B revenue). You beat them by shipping a product that 50 startups pay $49-$299/mo for within 90 days. Your weapons:
1. **AI/ML-first** — zero-day detection via anomaly scoring (Cloudflare is still signature-heavy)
2. **Firewall for AI** — prompt injection + PII detection for LLM endpoints (massive gap in market)
3. **Transparent scoring** — attack score 0-100 visible per request (Cloudflare is a black box)
4. **5-minute setup** — CNAME your domain, get protected
5. **Cost disruption** — 10x cheaper for startups/SMBs

## What to CUT (Defer Post-Launch)

Do NOT touch these during the 8 weeks:
- DNS automation (`backend/routes/dns.py`) — customers CNAME manually
- ACME/SSL auto-provisioning — use Caddy/Traefik with cert-manager
- Edge CDN / geographic distribution — single region is fine for launch
- Continuous learning scheduler (`backend/ml/learning/`) — retrain offline
- AI Copilot (`backend/agents/`) — impressive demo, not MVP
- Upload scanning / ClamAV — enterprise feature
- Credential leak HIBP — keep disabled
- Adaptive DDoS learning — static thresholds fine for launch

## Pricing Strategy

- **Free**: 100K req/mo, 1 domain, community support
- **Startup ($49/mo)**: 5M req/mo, 5 domains, email alerts
- **Business ($299/mo)**: 50M req/mo, unlimited domains, Firewall for AI, priority support

---

## WEEK 1: Multi-Tenancy Foundation (CRITICAL PATH)

**Goal**: Every table, query, JWT token, and API route becomes tenant-aware.

### Task 1.1: Organization Model + Schema Migration
- Create `backend/models/organization.py` — fields: `id`, `name`, `slug`, `plan` (free/startup/business), `is_active`, `stripe_customer_id`, `request_quota_monthly`, `domains` (JSON)
- Add `org_id` foreign key to ALL 13+ tenant-scoped models:
  - `backend/models/users.py`, `security_event.py`, `threats.py`, `alerts.py`, `traffic.py`, `security_rules.py`, `ip_blacklist.py`, `geo_rules.py`, `rate_limit_config.py`, `settings.py`, `llm_endpoint.py`, `firewall_ai_pattern.py`, `audit_log.py`
- Initialize Alembic (replace ad-hoc ALTER TABLE in `database.py` lines 95-166)
- Create initial migration + org_id migration

### Task 1.2: Auth & JWT Hardening
- **Fix password hashing**: `backend/models/users.py` line 48 uses `hashlib.sha256` — replace with bcrypt via passlib
- **Add org_id to JWT**: `backend/auth.py` `create_access_token()` (line 26) must include `org_id`
- **Fix JWT_SECRET**: `auth.py` line 18 generates random secret on restart — must fail hard if not set in production
- Create `get_current_org()` dependency in `auth.py`

### Task 1.3: Query-Level Tenant Isolation
- Create `backend/middleware/tenant.py` — dependency that extracts org_id and applies to all queries
- Update every service in `backend/services/` to filter by org_id (analytics, threats, traffic, metrics, alerts, rules, ip_fencing, firewall_ai — 15+ files)
- Update all controllers to pass org_id through

### Deliverables
- [ ] All DB models have org_id
- [ ] Alembic migrations work
- [ ] JWT contains org_id
- [ ] Passwords use bcrypt
- [ ] Every query filters by org_id
- [ ] Integration test: 2 orgs cannot see each other's data

---

## WEEK 2: Self-Service Signup & Gateway Tenant Routing

**Goal**: New user can sign up, create org, and log into dashboard. Gateway routes traffic per tenant.

### Task 2.1: Signup Flow
- Create `POST /api/auth/signup` — creates Organization + User (admin role) + default settings + returns JWT
- New file: `backend/routes/auth.py`
- Add email verification fields to User model
- Create `frontend/app/signup/page.tsx`
- Update `frontend/lib/auth.ts` NextAuth config to include orgId in session

### Task 2.2: API Key System
- Create `backend/models/api_key.py` — replace JSON blob in User model with proper table
- Fields: id, org_id, user_id, key_hash, key_prefix, name, scopes, last_used_at, expires_at, is_active
- Update auth middleware to validate API keys from DB (cached in Redis)

### Task 2.3: Gateway Tenant Routing
- Gateway reads `Host` header → looks up org_id from Redis hash (`domain -> org_id`)
- Pass org_id in all event logging (MongoDB + backend)
- Domain registration API: `POST /api/v1/domains`, `GET /api/v1/domains`
- Instructions for CNAME setup (manual for MVP)

### Deliverables
- [ ] Signup works end-to-end (email → org → JWT → dashboard)
- [ ] API keys in proper DB table with revocation
- [ ] Gateway resolves tenant from Host header
- [ ] Domain registration API exists

---

## WEEK 3: ML Optimization & Gateway Hardening

**Goal**: 3-5x faster inference, production-grade gateway.

### Task 3.1: ONNX Runtime Conversion
- New script: `scripts/export_onnx.py` — convert DistilBERT to ONNX + INT8 quantization
- Output: `models/waf-distilbert-onnx/model.onnx` (~67MB vs 267MB)
- Create `ONNXWAFClassifier` in `backend/ml/waf_classifier.py` using `onnxruntime.InferenceSession`
- Update `backend/core/waf_factory.py` to prefer ONNX, fall back to PyTorch
- Implement micro-batching in `gateway/waf_inspect.py` — accumulate 5ms, batch-classify using existing `classify_batch()` (line 159, never called from gateway)

### Task 3.2: Gateway Production Hardening
- Structured JSON logging (replace loguru text with JSON: request_id, org_id, client_ip, decision, latency_ms)
- Health check includes model inference latency in readiness
- Audit all FAIL_OPEN configs — verify Redis/MongoDB failures don't crash process
- Add overall request timeout (currently only PROXY_TIMEOUT, no total including WAF inspection)

### Deliverables
- [ ] ONNX model validated (same accuracy, 3x faster)
- [ ] Structured JSON logs in gateway and backend
- [ ] Gateway survives Redis/MongoDB outages
- [ ] Target: <50ms p95 WAF inspection latency

---

## WEEK 4: End-to-End Feature Wiring & Testing

**Goal**: Finish what exists (50-70% → 100%). No new features.

### Task 4.1: Wire Firewall for AI (KEY DIFFERENTIATOR)
- Verify full chain: request → `gateway/firewall_ai.py` → backend service → pattern match → block/allow
- Test with real prompt injection payloads
- Wire LLM endpoint registration (customer registers paths like `/api/v1/chat/completions`)
- Verify events logged with org_id

### Task 4.2: Wire Bot Management + Managed Rules
- Verify bot score bands evaluation end-to-end (gateway → backend → decision)
- Verify OWASP CRS rules load and evaluate in `gateway/managed_rules.py`
- Test with real attack payloads from `scripts/attack_tests/`

### Task 4.3: Integration Test Suite
- `tests/integration/test_multi_tenant.py` — 2 orgs, verify data isolation
- `tests/integration/test_gateway_e2e.py` — requests through gateway, events in DB
- `tests/integration/test_signup_flow.py` — signup → login → domain → traffic
- `tests/integration/test_firewall_ai.py` — register LLM endpoint → prompt injection → block
- `tests/integration/test_api_keys.py` — create → use → verify

### Deliverables
- [ ] Firewall for AI works end-to-end
- [ ] Bot management works with real scoring
- [ ] Managed rules evaluate correctly
- [ ] Integration tests cover all critical paths

---

## WEEK 5: Kubernetes & CI/CD

**Goal**: Deploy to K8s. Auto-deploy on merge to main.

### Task 5.1: Kubernetes Manifests
- New `k8s/` directory with deployments for: gateway (2 replicas), backend (2 replicas), frontend (2 replicas), Redis (StatefulSet), MongoDB (StatefulSet)
- Ingress with TLS (cert-manager + Let's Encrypt)
- HPA for gateway (target 70% CPU)
- ConfigMap + Secrets for env vars

### Task 5.2: PostgreSQL Migration
- Switch from SQLite to managed PostgreSQL (~$15/mo)
- Test all Alembic migrations against PostgreSQL
- Add connection pooling config to `backend/database.py`

### Task 5.3: Docker Image Optimization
- Multi-stage Dockerfile: ONNX runtime only in prod (no PyTorch → image from ~2GB to ~500MB)
- Gunicorn + uvicorn workers for gateway

### Task 5.4: CI/CD Pipeline
- Expand `.github/workflows/ci.yml`: build + push Docker images, integration tests, SAST with bandit
- New `.github/workflows/deploy.yml`: deploy to staging on develop merge, production on main merge
- Smoke test after deploy, rollback on failure

### Deliverables
- [ ] K8s manifests for all services
- [ ] PostgreSQL primary database
- [ ] CI/CD auto-deploys to staging/production
- [ ] Docker images optimized
- [ ] HPA scales gateway under load

---

## WEEK 6: Observability & Onboarding UX

**Goal**: See what happens in production. Customers onboard in 5 minutes.

### Task 6.1: Prometheus Metrics + Grafana
- Add `prometheus_client` to gateway: `waf_requests_total`, `waf_request_duration_seconds`, `waf_model_inference_seconds`
- Create Grafana dashboards: request volume by org, block rate by attack type, inference latency p50/p95/p99
- Alerting rules: error rate > 5%, inference p99 > 500ms, gateway pods < 2

### Task 6.2: Customer Onboarding Wizard
- New `frontend/app/onboarding/page.tsx`: Enter domain → show CNAME record → verify DNS → choose mode → dashboard live
- `POST /api/v1/domains/{id}/verify` — check CNAME, mark verified, enable routing

### Task 6.3: Quick Start Documentation
- `docs/quickstart.md` — 5-minute guide
- API reference (OpenAPI auto-generated)
- curl/Python/JS examples

### Deliverables
- [ ] Grafana dashboards operational
- [ ] Prometheus alerts configured
- [ ] Onboarding wizard works end-to-end
- [ ] Quick start docs published

---

## WEEK 7: Billing & Usage Metering

**Goal**: Customers can pay you money.

### Task 7.1: Usage Metering
- New `gateway/metering.py`: increment Redis counter per request per org (`usage:{org_id}:{YYYY-MM}:requests`)
- Quota enforcement: check counter vs `request_quota_monthly`, return 429 if exceeded (10% grace)
- `GET /api/v1/usage` — current month's count, quota remaining
- Nightly background task aggregates Redis → `backend/models/usage.py` monthly records

### Task 7.2: Stripe Integration
- New `backend/services/billing_service.py`: Stripe customer creation, checkout sessions, webhook handler
- New `backend/routes/billing.py`: checkout, portal, webhook, subscription status endpoints
- Handle: subscription created, payment succeeded/failed, canceled → update org plan + quota

### Task 7.3: Frontend Billing Page
- New `frontend/app/billing/page.tsx`: current plan, usage meter, upgrade button (Stripe Checkout), manage button (Stripe Portal)

### Deliverables
- [ ] Every request metered per org
- [ ] Quota enforcement in gateway (429 on excess)
- [ ] Stripe checkout + subscription management
- [ ] Frontend billing page with usage display

---

## WEEK 8: Launch Preparation

**Goal**: Ship it.

### Task 8.1: Stress Testing (Days 50-52)
- Expand `scripts/load_test.py`: 1000 concurrent connections, test quota enforcement, test ONNX under sustained load
- Target: <50ms p95 latency, 1000 req/s sustained
- Security audit: bandit scan, verify no cross-tenant data leaks, CORS config, header redaction

### Task 8.2: Landing Page & Docs (Days 52-54)
- Replace `frontend/app/page.tsx` with marketing landing: hero, features, pricing table, "Start Free" CTA
- Ensure all API routes have OpenAPI descriptions
- Terms of Service + Privacy Policy (template, lawyer later)

### Task 8.3: Production Deploy (Days 54-56)
- Register product domain, configure DNS
- TLS via cert-manager + Let's Encrypt on K8s Ingress
- Set all production secrets in K8s
- Verify Grafana + alerts
- Smoke test: signup → domain → traffic → dashboard → billing
- Soft launch: invite 5-10 beta users

---

## Launch Checklist (Day 56)

- [ ] Signup → verify → login → dashboard works
- [ ] Domain onboarding: add → CNAME → verify → traffic flows
- [ ] Multi-tenancy: 2 orgs cannot see each other's data
- [ ] WAF: malicious requests scored > 70 and blocked
- [ ] Firewall for AI: prompt injection detected and blocked
- [ ] Billing: Stripe checkout works, plan shows in dashboard
- [ ] Quota: free tier enforced (429 after 100K requests)
- [ ] API keys: create, use, see usage
- [ ] Dashboard: real-time metrics, threat map, attack log
- [ ] Monitoring: Grafana shows volume, Prometheus alerts fire
- [ ] CI/CD: merge to main auto-deploys
- [ ] K8s: HPA scales gateway under load
- [ ] Latency: WAF inspection p95 < 50ms (ONNX)
- [ ] Load test: 1000 req/s sustained, no errors
- [ ] Security: bandit clean, no cross-tenant leaks
- [ ] Docs: quickstart, API reference, pricing page

---

## Critical Path

```
Week 1: Multi-tenancy (blocks everything)
  ↓
Week 2: Signup + Gateway Routing (blocks onboarding)
  ↓           ↓
Week 3: ML/ONNX (parallel)    Week 4: Feature Wiring (parallel)
  ↓                               ↓
Week 5: K8s + CI/CD (blocks deploy)
  ↓
Week 6: Observability + Onboarding UX
  ↓
Week 7: Billing (blocks monetization)
  ↓
Week 8: Stress Test + Launch
```

**If anything slips**: Multi-tenancy and billing are on the critical path. ML optimization and feature wiring can run in parallel.

## Key Files to Modify

| File | What Changes |
|------|-------------|
| `backend/models/users.py` | org_id, bcrypt passwords |
| `backend/auth.py` | org_id in JWT, get_current_org dependency |
| `gateway/main.py` | tenant routing, metering, quota enforcement |
| `backend/ml/waf_classifier.py` | ONNX inference backend |
| `backend/database.py` | Alembic, PostgreSQL pooling |
| `backend/config.py` | Stripe keys, billing config |
| All `backend/services/*.py` | org_id filtering |
| All `backend/routes/*.py` | org_id from JWT |
| `frontend/lib/auth.ts` | orgId in session |
| `.github/workflows/ci.yml` | Full CI/CD pipeline |
