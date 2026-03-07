# AI-WAF B2B Startup Roadmap: Weeks-Based Build Plan

> **Team**: 2 developers | **Goal**: Build a Cloudflare-competing AI-powered WAF for B2B customers (SMBs, mid-market, enterprise)

---

## Executive Summary

This roadmap transforms the current WAF pipeline into a **B2B SaaS product** over 16 weeks, split into 4 phases. Phase 1–2 build the core pipeline and enterprise-ready platform. Phase 3–4 add Cloudflare-differentiating features for product-market fit.

**Competitive Thesis**: Cloudflare dominates at scale (32M req/s) with global edge. You compete by: **self-hosted + on-prem control**, **per-app AI models**, **transparent ML**, **developer-first UX**, and **pricing for SMBs** who can’t afford Enterprise plans.

---

## Phase Overview

| Phase | Weeks | Focus | Key Deliverables |
|-------|-------|-------|------------------|
| 1 | 1–4 | Core pipeline | Log ingestion, parsing, benign training, anomaly model |
| 2 | 5–8 | B2B platform | Nginx integration, continuous learning, dashboard, CI/CD |
| 3 | 9–12 | Cloudflare differentiation | Self-hosted, per-app models, API, transparent pricing |
| 4 | 13–16 | Growth & scale | Multi-tenant, performance, developer portal |

---

# Phase 1: Foundation (Weeks 1–4)

## Week 1: Log Ingestion Pipeline

**Owner**: Dev A | **Support**: Dev B

### Goals
- Batch and streaming ingestion from Apache/Nginx access logs
- Format auto-detection (Common, Combined, Custom)
- Queue for downstream processing

### Tasks
- [ ] Create `backend/ingestion/` module
- [ ] Batch reader: chunk-based file reading, configurable path
- [ ] Stream tailer: file watcher (watchdog/inotify) for live logs
- [ ] Log format detector (Apache Common/Combined, Nginx Combined)
- [ ] Async queue (asyncio.Queue or Redis for scale)
- [ ] Config integration in `config/config.yaml`
- [ ] Unit tests for format detection

### Output
- `backend/ingestion/batch_reader.py`
- `backend/ingestion/stream_tailer.py`
- `backend/ingestion/format_detector.py`
- `backend/ingestion/__init__.py`

---

## Week 2: Parsing & Normalization

**Owner**: Dev B | **Support**: Dev A

### Goals
- Parse log lines → structured `HTTPRequest`
- Normalize dynamic values (UUIDs, timestamps, session IDs, IPs)
- Serialize for tokenizer input

### Tasks
- [ ] Create `backend/parsing/` module
- [ ] Log parser: Apache/Nginx regex patterns, extract method/path/params/headers/body
- [ ] Normalization rules: placeholders for `{UUID}`, `{TIMESTAMP}`, `{ID}`, `{IP}`
- [ ] Request serializer: compact string format for model
- [ ] Pipeline: `parse → normalize → serialize`
- [ ] Unit tests with sample log lines from 3 apps

### Output
- `backend/parsing/log_parser.py`
- `backend/parsing/normalizer.py`
- `backend/parsing/pipeline.py`
- `backend/parsing/__init__.py`

---

## Week 3: Training Data + Model

**Owner**: Dev A | **Support**: Dev B

### Goals
- Generate synthetic benign logs from Juice Shop, WebGoat, DVWA
- Train anomaly model on benign-only data
- Replace HuggingFace dataset with own pipeline

### Tasks
- [ ] `scripts/generate_training_data.py`:
  - Selenium/Playwright or requests-based crawler for 3 apps
  - Record access logs (proxy or app log collection)
  - Output: `data/training/benign_requests.json` (10k+ per app)
- [ ] Integrate ingestion + parsing into data generation
- [ ] Anomaly training: MSE loss, target=0 for all; or One-Class / reconstruction loss
- [ ] Update `finetune_waf_model.py` or create `scripts/train_anomaly_model.py`
- [ ] Vocabulary from training data (or keep HF tokenizer with custom prep)
- [ ] Validate: benign→low score, malicious→high score on test set

### Output
- `scripts/generate_training_data.py`
- `scripts/train_anomaly_model.py`
- `data/training/benign_requests.json`
- Trained model in `models/checkpoints/` or `models/waf-anomaly/`

---

## Week 4: Inference Integration + Validation

**Owner**: Dev B | **Support**: Dev A

### Goals
- Wire parsing + normalizer into live request path
- Fix Nginx integration (standalone WAF service)
- Run accuracy tests

### Tasks
- [ ] Update `waf_classifier.py`: apply normalizer to request before model
- [ ] Fix or replace `scripts/start_waf_service.py`: standalone FastAPI on port 8000 using `waf_classifier`
- [ ] Nginx config: proxy to WAF service 8000, then backend on allow
- [ ] Run `scripts/attack_tests/run_all_tests.py`, measure detection rate
- [ ] Tune threshold for <5% false positive target
- [ ] CI: add model accuracy gate (e.g., >80% detection)

### Output
- Working standalone WAF service
- Nginx integration verified
- Accuracy report
- CI workflow for model tests

### Phase 1 Milestone
- Full pipeline: Logs → Parse → Normalize → Tokenize → Model → Anomaly score
- Trained on benign-only data from 3 apps
- Live detection via Nginx or in-process middleware

---

# Phase 2: B2B Platform (Weeks 5–8)

## Week 5: Continuous Learning

**Owner**: Dev A | **Support**: Dev B

### Goals
- Incremental fine-tuning on new benign traffic
- No full retrain; periodic jobs

### Tasks
- [ ] `backend/learning/` (or `backend/ml/learning/`):
  - `data_collector.py`: collect new logs since last run
  - `fine_tuner.py`: fine-tune on incremental data (low LR, few epochs)
  - `version_manager.py`: model versions, rollback
- [ ] API: `POST /api/admin/trigger-retrain` or cron job
- [ ] Scheduler in backend startup (optional)
- [ ] Validation before deploying new model

### Output
- Incremental retraining pipeline
- Admin API or cron for updates

---

## Week 6: Dashboard & Observability

**Owner**: Dev B | **Support**: Dev A

### Goals
- Production-ready dashboard for B2B customers
- WAF metrics, blocked requests, anomaly distribution

### Tasks
- [ ] Dashboard: real-time blocked count, top attack types, false positive feedback
- [ ] Alerts: Slack/email on high anomaly rate
- [ ] Audit log export (CSV/JSON)
- [ ] Mobile-responsive layout
- [ ] Onboarding: “Add your first app” flow

### Output
- Polished dashboard for early B2B customers
- Alerting integration

---

## Week 7: Hardening & Documentation

**Owner**: Both (split)

### Goals
- CI/CD, tests, docs
- Enterprise deployment readiness

### Tasks
- [ ] GitHub Actions: CI (lint, test, model accuracy)
- [ ] CD: Docker build on release
- [ ] DoS pattern tests (expand `09_dos_patterns.py`)
- [ ] Header injection improvement (expand payloads, retrain)
- [ ] API docs (OpenAPI/Swagger)
- [ ] Deployment runbook for customers
- [ ] 1-pager: architecture, features, Cloudflare comparison

### Output
- CI/CD pipeline
- Customer-facing docs and runbook

---

## Week 8: Beta Customer Readiness

**Owner**: Both

### Goals
- First beta customers can deploy and evaluate
- Sales and onboarding materials ready

### Tasks
- [ ] End-to-end validation run (WAF + dashboard + Nginx)
- [ ] Test with diverse malicious payloads (SQLi, XSS, path traversal, etc.)
- [ ] Threshold tuning for production workloads
- [ ] Customer onboarding checklist
- [ ] Pitch deck (problem, solution, differentiation, roadmap for investors)

### Phase 2 Milestone
- Beta-ready platform for B2B pilots
- Continuous learning in place
- Sales and onboarding materials complete

---

# Phase 3: Cloudflare Differentiation (Weeks 9–12)

## Competitive Strategy vs Cloudflare

| Dimension | Cloudflare | Your Startup |
|-----------|------------|--------------|
| **Deployment** | Cloud-only, DNS change | Self-hosted, on-prem, no DNS change |
| **Data** | Traffic sent to CF | Traffic stays in customer infra |
| **Model** | One global model | Per-app custom model (learns your traffic) |
| **Transparency** | Black-box ML | Explainable scores, logs, feedback |
| **Pricing** | Enterprise-heavy | SMB-friendly, usage-based |
| **Customization** | Limited on lower tiers | Full control, dev-first API |

---

## Week 9: Self-Hosted & On-Prem

**Owner**: Dev A

### Goals
- One-command deploy (Docker Compose, K8s optional)
- No external deps for core WAF
- Data residency: everything on customer side

### Tasks
- [ ] Single `docker-compose.yml` for WAF + dashboard + optional Nginx
- [ ] Env-based config (no hardcoded secrets)
- [ ] Health checks, graceful shutdown
- [ ] Docs: “Deploy in 5 minutes”
- [ ] Optional: Helm chart for K8s

### Output
- Production-ready Docker deployment
- “Self-hosted WAF” marketing angle

---

## Week 10: Per-App Models

**Owner**: Dev B | **Support**: Dev A

### Goals
- Each customer app gets its own model (or model variant)
- Model learns that app’s benign patterns
- Better accuracy, fewer false positives

### Tasks
- [ ] Multi-model support: `model_id` or `app_id` in config
- [ ] Training per app: `train --app juice-shop`
- [ ] Model registry: store/load by app
- [ ] Dashboard: switch between apps, compare metrics
- [ ] API: `GET /api/apps/{id}/model-info`

### Output
- Per-app model training and inference
- Clear differentiator vs “one size fits all”

---

## Week 11: Developer API & SDK

**Owner**: Dev A

### Goals
- REST API for WAF check, config, feedback
- SDK (Python, curl examples)
- Webhooks for alerts

### Tasks
- [ ] Public API: `/v1/check`, `/v1/config`, `/v1/feedback`
- [ ] API keys, rate limits
- [ ] Webhooks: `waf.alert`, `waf.blocked`
- [ ] Python SDK: `from aiwaf import WAFClient`
- [ ] Postman/OpenAPI collection
- [ ] Feedback endpoint: mark false positive/negative for retraining

### Output
- Developer-friendly API
- Feedback loop for model improvement

---

## Week 12: Transparent Pricing & Free Tier

**Owner**: Dev B

### Goals
- Transparent pricing page
- Free tier for SMBs (e.g., <100k req/month)
- Usage-based upgrades

### Tasks
- [ ] Pricing page: Free / Pro / Team
- [ ] Free: 1 app, 50k req/month, community support
- [ ] Pro: 3 apps, 500k req/month, email support
- [ ] Team: Unlimited, SSO, SLA
- [ ] Usage metering: requests, blocked count, model retrains
- [ ] Billing integration (Stripe) or “Coming soon”

### Phase 3 Milestone
- Self-hosted, per-app models, dev API
- Transparent pricing and free tier
- Clear positioning vs Cloudflare

---

# Phase 4: Growth & Scale (Weeks 13–16)

## Week 13: Multi-Tenant (SaaS Path)

**Owner**: Dev A

### Tasks
- [ ] Tenant isolation (org_id, api_key)
- [ ] Per-tenant DB schemas or schema-level isolation
- [ ] Admin: org management, user invites
- [ ] Billing: per-tenant usage aggregation

---

## Week 14: Performance & Scale

**Owner**: Dev B

### Tasks
- [ ] Model quantization (ONNX, TensorRT, or PyTorch)
- [ ] Batch inference optimization
- [ ] Caching: repeated requests
- [ ] Horizontal scaling: multiple WAF workers
- [ ] Benchmark: req/s, p99 latency

---

## Week 15: Firewall for AI (Differentiator)

**Owner**: Dev A

### Tasks
- [ ] Prompt injection detection (LLM-focused attacks)
- [ ] AI scrapers / bot detection
- [ ] “Firewall for AI” landing page (similar to Cloudflare beta)
- [ ] Partner angle: protect AI apps, APIs

---

## Week 16: Launch Prep

**Owner**: Both

### Tasks
- [ ] Landing page, blog, docs site
- [ ] Product Hunt / HN launch plan
- [ ] Beta signup flow
- [ ] Security audit checklist
- [ ] Support: Discord, email

---

# Feature Roadmap (Post–Week 16)

| Phase | Feature | Description |
|-------|---------|-------------|
| 5 | Bot score | ML-based bot detection (Cloudflare has this) |
| 5 | Rate limiting | Per-IP, per-path, token bucket |
| 5 | DDoS mitigation | L3/L4 + L7 collaboration |
| 6 | Content scanning | Malware in uploads (like Cloudflare) |
| 6 | Leaked credentials | Check passwords against breach DBs |
| 6 | API discovery | Auto-detect and protect undocumented APIs |
| 7 | Geo-fencing | Allow/block by country (you have some of this) |
| 7 | IP reputation | Threat intel feed integration |
| 8 | Compliance | SOC2, GDPR documentation |
| 8 | Enterprise | SSO, audit logs, SLA |

---

# Two-Person Split Recommendation

| Role | Primary focus |
|------|---------------|
| **Dev A** | ML pipeline, ingestion, parsing, training, per-app models, infra |
| **Dev B** | WAF middleware, dashboard, API, SDK, pricing, docs |
| **Both** | CI/CD, deployment, beta launches, investor outreach |

**Suggested cadence**: 2-week sprints, sync on Mondays, sprint review on Friday of week 2.

---

# Success Metrics

| Phase | Metric |
|-------|--------|
| 1 | Pipeline complete, anomaly model trained, Nginx integration working |
| 2 | First beta customers onboarded, detection rate >85%, false positive <5% |
| 3 | First 10 beta users, self-hosted deploy <10 min |
| 4 | 100+ req/s sustained, p99 <50ms |
| 5+ | MRR, NPS, churn |

---

# References

- [WAF Pipeline Compliance Assessment](.cursor/plans/problem_statement_compliance_assessment_6802ed3f.plan.md)
- [Cloudflare WAF](https://www.cloudflare.com/application-services/products/waf/)
- [Cloudflare AI WAF Blog](https://blog.cloudflare.com/waf-ml/)
