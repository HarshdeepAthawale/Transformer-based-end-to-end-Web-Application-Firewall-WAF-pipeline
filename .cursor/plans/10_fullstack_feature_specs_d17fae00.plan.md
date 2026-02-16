---
name: 10 Fullstack Feature Specs
overview: Create 10 detailed fullstack markdown specifications in docs/features/ for the Cloudflare-competing features, each end-to-end (gateway, backend, frontend, config, data) with no mocks or hardcoded values.
todos: []
isProject: false
---

# Plan: 10 In-Detail Fullstack Feature Specification Documents

## Scope

Produce **10 markdown files** under **[docs/features/](docs/features/)**. Each document is a **fullstack build spec**: gateway, backend (FastAPI), frontend (Next.js), config/env, and data flows. All specs must be **end-to-end** and **no mock / no hardcoded**: config and env vars for all tunables, real DB tables/schemas where applicable, real external service integration (e.g. HIBP API, ClamAV, OWASP rule sources), and real API contracts between gateway, backend, and frontend.

## Document Naming and Location


| #   | Filename                           | Feature                                                                       |
| --- | ---------------------------------- | ----------------------------------------------------------------------------- |
| 1   | `01-waf-attack-score.md`           | WAF Attack Score (single ML score per request, configurable thresholds)       |
| 2   | `02-bot-management.md`             | Bot Score + Verified Bots (scoring, allowlist, challenge/block actions)       |
| 3   | `03-managed-rulesets.md`           | Managed Rulesets (OWASP CRS + auto-update from external source)               |
| 4   | `04-malicious-upload-scanning.md`  | Malicious Upload Scanning (ClamAV or cloud scan API, config-driven)           |
| 5   | `05-firewall-for-ai.md`            | Firewall for AI (LLM endpoint labeling, prompt/PII detection, abuse patterns) |
| 6   | `06-adaptive-ddos.md`              | Adaptive DDoS (learn baseline from traffic, auto-tune thresholds)             |
| 7   | `07-credential-leak-protection.md` | Credential Leak Protection (HIBP API or similar, safe handling)               |
| 8   | `08-unified-security-dashboard.md` | Unified Security Dashboard and Analytics (single view, real metrics)          |
| 9   | `09-waf-api.md`                    | WAF/Bot API (API-first rules, bot policies, multi-tenant/zone support)        |
| 10  | `10-alerting-metrics.md`           | Alerting and Metrics (Prometheus export, webhooks, configurable alerts)       |


## Standard Structure for Each Document

Every spec will follow the same sections so implementers get a consistent, fullstack blueprint. Structure (align with existing [docs/phase8-continuous-learning.md](docs/phase8-continuous-learning.md) and [docs/phase6-waf-integration.md](docs/phase6-waf-integration.md)):

1. **Overview** — What the feature does and how it fits the “beat Cloudflare” story.
2. **Objectives** — Bullet list of concrete deliverables (backend, gateway, frontend, config).
3. **Architecture** — Mermaid diagram: request flow and components (gateway, backend, frontend, DB, external services).
4. **Configuration (no hardcoding)** — Env vars and config keys (reference [.env.example](.env.example), [gateway/config.py](gateway/config.py), [backend/config.py](backend/config.py)). All thresholds, URLs, API keys, feature flags come from config.
5. **Backend** — Modules, routes, services, DB schema (new tables/migrations if any). Real file paths under `backend/`. Request/response schemas; no mock responses.
6. **Gateway** — Changes in [gateway/main.py](gateway/main.py) and new/updated modules under `gateway/`. How the feature is applied on the request path (e.g. attack score, bot score, upload scan).
7. **Frontend** — New or updated pages under `frontend/app/`, components under `frontend/components/`, API client usage in [frontend/lib/api.ts](frontend/lib/api.ts). Real endpoints and data shapes.
8. **Data Flow** — Step-by-step flow (client → gateway → backend → DB/external → response). Optional sequence or flowchart.
9. **External Integrations** — Real services (e.g. HIBP, ClamAV, OWASP rule feed). API contracts, auth, rate limits, and config keys.
10. **Database** — New or modified tables, columns, indexes. Migration approach (e.g. Alembic or SQL scripts).
11. **Testing** — How to verify end-to-end (no mocks): integration tests, env for external services, sample payloads.

## Feature-Specific Highlights (What Each Doc Must Specify)

- **01-waf-attack-score**: Expose [backend/ml/waf_classifier.py](backend/ml/waf_classifier.py) confidence as a numeric “attack score” (e.g. 0–100). Gateway sends score to backend events; configurable block/challenge thresholds via env; frontend shows score in request/event details.
- **02-bot-management**: Extend [backend/services/bot_detection.py](backend/services/bot_detection.py) (and [backend/models/bot_signatures.py](backend/models/bot_signatures.py)) with numeric bot score, verified-bots list source (DB + optional sync from URL), and action (allow/challenge/block) by score band. Gateway calls backend or local bot-scoring; frontend [frontend/app/bot-detection/page.tsx](frontend/app/bot-detection/page.tsx) for score bands and verified bots CRUD.
- **03-managed-rulesets**: Rule storage (DB), rule pack source (OWASP CRS or documented feed URL), versioning, enable/disable per pack. Backend job to fetch and parse rules; gateway evaluates rules (or backend as sidecar). Config for feed URL and update interval.
- **04-malicious-upload-scanning**: Config for scan engine (ClamAV socket/daemon or cloud API URL + key). Gateway or backend stream/forward upload to scanner; block or quarantine by policy. No hardcoded paths or mock “virus found.”
- **05-firewall-for-ai**: DB or config for “LLM endpoints” (path/label); optional PII detection (regex or external); prompt-injection patterns. Gateway or backend tags requests to labeled routes and applies logic; config for feature flags and pattern sources.
- **06-adaptive-ddos**: Baseline learning from real traffic (e.g. Redis or DB aggregates by time window). Config for learning window, percentile for “normal” rate, and how to auto-tune [gateway/ddos_protection.py](gateway/ddos_protection.py) (or equivalent) thresholds. No fixed magic numbers.
- **07-credential-leak-protection**: HIBP (or equivalent) API: k-anonymity, env for API key and endpoint. Backend endpoint or gateway-side call on login path; block or flag only; no storage of raw passwords in spec.
- **08-unified-security-dashboard**: Single dashboard page aggregating WAF blocks, rate-limit, DDoS, bot score, attack score from existing [backend/routes/events.py](backend/routes/events.py), [backend/routes/metrics.py](backend/routes/metrics.py), and new metrics. Real API contracts and time-range filters; no mock charts.
- **09-waf-api**: REST API for rules, bot policies, rate limits (CRUD). Auth (e.g. [backend/auth.py](backend/auth.py)), multi-zone/tenant if applicable (DB schema for scope). Document endpoints and request/response bodies.
- **10-alerting-metrics**: Prometheus metrics export (format and labels); webhook URL(s) from config; alert rules (e.g. block rate &gt; X). Backend or dedicated exporter; [backend/services/notification_service.py](backend/services/notification_service.py) / [backend/services/alert_service.py](backend/services/alert_service.py) for webhooks. No hardcoded webhook URLs or thresholds.

## Key Files to Reference in Specs

- **Backend**: [backend/main.py](backend/main.py), [backend/config.py](backend/config.py), [backend/ml/waf_classifier.py](backend/ml/waf_classifier.py), [backend/core/waf_factory.py](backend/core/waf_factory.py), [backend/routes/events.py](backend/routes/events.py), [backend/services/bot_detection.py](backend/services/bot_detection.py), [backend/database.py](backend/database.py).
- **Gateway**: [gateway/main.py](gateway/main.py), [gateway/config.py](gateway/config.py), [gateway/waf_inspect.py](gateway/waf_inspect.py), [gateway/rate_limit.py](gateway/rate_limit.py), [gateway/ddos_protection.py](gateway/ddos_protection.py), [gateway/blacklist.py](gateway/blacklist.py), [gateway/events.py](gateway/events.py).
- **Frontend**: [frontend/app/dashboard/page.tsx](frontend/app/dashboard/page.tsx), [frontend/app/bot-detection/page.tsx](frontend/app/bot-detection/page.tsx), [frontend/lib/api.ts](frontend/lib/api.ts).
- **Config**: [.env.example](.env.example), [config/production.yaml](config/production.yaml).

## Deliverable

- **10 new files**: `docs/features/01-waf-attack-score.md` through `docs/features/10-alerting-metrics.md`.
- **Optional**: `docs/features/README.md` — short index of the 10 features with one-line descriptions and links to each spec.

No code or config changes in this plan; only creation of the markdown specification documents.