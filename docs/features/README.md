# WAF Feature Specifications (Cloudflare-Competing)

End-to-end fullstack build specs for 10 features. Each spec is **no mock / no hardcoded**: config and env for all tunables, real DB and external services, real API contracts.

| # | Spec | Description |
|---|------|-------------|
| 1 | [01-waf-attack-score.md](01-waf-attack-score.md) | WAF Attack Score: single ML score (0–100) per request, configurable block/challenge thresholds, gateway events and dashboard display. |
| 2 | [02-bot-management.md](02-bot-management.md) | Bot Management: bot score (1–99), verified bots list, score bands (allow/challenge/block), gateway enforcement and frontend CRUD. |
| 3 | [03-managed-rulesets.md](03-managed-rulesets.md) | Managed Rulesets: OWASP CRS (or feed) storage, versioning, enable/disable per pack, sync job and gateway evaluation. |
| 4 | [04-malicious-upload-scanning.md](04-malicious-upload-scanning.md) | Malicious Upload Scanning: ClamAV or cloud scan API, config-driven engine and policy, block/quarantine/log and event reporting. |
| 5 | [05-firewall-for-ai.md](05-firewall-for-ai.md) | Firewall for AI: LLM endpoint labeling, prompt-injection and PII detection, abuse rate limiting, configurable patterns and actions. |
| 6 | [06-adaptive-ddos.md](06-adaptive-ddos.md) | Adaptive DDoS: baseline learning from traffic, percentile-based auto-tune of burst threshold, Redis and gateway integration. |
| 7 | [07-credential-leak-protection.md](07-credential-leak-protection.md) | Credential Leak Protection: HIBP k-anonymity API, login-path check, block or flag, no raw password storage. |
| 8 | [08-unified-security-dashboard.md](08-unified-security-dashboard.md) | Unified Security Dashboard: single view aggregating WAF, rate limit, DDoS, bot, upload scan, credential leak, firewall_ai metrics and events. |
| 9 | [09-waf-api.md](09-waf-api.md) | WAF/Bot API: REST CRUD for rules, bot policies, rate limits; auth; optional multi-zone/tenant. |
| 10 | [10-alerting-metrics.md](10-alerting-metrics.md) | Alerting and Metrics: Prometheus /metrics export, configurable webhooks and alert rules, notification_service integration. |

## Document structure (each spec)

1. **Overview** — What the feature does.
2. **Objectives** — Backend, gateway, frontend, config deliverables.
3. **Architecture** — Mermaid diagram.
4. **Configuration (no hardcoding)** — Env vars and config keys.
5. **Backend** — Modules, routes, services, DB, schemas.
6. **Gateway** — Changes and new modules.
7. **Frontend** — Pages, components, API client.
8. **Data Flow** — Step-by-step and optional diagram.
9. **External Integrations** — Real services and contracts.
10. **Database** — Tables, migrations.
11. **Testing** — How to verify end-to-end without mocks.

## Key code references

- Backend: [backend/main.py](../../backend/main.py), [backend/config.py](../../backend/config.py), [backend/ml/waf_classifier.py](../../backend/ml/waf_classifier.py), [backend/routes/events.py](../../backend/routes/events.py), [backend/services/bot_detection.py](../../backend/services/bot_detection.py).
- Gateway: [gateway/main.py](../../gateway/main.py), [gateway/config.py](../../gateway/config.py), [gateway/waf_inspect.py](../../gateway/waf_inspect.py), [gateway/events.py](../../gateway/events.py).
- Frontend: [frontend/app/dashboard/page.tsx](../../frontend/app/dashboard/page.tsx), [frontend/lib/api.ts](../../frontend/lib/api.ts).
- Config: [.env.example](../../.env.example), [config/production.yaml](../../config/production.yaml).
