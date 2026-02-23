# MEGA PLAN 2: Testing, Quality & CI/CD

**Focus:** Production-grade testing infrastructure and automated quality gates
**Completion lift:** 92% -> 97%
**Key metric:** Test coverage: ~60% -> 80%+
**Critical fix:** Automated quality gates
**Dependencies:** Mega Plan 1 (retrained model needed for accuracy gates)
**Estimated scope:** ~40 files modified/created

---

## Phase 2A — Unit Test Coverage (target: 80%+)

| Step | Action | Files |
|------|--------|-------|
| 1 | WAF classifier unit tests | `tests/unit/test_waf_classifier.py` — benign/malicious classification, batch inference, confidence scores, edge cases (empty input, oversized input, unicode) |
| 2 | IP fencing tests | `tests/unit/test_ip_fencing.py` — blacklist/whitelist matching, CIDR ranges, temporary blocks, Redis sync |
| 3 | Geo fencing tests | `tests/unit/test_geo_fencing.py` — country block/allow, exception IPs, unknown IPs |
| 4 | Bot detection tests | `tests/unit/test_bot_detection.py` — scoring, verified bot recognition, signature matching |
| 5 | Rate limiter tests | `tests/unit/test_rate_limiter.py` — sliding/fixed window, per-IP, burst detection |
| 6 | DDoS protection tests | `tests/unit/test_ddos_protection.py` — burst thresholds, request size limits, temp blocking |
| 7 | Threat intel tests | `tests/unit/test_threat_intel.py` — feed sync, indicator matching |
| 8 | Security rules tests | `tests/unit/test_security_rules.py` — rule CRUD, rule matching engine |
| 9 | Middleware tests | `tests/unit/test_waf_middleware.py`, `test_audit_middleware.py` — request interception, logging |
| 10 | Parsing/normalization tests | Expand `tests/unit/test_parsing.py` — edge cases in UUID/timestamp/token replacement |

---

## Phase 2B — Integration Tests

| Step | Action | Details |
|------|--------|---------|
| 1 | End-to-end request flow | Client -> Gateway -> WAF inspection -> Backend -> Response (benign + malicious) |
| 2 | Rate limit integration | Verify Redis-backed rate limiting works across gateway + backend |
| 3 | WebSocket live updates | Verify dashboard receives real-time events when attacks are blocked |
| 4 | Multi-tenant isolation | Verify tenant A cannot see tenant B's data/events |
| 5 | Database migration tests | Verify schema migrations work on fresh DB |

---

## Phase 2C — Performance Testing with Locust

| Step | Action | Details |
|------|--------|---------|
| 1 | Create `tests/performance/locustfile.py` | Define user scenarios: browsing, API calls, mixed traffic |
| 2 | Latency benchmarks | ML inference <100ms p95, API endpoints <200ms p95 |
| 3 | Throughput tests | Target >100 RPS per gateway instance |
| 4 | Concurrent user load | 100, 500, 1000 concurrent users |
| 5 | Document baselines | Create `reports/performance_baseline.md` |

---

## Phase 2D — CI/CD Pipeline

| Step | Action | Files |
|------|--------|-------|
| 1 | CI workflow | `.github/workflows/ci.yml` — lint (ruff), unit tests, integration tests, coverage report, model accuracy gate (>80%) |
| 2 | CD workflow | `.github/workflows/cd.yml` — Docker build, push to GHCR, staging deploy, smoke tests, prod deploy (manual gate) |
| 3 | Model test workflow | `.github/workflows/model-test.yml` — run attack payload suite, fail if accuracy <80% |
| 4 | Pre-commit hooks | `.pre-commit-config.yaml` — ruff, black, eslint, prettier, commit message lint |
| 5 | Dependabot | `.github/dependabot.yml` — weekly dependency updates for Python and npm |

---

## Phase 2E — API Documentation

| Step | Action | Details |
|------|--------|---------|
| 1 | Add Pydantic response models | Add typed response schemas to all 20+ route files |
| 2 | Add route descriptions | Summary + description for every FastAPI endpoint |
| 3 | Add request/response examples | `example` field on Pydantic models |
| 4 | Export OpenAPI spec | Generate `docs/openapi.json`, commit to repo |
| 5 | Verify Swagger UI | Ensure `/docs` and `/redoc` work correctly |

---

## Deliverables

- [ ] 80%+ unit test coverage
- [ ] Full integration test suite
- [ ] Locust performance benchmarks with documented baselines
- [ ] GitHub Actions CI/CD (3 workflows)
- [ ] Pre-commit hooks
- [ ] Complete OpenAPI documentation at `/docs`
