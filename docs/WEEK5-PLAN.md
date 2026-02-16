# Week 5: CI/CD + Unit Tests + Continuous Learning Foundation — Plan

> **Duration**: Week 5 of Phase 1
> **Goal**: Establish full CI pipeline, expand unit test coverage, and lay groundwork for continuous learning
> **Status**: PLAN

---

## Context

**Week 4 completed** the inference integration: normalizer in classifier, standalone WAF service, Nginx config, and model-accuracy CI workflow. The full pipeline (Logs → Parse → Normalize → Model → Decision) is operational.

**Week 5** focuses on **quality, automation, and long-term maintainability**:
1. Full CI pipeline (lint, test, coverage) — enables faster iteration
2. Unit tests for core services — WAF classifier, parsing, integration
3. Continuous learning foundation — feedback endpoint + data collection

---

## Objectives

| # | Objective | Priority | Est. Effort |
|---|-----------|----------|-------------|
| 1 | Main CI Pipeline (ci.yml) | 🔴 High | Medium |
| 2 | Unit Tests for Core Services | 🔴 High | Medium |
| 3 | Continuous Learning: Feedback Endpoint | 🟠 Medium | Low |
| 4 | Header Injection Payload Expansion | 🟠 Medium | Low |
| 5 | Coverage Reporting | 🟡 Low | Low |

---

## Task Breakdown

### 1. Main CI Pipeline (`.github/workflows/ci.yml`)

**Objective**: Every PR/push runs linting, unit tests, and integration tests before merge.

| Task | Description |
|------|-------------|
| 1.1 | Trigger on push/PR to `main`, `master`, `develop` |
| 1.2 | Set up Python 3.10/3.11 |
| 1.3 | Install deps from `requirements.txt` |
| 1.4 | Run ruff/flake8 (lint) |
| 1.5 | Run unit tests via pytest |
| 1.6 | Run integration tests (`test_waf_service`, `test_ingestion_e2e`) |
| 1.7 | Generate coverage report (`pytest --cov`) |
| 1.8 | Optional: Upload coverage to Codecov |
| 1.9 | Keep `model-accuracy.yml` as separate job (runs when model exists) |

**Files**:
- Create: `.github/workflows/ci.yml`
- Optional: `pyproject.toml` or `.ruff.toml` for lint config

---

### 2. Unit Tests for Core Services

**Objective**: 80%+ coverage for WAF classifier, parsing pipeline, and key integration paths.

| Task | File | Description |
|------|------|-------------|
| 2.1 | `tests/unit/test_waf_classifier.py` | Benign vs malicious classification, batch inference, placeholder mode, threshold behavior |
| 2.2 | `tests/unit/test_parsing_pipeline.py` | Parse → normalize → serialize, edge cases (empty body, malformed), `process_dict` |
| 2.3 | `tests/unit/test_ingestion.py` | (Existing) Verify still passes; add edge cases if needed |
| 2.4 | `tests/integration/test_waf_service.py` | (Existing) Ensure placeholder + real-model paths covered |
| 2.5 | `tests/unit/test_utils.py` | Helper functions, config loading |

**Structure**:
```
tests/
├── unit/
│   ├── test_waf_classifier.py    # NEW / EXPAND
│   ├── test_parsing_pipeline.py   # NEW (or expand test_parsing.py)
│   ├── test_parsing.py            # Existing
│   ├── test_ingestion.py          # Existing
│   └── test_utils.py             # NEW
└── integration/
    ├── test_waf_service.py       # Existing
    └── test_ingestion_e2e.py     # Existing
```

---

### 3. Continuous Learning: Feedback Endpoint

**Objective**: Backend endpoint to collect labeled samples (false positive / false negative) for future retraining.

| Task | Description |
|------|-------------|
| 3.1 | Add `POST /api/waf/feedback` to backend |
| 3.2 | Payload: `{request_data, label: "benign"|"malicious", source, notes}` |
| 3.3 | Store in SQLite (or existing DB) via new table `waf_feedback` |
| 3.4 | Add export script: `scripts/export_feedback.py` → JSON for training |
| 3.5 | Document in API / Phase 8 notes |

**Schema (minimal)**:
```sql
CREATE TABLE waf_feedback (
  id INTEGER PRIMARY KEY,
  request_text TEXT,
  label TEXT,  -- 'benign' | 'malicious'
  source TEXT, -- 'manual' | 'dashboard' | 'api'
  notes TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

### 4. Header Injection Payload Expansion

**Objective**: Increase header injection detection (currently 3.3%). Expand payloads in training data and test suite.

| Task | Description |
|------|-------------|
| 4.1 | Add 20+ CRLF injection payloads to `tests/payloads/` or `scripts/attack_tests/07_header_injection.py` |
| 4.2 | Add HTTP response splitting patterns |
| 4.3 | Add header smuggling patterns (Transfer-Encoding, Content-Length tricks) |
| 4.4 | Include new payloads in `train_anomaly_model.py` malicious dataset |
| 4.5 | Re-run training with augmented data (follow-up week if needed) |

**Payload sources**:
- PayloadsAllTheThings (CRLF Injection)
- SecLists Fuzzing
- OWASP header injection examples

---

### 5. Coverage Reporting

| Task | Description |
|------|-------------|
| 5.1 | Add `pytest-cov` to `requirements.txt` if missing |
| 5.2 | Add `[tool.pytest]` / coverage config to `pyproject.toml` or `pytest.ini` |
| 5.3 | CI fails if coverage drops below 60% (or 70% for critical paths) |
| 5.4 | Generate `htmlcov/` for local inspection |

---

## Files to Create / Update

### Create
```
.github/workflows/
└── ci.yml                      # Main CI: lint, unit, integration, coverage

tests/unit/
├── test_waf_classifier.py      # WAFClassifier unit tests
├── test_parsing_pipeline.py   # ParsingPipeline unit tests (or expand test_parsing)
└── test_utils.py              # Utility tests

backend/
└── (add feedback endpoint + migration/table)
scripts/
└── export_feedback.py         # Export feedback for training
```

### Update
```
requirements.txt                # pytest-cov if missing
scripts/attack_tests/07_header_injection.py  # More CRLF payloads
scripts/train_anomaly_model.py  # Include new header payloads
pyproject.toml / pytest.ini     # Coverage config
```

---

## Success Criteria

| Criterion | Target |
|-----------|--------|
| CI runs on every PR | ✅ |
| Lint passes (ruff/flake8) | ✅ |
| Unit tests pass | ✅ |
| Integration tests pass | ✅ |
| Coverage | ≥ 60% |
| Feedback endpoint functional | ✅ |
| Header injection payloads added | 20+ |

---

## Dependencies Unlocked for Week 6

| Component | What's Unlocked |
|-----------|----------------|
| **Retraining** | `export_feedback.py` → merge with benign/malicious → retrain |
| **Dashboard** | Can wire "Report FP/FN" button to feedback endpoint |
| **Header Injection** | Augmented payloads ready for model retraining |
| **Quality Gates** | CI blocks merge on failing tests |

---

## Execution Order

```
Day 1–2: CI Pipeline (ci.yml) + fix any lint/test failures
Day 3:    Unit tests (WAF classifier, parsing pipeline)
Day 4:    Feedback endpoint + export script
Day 5:    Header injection payload expansion + coverage config
```

---

## Notes

- **model-accuracy.yml** stays separate; it runs attack tests when model exists. Main CI runs even without model.
- **Placeholder mode**: Unit tests should work with `--placeholder` when no model file present.
- **Header injection**: Full fix may require model retraining; Week 5 prepares payloads and data pipeline.
- **Continuous learning**: Phase 8 doc describes full architecture; Week 5 implements the data collection layer only.

---

*Created: 2026-02-15*
*Project: Transformer-based End-to-End WAF Pipeline*
