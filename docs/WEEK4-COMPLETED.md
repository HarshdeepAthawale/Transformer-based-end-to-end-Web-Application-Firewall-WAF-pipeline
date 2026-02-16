# Week 4: Inference Integration + Validation — Completed

> **Duration**: Week 4 of Phase 1
> **Goal**: Wire parsing + normalizer into live request path, fix Nginx integration, run accuracy tests
> **Status**: COMPLETE — Normalizer applied, standalone WAF service created, Nginx config updated, CI workflow added

---

## Summary

Completed six tasks to integrate the trained anomaly model into the live request path:

1. **Normalizer in WAF Classifier** — `check_request` and `check_request_async` now use `ParsingPipeline.process_dict()` to normalize requests before classification. Path and query string are split correctly to avoid duplication.

2. **Standalone WAF Service** — Created `integration/waf_service.py` as a FastAPI service on port 8000 for Nginx Lua integration. Supports model and placeholder modes.

3. **Start Script** — Fixed `start_waf_service.py` to use HuggingFace model dir (`models/waf-anomaly`) instead of legacy checkpoint. Falls back to placeholder when model is missing.

4. **Nginx Config** — Updated `scripts/nginx_waf.conf` so Lua sends `path` without query string (uses `ngx.var.uri`) when `query_params` is populated, avoiding serialization duplication.

5. **CI Workflow** — Added `.github/workflows/model-accuracy.yml` to run attack tests against the backend with an 80% detection rate gate when the model exists.

6. **Week 4 Doc** — This file.

---

## Tasks Completed

| # | Task | File | Description |
|---|------|------|-------------|
| 22 | Normalizer in classifier | `backend/ml/waf_classifier.py` | ParsingPipeline.process_dict() used in check_request and check_request_async |
| 23 | Standalone WAF service | `integration/waf_service.py` | FastAPI /check, /health, /metrics for Nginx |
| 24 | Fix start script | `scripts/start_waf_service.py` | HuggingFace model dir, no vocab_path, placeholder fallback |
| 25 | Nginx config | `scripts/nginx_waf.conf` | path = ngx.var.uri (no query duplication) |
| 26 | CI workflow | `.github/workflows/model-accuracy.yml` | 80% detection gate when model present |
| 27 | Week 4 doc | `weeks/week4/WEEK4-COMPLETED.md` | This file |

---

## Files Created / Updated

### Created
```
integration/
├── __init__.py
└── waf_service.py         # Standalone FastAPI service (port 8000)

.github/workflows/
└── model-accuracy.yml     # CI: attack tests, 80% gate

weeks/week4/
└── WEEK4-COMPLETED.md     # This file
```

### Updated
```
backend/ml/waf_classifier.py   # Normalizer via process_dict
scripts/start_waf_service.py   # models/waf-anomaly, --placeholder
scripts/nginx_waf.conf         # path = ngx.var.uri
scripts/attack_tests/run_all_tests.py  # --min-rate 80
config/config.yaml             # model_path: models/waf-anomaly
tests/integration/test_waf_service.py  # sys.path, --placeholder
```

---

## Pipeline Flow

```
Request (method, path, query_params, headers, body)
                    │
                    ▼
         ┌──────────────────────┐
         │ _to_process_dict     │  Split path?query → path + query_params
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ ParsingPipeline      │  parse → normalize → serialize
         │ process_dict         │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ WAFClassifier        │  Tokenize → DistilBERT → softmax
         │ classify             │
         └──────────┬───────────┘
                    │
                    ▼
         is_anomaly, anomaly_score, threshold, processing_time_ms
```

---

## Running the Standalone WAF Service

```bash
# With trained model
python scripts/start_waf_service.py

# Placeholder mode (no model, for testing)
python scripts/start_waf_service.py --placeholder

# Custom model path
python scripts/start_waf_service.py --model_path models/waf-anomaly --port 8000
```

---

## Nginx Integration

1. Start the WAF service: `python scripts/start_waf_service.py`
2. Use `scripts/nginx_waf.conf` with OpenResty (Nginx + Lua). The Lua block calls `http://127.0.0.1:8000/check` with `{method, path, query_params, headers, body}`.
3. On `is_anomaly: true`, Nginx returns 403. Otherwise, the request is proxied to `backend_app` (port 8080).

---

## Attack Tests

```bash
# Start backend (with model) on 3001
WAF_MODEL_PATH=models/waf-anomaly uvicorn backend.main:app --port 3001

# Run attack suite (default 60% min)
python scripts/attack_tests/run_all_tests.py

# CI gate: 80% minimum
python scripts/attack_tests/run_all_tests.py --min-rate 80
```

---

## CI Workflow

- **Trigger**: Push/PR to `main` or `master`, or `workflow_dispatch`
- **Model check**: Looks for `models/waf-anomaly` with `config.json`, `tokenizer.json`, and model weights.
- **If model exists**: Starts backend with `WAF_MODEL_PATH=models/waf-anomaly`, runs attack tests, fails if detection rate < 80%.
- **If model missing**: Skips accuracy tests with a notice; workflow passes.

---

## Dependencies Unlocked for Week 5

| Component | What's Unlocked |
|-----------|----------------|
| **Continuous Learning** | Pipeline complete; can add incremental fine-tuning on new benign traffic |
| **Dashboard** | WAF metrics and anomaly counts available via backend |
| **Hardening** | CI accuracy gate in place; can expand test coverage |

---

## Phase 1 Milestone

- **Full pipeline**: Logs → Parse → Normalize → Tokenize → Model → Anomaly score
- **Trained** on benign-only data from 3 apps (Week 3)
- **Live detection** via standalone WAF service (port 8000) or in-process backend (port 3001)
- **Nginx integration** verified (Lua → WAF service → allow/block)
- **CI** model accuracy gate at 80%
