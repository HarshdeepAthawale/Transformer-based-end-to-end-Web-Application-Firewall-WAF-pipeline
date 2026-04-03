# Scripts Directory

This directory contains operational and testing scripts for the WAF project.

## Quick Reference

### Startup & Shutdown

| Script | Purpose | Usage |
|--------|---------|-------|
| `start_all.sh` | Start all services (docker-compose) | `./start_all.sh` |
| `stop_all.sh` | Stop all services | `./stop_all.sh` |

### Setup Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup_nginx_waf_advanced.sh` | Setup Nginx with advanced WAF integration | `./setup_nginx_waf_advanced.sh` |
| `setup_openresty_arch.sh` | Setup OpenResty (Nginx + Lua) for specialized WAF | `./setup_openresty_arch.sh` |

### Testing Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_waf_integration.py` | **Primary** - Comprehensive integration tests | `python3 test_waf_integration.py` |
| `test_waf_200_requests_simple.py` | Load testing (200 concurrent requests) | `python3 test_waf_200_requests_simple.py` |
| `quick_waf_test.py` | Quick smoke test for development | `python3 quick_waf_test.py` |
| `attack_tests/run_all_tests.py` | **Comprehensive** - Run all 10 attack test suites | `cd attack_tests && python3 run_all_tests.py` |

### Attack Test Suites (`attack_tests/`)

Comprehensive attack simulation with 453 total test payloads across 10 categories:

| Test Suite | File | Payloads | Coverage |
|------------|------|----------|----------|
| SQL Injection | `01_sql_injection.py` | 39 | Classic, UNION, Blind, Time-based, NoSQL |
| XSS | `02_xss_attacks.py` | 57 | Reflected, Stored, DOM, Filter bypass |
| Command Injection | `03_command_injection.py` | 47 | Shell, OS commands, encoding bypass |
| Path Traversal | `04_path_traversal.py` | 60 | LFI, RFI, encoding variations |
| XXE | `05_xxe_attacks.py` | 27 | External entities, SSRF via XXE |
| SSRF | `06_ssrf_attacks.py` | 64 | Internal services, cloud metadata, DNS |
| Header Injection | `07_header_injection.py` | 60 | CRLF, response splitting, smuggling |
| LDAP/XPATH/SSTI | `08_ldap_xpath_injection.py` | 60 | LDAP, XPATH, Template injection, EL |
| DoS Patterns | `09_dos_patterns.py` | 0 | Currently empty |
| Mixed Attacks | `10_mixed_blended.py` | 39 | Multi-stage, polyglot, complex chains |

**Run all tests:**
```bash
cd attack_tests
python3 run_all_tests.py
```

**Run individual test:**
```bash
cd attack_tests
python3 01_sql_injection.py
```

### Model Training

| Script | Purpose | Usage |
|--------|---------|-------|
| `finetune_waf_model.py` | Fine-tune DistilBERT model | `python3 finetune_waf_model.py` |
| `start_waf_service.py` | Start WAF API service | `python3 start_waf_service.py` |

### Presentation deck

| Script | Purpose | Usage |
|--------|---------|-------|
| `update_pitch_deck.py` | Update a PowerPoint deck from [PRESENTATION_PITCH_CONTENT.md](../PRESENTATION_PITCH_CONTENT.md) | See [Pitch deck update](#pitch-deck-update) below |

## Environment Variables

Configure these before running scripts:

```bash
# Backend API
export API_SERVER_URL=http://localhost:3001

# Database
export POSTGRES_PASSWORD=your_password
export POSTGRES_DB=waf_db

# Model
export MODEL_PATH=models/waf-distilbert
export WAF_THRESHOLD=0.5
```

## Common Workflows

### 1. Development Testing

```bash
# Quick smoke test
python3 quick_waf_test.py

# Run specific attack category
cd attack_tests
python3 01_sql_injection.py
```

### 2. Full Integration Testing

```bash
# Comprehensive test suite
cd attack_tests
python3 run_all_tests.py
```

### 3. Load Testing

```bash
# Test with 200 concurrent requests
python3 test_waf_200_requests_simple.py
```

### 4. Model Training

```bash
# Fine-tune model with new data
python3 finetune_waf_model.py --epochs 3 --batch-size 16

# Or use the Jupyter notebook
jupyter notebook ../notebooks/finetune_with_payloads.ipynb
```

## Test Results (Current Model)

Latest test results with fine-tuned model:

| Category | Detection Rate | Status |
|----------|---------------|--------|
| SQL Injection | 97.4% |  Excellent |
| XSS | 96.5% |  Excellent |
| Command Injection | 100% |  Perfect |
| Path Traversal | 100% |  Perfect |
| XXE | 100% |  Perfect |
| SSRF | 95.3% |  Excellent |
| Header Injection | 3.3% |  Needs improvement |
| LDAP/XPATH/Template | 75% |  Good |
| Mixed/Blended | 94.9% |  Excellent |
| **Overall** | **82.1%** | ** EXCELLENT** |

## Archived Scripts

Older/redundant scripts have been moved to `archived/` for reference:
- `stop_apps.sh` - Legacy stop script
- `stop_apps_docker.sh` - Old Docker stop script
- `stop_frontend_backend.sh` - Partial stop script
- `stop_real_apps.sh` - Real apps stop script
- `start_apps_docker.sh` - Old Docker start script
- `test_waf_attacks.py` - Replaced by attack_tests/
- `test_waf_detection.py` - Replaced by test_waf_integration.py

**Do not use archived scripts** - they are kept for historical reference only.

## Troubleshooting

### Backend Not Responding

```bash
# Check if backend is running
curl http://localhost:3001/health

# Restart backend
docker-compose restart backend

# Check logs
docker-compose logs backend
```

### Model Not Loaded

```bash
# Check model status
curl http://localhost:3001/api/waf/model-info

# Verify model files exist
ls -la models/waf-distilbert/
```

### Test Failures Due to Load

The attack test suite includes automatic delays between tests to prevent backend overload. If you still see failures:

1. Increase delays in `run_all_tests.py` (currently 2-5 seconds)
2. Run tests individually instead of the full suite
3. Restart backend between test categories

## Contributing

When adding new scripts:

1. Add documentation to this README
2. Use environment variables instead of hardcoded URLs
3. Follow the naming convention: `<purpose>_<target>.py/sh`
4. Add error handling and helpful error messages
5. Update the test suite if adding new attack patterns

## Pitch deck update

Use `update_pitch_deck.py` to fill a PowerPoint template with the WAF pitch content (18 slides). Requires `pip install python-pptx` (or use the project `requirements.txt`).

```bash
# Install dependency (if not already)
pip install python-pptx

# Inspect an existing .pptx (slide count and placeholder layout)
python scripts/update_pitch_deck.py --inspect --pptx /path/to/your.pptx

# Update the deck (writes a new file; does not overwrite the original)
python scripts/update_pitch_deck.py --pptx "/path/to/Black Elegant...pptx" [--out /path/to/output.pptx]

# Default: reads PRESENTATION_PITCH_CONTENT.md from repo root; writes to <pptx_stem>_WAF_updated.pptx
python scripts/update_pitch_deck.py

# Export only: generate SLIDE_CONTENT_FOR_DECK.md for manual paste into Canva/PowerPoint
python scripts/update_pitch_deck.py --export-md SLIDE_CONTENT_FOR_DECK.md --pptx /path/to/any.pptx
```

- **Canva:** The script cannot edit Canva directly. Use the generated [SLIDE_CONTENT_FOR_DECK.md](../SLIDE_CONTENT_FOR_DECK.md) (or the updated .pptx) to copy title and body text into each slide in the Canva editor. For slides 5 and 6, add the architecture/flow diagrams from the Mermaid blocks in PRESENTATION_PITCH_CONTENT.md or redraw them in Canva.
- **Output:** By default the script writes a new file (e.g. `Black Elegant..._WAF_updated.pptx`) so the original template is unchanged.

## See Also

- [Main README](../README.md) - Project overview
- [CLEANUP_REPORT.md](../CLEANUP_REPORT.md) - Recent cleanup details
- [Backend Documentation](../backend/README.md) - API documentation
- [Model Training Notebook](../notebooks/finetune_with_payloads.ipynb)

---
*Last updated: January 31, 2026*
