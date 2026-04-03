# Claude Code Project Rules - WAF Pipeline

This document defines guidelines for Claude Code when working on the Transformer-based WAF pipeline project.

---

## Rule 1: CI/CD Tests Before Push

**BLOCKING RULE:** Never push to the GitHub repository until ALL CI/CD tests pass.

**Implementation:**
- Always run `bash scripts/pre-commit-check.sh` before committing
- Verify output: "ALL CHECKS PASSED - Safe to push!"
- If tests fail, fix the issues and re-run `pre-commit-check.sh`
- Only after passing checks, proceed with `git add`, `git commit`, `git push`

**Why:** Failed CI/CD blocks the entire pipeline and wastes time debugging. The pre-commit check prevents this.

---

## Rule 2: Store All Weekly Plans in `/plans` Directory

**STORAGE RULE:** Save all weekly planning documents to `/home/harshdeep/Documents/Projects/Transformer based end-to-end Web Application Firewall (WAF) pipeline/plans/`

**Structure:**
```
plans/
  battle-plan-beat-cloudflare-8-weeks.md
  week-1/
    Day-1-Multi-Tenancy-Foundation.md
    Day-2-TenantMiddleware-Auth.md
    ...
    README.md
  week-2/
    Day-1-Route-Controller-Hardening.md
    Day-2-Bot-Detection-Improvements.md
    ...
    README.md
  week-3/
    Day-1-Multi-Tenancy-Critical-Fixes.md
    Day-2-Gateway-Rate-Limits.md
    ...
    README.md
```

**Pattern per week:**
- Create a `week-N/` folder
- Save daily plan files: `Day-X-<Topic>.md`
- Include a `week-N/README.md` summarizing the week and progress

**Why:** Centralized documentation makes it easy to track what was planned vs. executed, reference past decisions, and brief new team members.

---

## Rule 3: Multi-Tenancy First

**ARCHITECTURE RULE:** All new services, controllers, and routes must support multi-tenancy via `org_id`.

**Pattern:**
```python
# Routes: Always require authentication
@router.get("/endpoint")
async def handler(
    org_id: int = Depends(get_current_tenant),  # MANDATORY
    db: Session = Depends(get_db),
):
    return ctrl.method(db, org_id, ...)  # Pass to controller

# Controllers: Accept org_id and pass to service
def method(db: Session, org_id: int, ...):
    return service.method(org_id, ...)  # Pass to service

# Services: Filter by org_id
def method(org_id: int, ...):
    return db.query(Model).filter(Model.org_id == org_id)
```

**Cross-tenant protection:** When accessing a single resource by ID, filter by both ID and org_id:
```python
db.query(Resource).filter(
    Resource.id == resource_id,
    Resource.org_id == org_id
).first()
# Returns 404 (same as not found) if org_id mismatch — no info leakage
```

**Why:** Data isolation is non-negotiable for a SaaS WAF. Every new feature must respect tenant boundaries.

---

## Rule 4: No Emojis in Code or Docs

**STYLE RULE:** Remove all emojis from codebase, documentation, and comments. Keep it professional.

**Includes:**
- Source code comments
- Docstrings
- Commit messages
- Plan documents
- Git branch names

**Why:** Professional standards. Emojis add noise and can cause encoding issues in some tools.

---

## Rule 5: Follow Established Auth Patterns

**SECURITY RULE:** Use `get_current_tenant` dependency for all tenant-scoped endpoints. Use `require_waf_api_auth` for admin/global operations.

**Import:**
```python
from backend.auth import get_current_tenant, require_waf_api_auth
```

**Tenant-scoped endpoint:**
```python
@router.get("/endpoint")
async def handler(org_id: int = Depends(get_current_tenant), db: Session = Depends(get_db)):
    ...
```

**Admin/global endpoint:**
```python
@router.post("/admin/endpoint")
async def handler(_auth=Depends(require_waf_api_auth), db: Session = Depends(get_db)):
    ...
```

**JWT payload contains:**
- `user_id` — backend user ID
- `username` — username string
- `org_id` — tenant organization ID
- `exp` — expiration timestamp

**Why:** Consistency prevents auth bypasses and makes security audits easier.

---

## Rule 6: Test Coverage for New Features

**TESTING RULE:** Any new feature must include unit tests and integration tests.

**Minimum test coverage:**
- Unit tests: happy path + error cases
- Integration tests: cross-org isolation if tenant-scoped
- CI/CD: tests pass before merge

**File locations:**
```
tests/
  unit/test_<feature>.py
  integration/test_<feature>.py
```

**Run before commit:**
```bash
pytest tests/unit/test_<feature>.py -v
pytest tests/integration/test_<feature>.py -v
bash scripts/pre-commit-check.sh
```

**Why:** Catches bugs early. Integration tests prove multi-tenancy correctness.

---

## Rule 7: Backward Compatibility with Graceful Degradation

**STABILITY RULE:** When adding optional features (e.g., ONNX), provide fallback to stable version.

**Pattern (from ONNX example):**
```python
# In waf_factory.py:
use_onnx = os.environ.get("WAF_USE_ONNX", "false").lower() == "true"
if use_onnx:
    try:
        classifier = ONNXWAFClassifier(...)
        if classifier.is_loaded:
            return classifier
    except ImportError:
        logger.warning("ONNX not available, falling back to PyTorch")

# Default PyTorch path unchanged
classifier = WAFClassifier(...)
return classifier
```

**Why:** Allows experimental features to ship without risking production stability.

---

## Rule 8: Daily Commits After Each Day's Work

**WORKFLOW RULE:** After completing a day's work, run pre-commit check, commit, and push. Stop and wait for next day.

**Pattern:**
1. Complete the day's tasks
2. Run `bash scripts/pre-commit-check.sh`
3. Fix any lint/test failures
4. `git add`, `git commit -m "Week N Day X: <topic>"`, `git push origin master`
5. Stop and wait for user to say "resume" or request next day

**Commit message format:**
```
Week N Day X: <Topic Name>

- Bullet point 1
- Bullet point 2
- Bullet point 3
```

**Why:** Clear separation of work, easy rollback if needed, prevents mega-commits.

---

## Rule 9: Service Layer Should Be Agnostic to Transport

**ARCHITECTURE RULE:** Business logic lives in services. Routes/controllers are thin adapters.

**Anti-pattern:**
```python
# BAD: Business logic in route
@router.get("/endpoint")
async def handler(...):
    if org_id is None:
        return {"error": "auth required"}
    # ... 20 lines of business logic ...
```

**Good pattern:**
```python
# GOOD: Business logic in service
def service_method(org_id: int, ...):
    # All the logic here

@router.get("/endpoint")
async def handler(org_id: int = Depends(get_current_tenant), db: Session = Depends(get_db)):
    return ctrl.method(db, org_id, ...)  # Delegate to controller → service
```

**Why:** Services can be tested without HTTP. Controllers remain thin and easy to audit.

---

## Rule 10: Database Migrations for Schema Changes

**DATA RULE:** Use Alembic for all schema changes. Never hand-modify the database.

**Process:**
```bash
# After changing a model in backend/models/*.py:
alembic revision --autogenerate -m "short_description"
# Review the migration file: alembic/versions/*.py
alembic upgrade head  # Apply locally
# Commit migration with code changes
```

**Why:** Ensures schema and code stay in sync. Easy rollback. Team members stay aligned.

---

## Rule 11: Performance Benchmarking for Critical Paths

**PERFORMANCE RULE:** When optimizing critical paths (ML inference, rate limiting, alert evaluation), include before/after latency measurements.

**Benchmark format:**
```python
import time
texts = [...] * 100  # 100 samples

t0 = time.perf_counter()
# Old implementation
old_time = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
# New implementation
new_time = (time.perf_counter() - t0) * 1000

print(f"Old: {old_time:.1f}ms, New: {new_time:.1f}ms, Speedup: {old_time/new_time:.2f}x")
```

**Document in commit message:**
```
ONNX export + gateway batching: 3.2x inference latency reduction
- PyTorch 100x: 320ms
- ONNX batch 100x: 100ms
- Speedup: 3.2x
```

**Why:** Proves optimizations work. Prevents regressions.

---

## Rule 12: Security Comments for Sensitive Code

**SECURITY RULE:** Mark code that handles secrets, auth, or data isolation with `# SECURITY:` comments.

**Examples:**
```python
# SECURITY: org_id mismatch returns 404 (not 403) to avoid info leakage
if resource.org_id != org_id:
    raise HTTPException(status_code=404)

# SECURITY: webhook_secret is stored in cleartext; plan encryption for Week 4
webhook_secret = cfg.get("webhook_secret") or ""

# SECURITY: verify JWT signature with RS256 public key (not symmetric key)
payload = jwt.decode(token, public_key, algorithms=["RS256"])
```

**Why:** Flags sensitive code for security review and future hardening.

---

## Summary

| Rule | Enforcement | Consequence of Violation |
|------|-------------|--------------------------|
| 1. CI/CD before push | Blocking | Cannot push unless tests pass |
| 2. Plans in /plans | Convention | Easy reference, team alignment |
| 3. Multi-tenancy first | Code review | Rejected if org_id missing |
| 4. No emojis | Linting | Caught by pre-commit |
| 5. Auth patterns | Code review | Enforced at merge |
| 6. Test coverage | CI/CD | Tests must pass |
| 7. Graceful degradation | Code review | Prevents stability regressions |
| 8. Daily commits | Workflow | Clear work tracking |
| 9. Service layer separation | Code review | Enforced at merge |
| 10. Alembic migrations | Code review | Enforced at merge |
| 11. Benchmarks for perf | Documentation | Commit messages reviewed |
| 12. Security comments | Code review | Flagged for audit |

---

**Last Updated:** 2026-04-04
**Version:** 1.0
