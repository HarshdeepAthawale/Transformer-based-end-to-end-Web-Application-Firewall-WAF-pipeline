# CI/CD Error Prevention Guide

##  Quick Start

### Before Every Push:
```bash
bash scripts/pre-commit-check.sh
```

This will:
1.  Check Python syntax
2.  Run critical unit tests
3.  Report status

### Automated Error Fixer (When Needed):
```bash
python3 scripts/cicd_auto_fixer.py
```

---

##  Common Errors & Solutions

### Error Type 1: Undefined Variable
```
F821 Undefined name `org_id`
    rule = GeoRule(org_id=org_id, ...)  # org_id not in method signature
```

**Fix:**
```python
#  WRONG
def create_rule(self, country_code: str):
    rule = GeoRule(org_id=org_id, ...)

#  CORRECT
def create_rule(self, org_id: int, country_code: str):
    rule = GeoRule(org_id=org_id, ...)
```

### Error Type 2: Missing Test Arguments
```
TypeError: service.detect_bot() missing 1 required positional argument: 'org_id'
```

**Fix:**
```python
#  WRONG
service.detect_bot(user_agent="...", ip="...")

#  CORRECT
service.detect_bot(org_id=1, user_agent="...", ip="...")
```

---

##  Recommended Workflow

```bash
# Before changes
bash scripts/pre-commit-check.sh

# After changes
bash scripts/pre-commit-check.sh

# If errors
python3 scripts/cicd_auto_fixer.py
bash scripts/pre-commit-check.sh

# Commit & push
git add .
git commit -m "..."
git push origin master
```

---

##  Pre-Commit Checks

-  Python syntax validation
-  Import test
-  Unit test verification (critical tests)
-  Summary report

All checks must pass before pushing to GitHub.
