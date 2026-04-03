#!/bin/bash
# Pre-commit hook: Check for CI/CD errors before pushing

set -e

echo ""
echo " PRE-COMMIT CI/CD CHECK"
echo ""
echo ""

ERRORS=0

# Check 1: Quick syntax check
echo "  Checking Python syntax..."
for file in backend/services/*.py; do
    if ! python3 -m py_compile "$file" 2>/dev/null; then
        echo "    Syntax error in $file"
        ERRORS=$((ERRORS + 1))
    fi
done
if [ $ERRORS -eq 0 ]; then
    echo "    All Python files have valid syntax"
fi

echo ""

# Check 2: Unit tests
echo "  Running critical tests..."
if python3 -m pytest tests/unit/test_ip_fencing.py tests/unit/test_bot_management.py -q > /dev/null 2>&1; then
    echo "    Unit tests PASSED"
else
    echo "    Tests failed - run: python3 -m pytest tests/unit/test_ip_fencing.py tests/unit/test_bot_management.py -v"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# Summary
echo ""
if [ $ERRORS -eq 0 ]; then
    echo " ALL CHECKS PASSED - Safe to push!"
else
    echo " $ERRORS CHECK(S) FAILED - Fix before pushing"
fi
echo ""

exit $ERRORS
