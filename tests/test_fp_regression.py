#!/usr/bin/env python3
"""
False Positive Regression Test Suite

Verifies that known-benign requests are NOT flagged as malicious.
Run after every model retrain to catch FP regressions.

Usage:
    pytest tests/test_fp_regression.py -v
    python tests/test_fp_regression.py          # standalone
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from backend.ml.waf_classifier import WAFClassifier

# Known-benign requests that must NOT be flagged
BENIGN_REQUESTS = [
    # Standard browsing
    {"method": "GET", "path": "/", "query_params": {}, "headers": {}, "body": None},
    {"method": "GET", "path": "/products", "query_params": {"category": "electronics", "page": "2"}, "headers": {}, "body": None},
    {"method": "GET", "path": "/api/users/42/profile", "query_params": {}, "headers": {}, "body": None},

    # Login/auth (normal)
    {"method": "POST", "path": "/api/login", "query_params": {}, "headers": {"Content-Type": "application/json"}, "body": '{"username":"john","password":"securePass123!"}'},

    # Search with innocent terms
    {"method": "GET", "path": "/api/search", "query_params": {"q": "best wireless headphones 2024"}, "headers": {}, "body": None},
    {"method": "POST", "path": "/api/search", "query_params": {}, "headers": {"Content-Type": "application/json"}, "body": '{"query":"laptop"}'},

    # Technical discussion (FP-prone keywords)
    {"method": "POST", "path": "/api/forum", "query_params": {}, "headers": {"Content-Type": "application/json"}, "body": '{"text":"Use SELECT * FROM users WHERE active = 1 for the query"}'},
    {"method": "POST", "path": "/api/comments", "query_params": {}, "headers": {"Content-Type": "application/json"}, "body": '{"text":"The OR operator in Python returns the first truthy value"}'},
    {"method": "GET", "path": "/api/docs", "query_params": {"topic": "xpath selectors tutorial"}, "headers": {}, "body": None},
    {"method": "POST", "path": "/api/support", "query_params": {}, "headers": {"Content-Type": "application/json"}, "body": '{"issue":"LDAP connection timeout after 30 seconds"}'},
    {"method": "POST", "path": "/api/forum", "query_params": {}, "headers": {"Content-Type": "application/json"}, "body": '{"title":"How to use Jinja2 templates","body":"I want to render variables"}'},

    # Normal headers (should not trigger header injection)
    {"method": "GET", "path": "/api/data", "query_params": {}, "headers": {"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signature", "Accept": "application/json"}, "body": None},
    {"method": "GET", "path": "/api/profile", "query_params": {}, "headers": {"Cookie": "session=abc123; theme=dark", "Accept-Language": "en-US,en;q=0.9"}, "body": None},
    {"method": "GET", "path": "/api/feed", "query_params": {}, "headers": {"X-Forwarded-For": "10.0.0.1", "X-Request-ID": "req-12345"}, "body": None},

    # Redirect/URL params (FP-prone)
    {"method": "GET", "path": "/api/redirect", "query_params": {"url": "https://example.com/callback"}, "headers": {}, "body": None},
    {"method": "GET", "path": "/api/download", "query_params": {"file": "report-2024-Q1.pdf"}, "headers": {}, "body": None},
    {"method": "GET", "path": "/api/oauth/callback", "query_params": {"code": "abc123", "state": "xyz"}, "headers": {}, "body": None},

    # JSON API calls
    {"method": "POST", "path": "/api/orders", "query_params": {}, "headers": {"Content-Type": "application/json"}, "body": '{"items":[{"id":1,"qty":2}],"shipping":"express"}'},
    {"method": "PUT", "path": "/api/users/5", "query_params": {}, "headers": {"Content-Type": "application/json"}, "body": '{"name":"Alice Smith","email":"alice@example.com"}'},
    {"method": "DELETE", "path": "/api/cart/items/7", "query_params": {}, "headers": {}, "body": None},

    # Webhook with normal headers
    {"method": "POST", "path": "/api/webhook", "query_params": {}, "headers": {"Content-Type": "application/json", "X-Hub-Signature": "sha256=abc123"}, "body": '{"event":"push","ref":"refs/heads/main"}'},
]


def _req_id(req):
    """Generate a short test ID from a request dict."""
    return f"{req['method']}_{req['path'].replace('/', '_').strip('_')}"


@pytest.fixture(scope="module")
def classifier():
    model_path = str(PROJECT_ROOT / "models" / "waf-distilbert")
    clf = WAFClassifier(model_path=model_path, threshold=0.5)
    if not clf.is_loaded:
        pytest.skip("Model not available at models/waf-distilbert")
    return clf


@pytest.mark.parametrize(
    "request_data",
    BENIGN_REQUESTS,
    ids=[_req_id(r) for r in BENIGN_REQUESTS],
)
def test_benign_not_flagged(classifier, request_data):
    """Each known-benign request must NOT be classified as malicious."""
    result = classifier.check_request(request_data)
    malicious_score = result.get("malicious_score", 0.0)
    assert not result.get("is_malicious", False), (
        f"False positive! {request_data['method']} {request_data['path']} "
        f"scored {malicious_score:.4f} (threshold: {classifier.threshold})"
    )


def test_fp_rate_under_one_percent(classifier):
    """Overall FP rate across all benign requests must be < 1%."""
    fp_count = 0
    for req in BENIGN_REQUESTS:
        result = classifier.check_request(req)
        if result.get("is_malicious", False):
            fp_count += 1
    fp_rate = fp_count / len(BENIGN_REQUESTS) * 100
    assert fp_rate < 1.0, (
        f"FP rate {fp_rate:.1f}% exceeds 1% threshold "
        f"({fp_count}/{len(BENIGN_REQUESTS)})"
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model_path = str(PROJECT_ROOT / "models" / "waf-distilbert")
    clf = WAFClassifier(model_path=model_path, threshold=0.5)
    if not clf.is_loaded:
        print("ERROR: Model not loaded from", model_path)
        sys.exit(1)

    print(f"Testing {len(BENIGN_REQUESTS)} benign requests for false positives...\n")

    fp_count = 0
    for i, req in enumerate(BENIGN_REQUESTS, 1):
        result = clf.check_request(req)
        is_fp = result.get("is_malicious", False)
        score = result.get("malicious_score", 0)
        status = "\033[91mFP!\033[0m" if is_fp else "\033[92mOK \033[0m"
        print(f"[{i:02d}] [{status}] {req['method']} {req['path'][:35]:<35} score={score:.4f}")
        if is_fp:
            fp_count += 1

    fp_rate = fp_count / len(BENIGN_REQUESTS) * 100
    print(f"\nFP Rate: {fp_rate:.1f}% ({fp_count}/{len(BENIGN_REQUESTS)})")
    print("PASS" if fp_rate < 1.0 else "FAIL")
    sys.exit(0 if fp_rate < 1.0 else 1)
