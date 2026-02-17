#!/usr/bin/env python3
"""
Send 500 requests (mixed benign + attack) to the portal via gateway.
Verifies WAF detection and that gateway events reach the backend for charts.

Usage:
  # Via gateway (portal = upstream, e.g. frontend) - events go to backend ingest
  TARGET_URL=http://localhost:8080 python scripts/run_500_attack_portal.py

  # Via backend /test/* (WAF middleware) - traffic logged directly
  TARGET_URL=http://localhost:3001 python scripts/run_500_attack_portal.py

Requires: Backend (3001), and either Gateway (8080) or use 3001 for /test/*.
"""

import os
import random
import time
from datetime import datetime

import requests

TARGET_URL = os.getenv("TARGET_URL", "http://localhost:8080")
TOTAL_REQUESTS = 500
# Use gateway-style paths (portal) or backend /test/* paths
USE_GATEWAY = "8080" in TARGET_URL or "gateway" in TARGET_URL.lower()

# Portal-like paths (when using gateway → upstream frontend)
BENIGN = [
    {"method": "GET", "path": "/"},
    {"method": "GET", "path": "/dashboard"},
    {"method": "GET", "path": "/analytics"},
    {"method": "GET", "path": "/traffic"},
    {"method": "GET", "path": "/bot-detection"},
    {"method": "GET", "path": "/api/health"},
]
# Attack payloads in query or path (WAF should block many)
ATTACKS = [
    {"method": "GET", "path": "/", "params": {"q": "1' OR '1'='1"}},
    {"method": "GET", "path": "/dashboard", "params": {"id": "1 UNION SELECT * FROM users--"}},
    {"method": "GET", "path": "/search", "params": {"q": "<script>alert(1)</script>"}},
    {"method": "GET", "path": "/", "params": {"x": "<img src=x onerror=alert(1)>"}},
    {"method": "GET", "path": "/api/", "params": {"cmd": "'; cat /etc/passwd"}},
    {"method": "GET", "path": "/login", "params": {"user": "../../../etc/passwd"}},
    {"method": "POST", "path": "/search", "json": {"query": "<svg onload=alert(1)>"}},
    {"method": "GET", "path": "/?id=1; rm -rf /"},
]
# Backend /test/* paths (when TARGET_URL is backend)
BACKEND_BENIGN = [
    {"method": "GET", "path": "/test/endpoint", "params": {"id": "123"}},
    {"method": "GET", "path": "/test/profile", "params": {"name": "John"}},
]
BACKEND_ATTACKS = [
    {"method": "GET", "path": "/test/endpoint", "params": {"id": "1' OR '1'='1"}},
    {"method": "GET", "path": "/test/endpoint", "params": {"search": "<script>alert(1)</script>"}},
]


def send_one(base_url, req, timeout=8):
    path = req.get("path", "/")
    params = req.get("params")
    json_data = req.get("json")
    method = req.get("method", "GET")
    url = base_url.rstrip("/") + path
    try:
        if method == "GET":
            r = requests.get(url, params=params, timeout=timeout)
        else:
            r = requests.post(url, json=json_data, timeout=timeout)
        return r.status_code, r.headers.get("X-Request-ID")
    except requests.exceptions.RequestException as e:
        return 0, str(e)


def main():
    print("=" * 64)
    print("  500-Request Attack on Portal (Gateway/Backend)")
    print("=" * 64)
    print(f"  Target: {TARGET_URL}")
    print(f"  Total requests: {TOTAL_REQUESTS}")
    print(f"  Mode: {'Gateway (portal upstream)' if USE_GATEWAY else 'Backend /test/*'}")
    print("=" * 64)

    # Health check
    try:
        r = requests.get(TARGET_URL.rstrip("/") + ("/" if USE_GATEWAY else "/health"), timeout=5)
        if r.status_code not in (200, 403, 429, 404):
            print(f"  Warning: target returned {r.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"  ERROR: Target not reachable: {e}")
        print("  Start gateway: (from project root) or backend on 3001")
        return 1

    benign_pool = BENIGN if USE_GATEWAY else BACKEND_BENIGN
    attack_pool = ATTACKS if USE_GATEWAY else BACKEND_ATTACKS

    stats = {"total": 0, "allowed": 0, "blocked": 0, "rate_limit": 0, "errors": 0}
    start = time.time()

    for i in range(TOTAL_REQUESTS):
        is_attack = random.random() < 0.25  # 25% attack
        req = random.choice(attack_pool) if is_attack else random.choice(benign_pool)
        code, _ = send_one(TARGET_URL, req)
        stats["total"] += 1
        if code == 200:
            stats["allowed"] += 1
        elif code == 403:
            stats["blocked"] += 1
        elif code == 429:
            stats["rate_limit"] += 1
        else:
            stats["errors"] += 1
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1:4d}/{TOTAL_REQUESTS}] allowed={stats['allowed']} blocked={stats['blocked']} 429={stats['rate_limit']} err={stats['errors']} ({elapsed:.1f}s)")
        time.sleep(0.02)  # ~50 req/s max to avoid overwhelming

    elapsed = time.time() - start
    print()
    print("  SUMMARY")
    print("  -" * 32)
    print(f"  Total:    {stats['total']}")
    print(f"  Allowed:  {stats['allowed']} ({100*stats['allowed']/stats['total']:.1f}%)")
    print(f"  Blocked:  {stats['blocked']} ({100*stats['blocked']/stats['total']:.1f}%)")
    print(f"  429:      {stats['rate_limit']}")
    print(f"  Errors:   {stats['errors']}")
    print(f"  Duration: {elapsed:.1f}s ({stats['total']/elapsed:.0f} req/s)")
    print()
    print("  Next: open http://localhost:3000/dashboard and check Request Volume & Threats, Analytics, Traffic.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    exit(main())
