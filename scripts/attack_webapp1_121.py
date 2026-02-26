#!/usr/bin/env python3
"""
Attack Web App 1 with 121 malicious requests through the backend.
The WAF middleware inspects each request and blocks anomalies (403).
Run with backend up: ./scripts/start_local.sh (or backend on 3001).

Usage:
  python scripts/attack_webapp1_121.py
  python scripts/attack_webapp1_121.py --count 121 --base http://localhost:3001
"""
import os
import random
import time
import argparse

import requests

API_BASE = os.getenv("API_BASE", "http://localhost:3001")

# Paths that go through WAF middleware (backend /test/*)
PATHS = [
    "/test/endpoint",
    "/test/profile",
    "/test/search",
    "/test/login",
]

# Malicious payloads (WAF should flag these)
MALICIOUS_GET_PARAMS = [
    {"id": "1' OR '1'='1"},
    {"id": "1 UNION SELECT * FROM users--"},
    {"search": "<script>alert(1)</script>"},
    {"user": "../../../etc/passwd"},
    {"id": "'; DROP TABLE users--"},
    {"search": "<img src=x onerror=alert(1)>"},
    {"id": "1' AND 1=CONVERT(int, @@version)--"},
    {"q": "' OR 1=1--"},
    {"name": "<svg onload=alert(document.cookie)>"},
    {"id": "| cat /etc/passwd"},
    {"search": "javascript:alert(1)"},
    {"id": "1; WAITFOR DELAY '0:0:5'--"},
    {"user": "file:///etc/passwd"},
    {"id": "1 AND EXTRACTVALUE(1, CONCAT(0x7e, VERSION()))--"},
    {"search": "{{constructor.constructor('alert(1)')()}}"},
]

MALICIOUS_POST_BODIES = [
    {"query": "1' OR '1'='1", "q": "admin"},
    {"query": "<script>alert('XSS')</script>"},
    {"text": "' UNION SELECT username, password FROM users--"},
    {"query": "<img src=x onerror=alert(1)>"},
    {"input": "'; cat /etc/passwd"},
    {"query": "${7*7}", "search": "{{config.__class__.__init__.__globals__['os'].popen('id').read()}}"},
    {"content": "<iframe src=\"javascript:alert(1)\">"},
    {"data": "1 AND (SELECT * FROM (SELECT(SLEEP(5)))a)--"},
]


def build_requests(count: int):
    """Build exactly `count` malicious requests (GET and POST to /test/*)."""
    out = []
    for i in range(count):
        path = random.choice(PATHS)
        if random.random() < 0.5 and path != "/test/profile":
            # POST
            body = random.choice(MALICIOUS_POST_BODIES)
            out.append({"method": "POST", "path": path, "json": body})
        else:
            # GET
            params = random.choice(MALICIOUS_GET_PARAMS)
            out.append({"method": "GET", "path": path, "params": params})
    return out


def send_one(base_url: str, req: dict, timeout: int = 10):
    url = base_url.rstrip("/") + req["path"]
    try:
        if req["method"] == "GET":
            r = requests.get(url, params=req.get("params"), timeout=timeout)
        else:
            r = requests.post(url, json=req.get("json"), timeout=timeout)
        return r.status_code
    except requests.RequestException:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Send malicious requests to Web App 1 (via backend WAF)")
    parser.add_argument("--count", type=int, default=121, help="Number of malicious requests (default: 121)")
    parser.add_argument("--base", type=str, default=API_BASE, help="Backend base URL")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between requests (seconds)")
    args = parser.parse_args()

    base = args.base.rstrip("/")
    count = args.count

    print("=" * 60)
    print("  Attack on Web App 1 — WAF capture & block test")
    print("=" * 60)
    print(f"  Backend (WAF): {base}")
    print(f"  Malicious requests: {count}")
    print("=" * 60)

    # Health check
    try:
        r = requests.get(f"{base}/health", timeout=5)
        if r.status_code != 200:
            print(f"  Warning: /health returned {r.status_code}")
    except requests.RequestException as e:
        print(f"  ERROR: Backend not reachable: {e}")
        print("  Start backend first: ./scripts/start_local.sh")
        return 1

    requests_list = build_requests(count)
    blocked = 0
    allowed = 0
    errors = 0

    for i, req in enumerate(requests_list):
        code = send_one(base, req)
        if code == 403:
            blocked += 1
        elif code in (200, 201):
            allowed += 1
        else:
            errors += 1
        if (i + 1) % 25 == 0:
            print(f"  [{i+1:3d}/{count}] blocked={blocked} allowed={allowed} errors={errors}")
        if args.delay > 0:
            time.sleep(args.delay)

    print()
    print("  SUMMARY")
    print("  " + "-" * 40)
    print(f"  Total:   {count}")
    print(f"  Blocked (403): {blocked}  ({100 * blocked / count:.1f}%)")
    print(f"  Allowed (200): {allowed}  ({100 * allowed / count:.1f}%)")
    print(f"  Other/errors:  {errors}")
    print()
    print("  View events and charts: http://localhost:3000/dashboard")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
