#!/usr/bin/env python3
"""
Run 300 requests on 3 web apps: 125 malicious, 175 benign.
Requests go through the backend WAF (default http://localhost:3001).
Traffic is logically split across 3 web apps for reporting.

Usage:
  python scripts/run_300_requests_3_webapps.py
  python scripts/run_300_requests_3_webapps.py --base http://localhost:3001 --delay 0.05
"""
import os
import random
import time
import argparse

import requests

API_BASE = os.getenv("API_BASE", "http://localhost:3001")

PATHS = [
    "/test/endpoint",
    "/test/profile",
    "/test/search",
    "/test/login",
]

# Malicious payloads (same as attack_webapp1_121.py)
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

# Benign requests
BENIGN_GET_PARAMS = [
    {"id": "123"},
    {"search": "products"},
    {"name": "John"},
    {"id": "1"},
    {"search": "laptop"},
    {"user": "guest"},
]

BENIGN_POST_BODIES = [
    {"username": "user", "password": "pass"},
    {"query": "laptop"},
    {"query": "report"},
]


def build_malicious_requests(count: int):
    """Build exactly `count` malicious requests."""
    out = []
    for i in range(count):
        path = random.choice(PATHS)
        if random.random() < 0.5 and path != "/test/profile":
            body = random.choice(MALICIOUS_POST_BODIES)
            out.append({"method": "POST", "path": path, "json": body, "malicious": True})
        else:
            params = random.choice(MALICIOUS_GET_PARAMS)
            out.append({"method": "GET", "path": path, "params": params, "malicious": True})
    return out


def build_benign_requests(count: int):
    """Build exactly `count` benign requests."""
    out = []
    for i in range(count):
        path = random.choice(PATHS)
        if random.random() < 0.5 and path != "/test/profile":
            body = random.choice(BENIGN_POST_BODIES)
            out.append({"method": "POST", "path": path, "json": body, "malicious": False})
        else:
            params = random.choice(BENIGN_GET_PARAMS)
            out.append({"method": "GET", "path": path, "params": params, "malicious": False})
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
    parser = argparse.ArgumentParser(
        description="Run 300 requests on 3 web apps (125 malicious, 175 benign) via backend WAF"
    )
    parser.add_argument("--base", type=str, default=API_BASE, help="Backend base URL")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between requests (seconds)")
    args = parser.parse_args()

    base = args.base.rstrip("/")
    num_malicious = 125
    num_benign = 175
    total = 300
    num_apps = 3

    print("=" * 60)
    print("  300 requests on 3 web apps (125 malicious, 175 benign)")
    print("=" * 60)
    print(f"  Backend (WAF): {base}")
    print(f"  Malicious: {num_malicious}  Benign: {num_benign}  Total: {total}")
    print("=" * 60)

    try:
        r = requests.get(f"{base}/health", timeout=5)
        if r.status_code != 200:
            print(f"  Warning: /health returned {r.status_code}")
    except requests.RequestException as e:
        print(f"  ERROR: Backend not reachable: {e}")
        print("  Start backend first: ./scripts/start_local.sh")
        return 1

    malicious_list = build_malicious_requests(num_malicious)
    benign_list = build_benign_requests(num_benign)
    all_requests = malicious_list + benign_list
    random.shuffle(all_requests)

    blocked = 0
    allowed = 0
    errors = 0
    blocked_malicious = 0
    allowed_malicious = 0
    blocked_benign = 0
    allowed_benign = 0
    app_counts = [{"total": 0, "blocked": 0, "allowed": 0, "errors": 0} for _ in range(num_apps)]

    for i, req in enumerate(all_requests):
        app_idx = i % num_apps
        code = send_one(base, req)
        is_mal = req.get("malicious", False)

        if code == 403:
            blocked += 1
            if is_mal:
                blocked_malicious += 1
            else:
                blocked_benign += 1
            app_counts[app_idx]["blocked"] += 1
        elif code in (200, 201):
            allowed += 1
            if is_mal:
                allowed_malicious += 1
            else:
                allowed_benign += 1
            app_counts[app_idx]["allowed"] += 1
        else:
            errors += 1
            app_counts[app_idx]["errors"] += 1
        app_counts[app_idx]["total"] += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1:3d}/{total}] blocked={blocked} allowed={allowed} errors={errors}")
        if args.delay > 0:
            time.sleep(args.delay)

    print()
    print("  SUMMARY")
    print("  " + "-" * 50)
    print(f"  Total:      {total}")
    print(f"  Blocked:    {blocked}  ({100 * blocked / total:.1f}%)")
    print(f"  Allowed:    {allowed}  ({100 * allowed / total:.1f}%)")
    print(f"  Other:      {errors}")
    print()
    print("  Malicious (125):")
    print(f"    Blocked: {blocked_malicious}  Allowed: {allowed_malicious}")
    print("  Benign (175):")
    print(f"    Blocked: {blocked_benign}  Allowed: {allowed_benign}")
    print()
    print("  By web app (round-robin):")
    for a in range(num_apps):
        c = app_counts[a]
        print(f"    App {a+1}: total={c['total']} blocked={c['blocked']} allowed={c['allowed']} errors={c['errors']}")
    print()
    print("  Dashboard: http://localhost:3000/dashboard")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
