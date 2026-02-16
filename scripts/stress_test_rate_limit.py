#!/usr/bin/env python3
"""
Stress test for Rate Limit and DDoS protection.

Sends burst of requests from single IP to verify:
1. Rate limit triggers (429) after exceeding RATE_LIMIT_REQUESTS_PER_MINUTE
2. DDoS burst detection triggers block after exceeding DDOS_BURST_THRESHOLD

Usage:
    STRESS_TEST_BASE_URL=http://localhost:8080 python scripts/stress_test_rate_limit.py

Default target: http://localhost:8080 (gateway)
"""

import os
import time
import requests

BASE_URL = os.getenv("STRESS_TEST_BASE_URL", "http://localhost:8080")
RATE_LIMIT = int(os.getenv("STRESS_RATE_LIMIT", "120"))
BURST_THRESHOLD = int(os.getenv("STRESS_BURST_THRESHOLD", "50"))


def main():
    print("=" * 60)
    print("  Rate Limit & DDoS Stress Test")
    print("=" * 60)
    print(f"  Target: {BASE_URL}")
    print(f"  Rate limit: ~{RATE_LIMIT} req/min, Burst threshold: ~{BURST_THRESHOLD} req/5s")
    print("=" * 60)

    # Phase 1: Rate limit test - send more than limit in 1 minute
    print("\n[Phase 1] Rate limit test...")
    rate_limit_hits = 0
    for i in range(RATE_LIMIT + 30):
        try:
            r = requests.get(f"{BASE_URL}/", timeout=5)
            if r.status_code == 429:
                rate_limit_hits += 1
        except requests.exceptions.RequestException as e:
            print(f"  Request {i+1} failed: {e}")
            break
        if (i + 1) % 50 == 0:
            print(f"  Sent {i+1} requests, 429 count: {rate_limit_hits}")

    print(f"  Result: {rate_limit_hits} rate limit (429) responses")
    if rate_limit_hits > 0:
        print("  PASS: Rate limiting is active")
    else:
        print("  NOTE: No 429 seen - rate limit may be higher or disabled")

    time.sleep(2)

    # Phase 2: DDoS burst test - send rapid burst
    print("\n[Phase 2] DDoS burst test (rapid requests)...")
    burst_hits = 0
    start = time.time()
    for i in range(BURST_THRESHOLD + 20):
        try:
            r = requests.get(f"{BASE_URL}/", timeout=5)
            if r.status_code == 429:
                burst_hits += 1
        except requests.exceptions.RequestException as e:
            print(f"  Request {i+1} failed: {e}")
            break
    elapsed = time.time() - start
    print(f"  Sent {BURST_THRESHOLD + 20} requests in {elapsed:.2f}s ({elapsed and (BURST_THRESHOLD + 20) / elapsed:.0f} req/s)")
    print(f"  Result: {burst_hits} burst-related 429 responses")
    if burst_hits > 0:
        print("  PASS: DDoS burst detection is active")
    else:
        print("  NOTE: No burst block seen - threshold may be higher or disabled")

    print("\n" + "=" * 60)
    print("  Done. Check dashboard for Rate Limit & DDoS metrics.")
    print("=" * 60)


if __name__ == "__main__":
    main()
