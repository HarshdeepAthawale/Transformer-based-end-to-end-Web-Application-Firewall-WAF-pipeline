#!/usr/bin/env python3
"""
WAF Demo Script - Interactive demonstration of attack detection.

Tests SQL injection, XSS, path traversal, and RCE patterns against the WAF.
Default URL: http://localhost:3001/api/waf/check (backend API)

Usage:
    python scripts/demo.py
    python scripts/demo.py --url http://localhost:3001/api/waf/check
"""
import argparse
import time
from typing import Dict, List

import requests


def main():
    parser = argparse.ArgumentParser(description="WAF attack detection demo")
    parser.add_argument(
        "--url",
        default="http://localhost:3001/api/waf/check",
        help="WAF check API URL",
    )
    args = parser.parse_args()

    scenarios = [
        {
            "name": "Normal Request",
            "request": {
                "method": "GET",
                "path": "/api/users",
                "query_params": {"page": "1", "limit": "10"},
                "headers": {},
                "body": None,
            },
            "expected": False,
        },
        {
            "name": "SQL Injection Attack",
            "request": {
                "method": "GET",
                "path": "/api/users",
                "query_params": {"id": "1' OR '1'='1"},
                "headers": {},
                "body": None,
            },
            "expected": True,
        },
        {
            "name": "XSS Attack",
            "request": {
                "method": "GET",
                "path": "/search",
                "query_params": {"q": "<script>alert('XSS')</script>"},
                "headers": {},
                "body": None,
            },
            "expected": True,
        },
        {
            "name": "Path Traversal Attack",
            "request": {
                "method": "GET",
                "path": "/api/files",
                "query_params": {"file": "../../../etc/passwd"},
                "headers": {},
                "body": None,
            },
            "expected": True,
        },
        {
            "name": "RCE Attack",
            "request": {
                "method": "POST",
                "path": "/api/execute",
                "query_params": {},
                "headers": {},
                "body": "; cat /etc/passwd",
            },
            "expected": True,
        },
    ]

    print("\n" + "=" * 60)
    print("WAF TRANSFORMER DEMONSTRATION")
    print("=" * 60)
    print(f"URL: {args.url}\n")

    results: List[Dict] = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] Testing: {scenario['name']}")
        print(f"  Request: {scenario['request']['method']} {scenario['request']['path']}")

        try:
            resp = requests.post(
                args.url,
                json=scenario["request"],
                timeout=10,
                headers={"Content-Type": "application/json"},
            )
            data = resp.json()
            is_anomaly = data.get("is_anomaly", False)
            score = data.get("anomaly_score", 0.0)

            status = "DETECTED" if is_anomaly else "NOT DETECTED"
            match = is_anomaly == scenario["expected"]
            symbol = "\033[92m\033[0m" if match else "\033[91m\033[0m"

            print(f"  Result: {status} (Score: {score:.3f}) {symbol}")

            results.append({
                "scenario": scenario["name"],
                "detected": is_anomaly,
                "expected": scenario["expected"],
                "score": score,
                "match": match,
            })
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "scenario": scenario["name"],
                "error": str(e),
            })

        time.sleep(0.3)

    # Summary
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)

    valid = [r for r in results if "match" in r]
    total = len(valid)
    correct = sum(1 for r in valid if r["match"])

    print(f"Total Tests: {total}")
    print(f"Correct Detections: {correct}")
    print(f"Accuracy: {correct / total * 100:.1f}%" if total > 0 else "N/A")

    print("\nDetailed Results:")
    for r in results:
        if "match" in r:
            s = "\033[92m\033[0m" if r["match"] else "\033[91m\033[0m"
            print(f"  {s} {r['scenario']}: Detected={r['detected']}, Score={r['score']:.3f}")
        else:
            print(f"  \033[91m\033[0m {r['scenario']}: Error - {r.get('error', 'Unknown')}")

    print()


if __name__ == "__main__":
    main()
