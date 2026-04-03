#!/usr/bin/env python3
"""
Real Traffic Generator for WAF Dashboard

This script generates real HTTP requests that go through the WAF middleware,
creating traffic logs that appear in the dashboard.
"""
import requests
import time
import random
from datetime import datetime

# Target endpoint (not in /api/* so it goes through WAF middleware)
BASE_URL = "http://localhost:3001"

# Benign requests
BENIGN_REQUESTS = [
    {"method": "GET", "url": "/test/endpoint", "params": {"id": "123"}},
    {"method": "GET", "url": "/test/endpoint", "params": {"search": "products"}},
    {"method": "GET", "url": "/test/profile", "params": {"name": "John"}},
    {"method": "POST", "url": "/test/login", "json": {"username": "user", "password": "pass"}},
    {"method": "POST", "url": "/test/search", "json": {"query": "laptop"}},
]

# Malicious requests (attack payloads)
MALICIOUS_REQUESTS = [
    # SQL Injection
    {"method": "GET", "url": "/test/endpoint", "params": {"id": "1' OR '1'='1"}},
    {"method": "GET", "url": "/test/endpoint", "params": {"id": "1 UNION SELECT * FROM users--"}},
    {"method": "GET", "url": "/test/endpoint", "params": {"search": "' OR 1=1--"}},

    # XSS
    {"method": "GET", "url": "/test/profile", "params": {"name": "<script>alert('XSS')</script>"}},
    {"method": "GET", "url": "/test/endpoint", "params": {"search": "<img src=x onerror=alert(1)>"}},
    {"method": "POST", "url": "/test/search", "json": {"query": "<svg onload=alert(1)>"}},

    # Command Injection
    {"method": "GET", "url": "/test/endpoint", "params": {"id": "1; cat /etc/passwd"}},
    {"method": "GET", "url": "/test/endpoint", "params": {"search": "$(whoami)"}},

    # Path Traversal
    {"method": "GET", "url": "/test/endpoint", "params": {"id": "../../etc/passwd"}},
    {"method": "GET", "url": "/test/endpoint", "params": {"search": "../../../etc/shadow"}},

    # NoSQL Injection
    {"method": "POST", "url": "/test/login", "json": {"username": {"$gt": ""}, "password": {"$gt": ""}}},
    {"method": "POST", "url": "/test/search", "json": {"query": {"$ne": None}}},

    # XXE
    {"method": "POST", "url": "/test/search", "json": {"query": "<?xml version='1.0'?><!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>"}},

    # LDAP Injection
    {"method": "GET", "url": "/test/endpoint", "params": {"user": "admin)(|(password=*))"}},
    {"method": "GET", "url": "/test/endpoint", "params": {"search": "*)(uid=*))(|(uid=*"}},
]


def send_request(req):
    """Send a single request"""
    try:
        method = req.get("method", "GET")
        url = BASE_URL + req.get("url", "/test/endpoint")
        params = req.get("params")
        json_data = req.get("json")

        if method == "GET":
            response = requests.get(url, params=params, timeout=5)
        else:
            response = requests.post(url, json=json_data, timeout=5)

        return {
            "status": response.status_code,
            "blocked": response.status_code == 403
        }
    except requests.exceptions.Timeout:
        return {"status": 0, "blocked": False, "error": "Timeout"}
    except requests.exceptions.ConnectionError:
        return {"status": 0, "blocked": False, "error": "Connection Error"}
    except Exception as e:
        return {"status": 0, "blocked": False, "error": str(e)}


def generate_traffic(duration_seconds=60, requests_per_second=2):
    """
    Generate continuous traffic for the specified duration

    Args:
        duration_seconds: How long to generate traffic
        requests_per_second: How many requests per second
    """
    print("")
    print("            WAF Live Traffic Generator                             ")
    print("")
    print(f"\nTarget: {BASE_URL}")
    print(f"Duration: {duration_seconds} seconds")
    print(f"Rate: {requests_per_second} requests/second")
    print(f"Total requests: ~{duration_seconds * requests_per_second}")
    print(f"\nStarted at: {datetime.now().strftime('%H:%M:%S')}")
    print("" * 70)

    stats = {
        "total": 0,
        "benign": 0,
        "malicious": 0,
        "blocked": 0,
        "allowed": 0,
        "errors": 0
    }

    start_time = time.time()
    end_time = start_time + duration_seconds

    request_interval = 1.0 / requests_per_second

    try:
        while time.time() < end_time:
            # Randomly choose benign or malicious (70% benign, 30% malicious)
            is_malicious = random.random() < 0.3

            if is_malicious:
                req = random.choice(MALICIOUS_REQUESTS)
                req_type = "ATTACK"
                stats["malicious"] += 1
            else:
                req = random.choice(BENIGN_REQUESTS)
                req_type = "BENIGN"
                stats["benign"] += 1

            result = send_request(req)
            stats["total"] += 1

            if result.get("error"):
                stats["errors"] += 1
                status = f" {result['error']}"
            elif result.get("blocked"):
                stats["blocked"] += 1
                status = " BLOCKED"
            else:
                stats["allowed"] += 1
                status = " ALLOWED"

            # Print progress every 10 requests
            if stats["total"] % 10 == 0:
                elapsed = time.time() - start_time
                rate = stats["total"] / elapsed if elapsed > 0 else 0
                print(f"[{stats['total']:4d}] {req_type:7s} | {status:12s} | "
                      f"Rate: {rate:.1f} req/s | "
                      f"Blocked: {stats['blocked']:3d}/{stats['malicious']:3d} "
                      f"({stats['blocked']/max(1,stats['malicious'])*100:.0f}%)")

            # Wait before next request
            time.sleep(request_interval)

    except KeyboardInterrupt:
        print("\n\n  Traffic generation stopped by user")

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("")
    print("                      TRAFFIC GENERATION SUMMARY                    ")
    print("")
    print(f"  Total Requests Sent:     {stats['total']:6d}                              ")
    print(f"  Benign Requests:         {stats['benign']:6d} ({stats['benign']/max(1,stats['total'])*100:5.1f}%)                  ")
    print(f"  Malicious Requests:      {stats['malicious']:6d} ({stats['malicious']/max(1,stats['total'])*100:5.1f}%)                  ")
    print("")
    print(f"  Requests Blocked:        {stats['blocked']:6d} ({stats['blocked']/max(1,stats['total'])*100:5.1f}%)                  ")
    print(f"  Requests Allowed:        {stats['allowed']:6d} ({stats['allowed']/max(1,stats['total'])*100:5.1f}%)                  ")
    print(f"  Errors:                  {stats['errors']:6d}                              ")
    print("")
    if stats['malicious'] > 0:
        detection_rate = stats['blocked'] / stats['malicious'] * 100
        print(f"  Detection Rate:          {detection_rate:5.1f}%                              ")
    print(f"  Average Rate:            {stats['total']/elapsed_time:5.1f} req/s                         ")
    print(f"  Duration:                {elapsed_time:5.1f} seconds                        ")
    print("")
    print(f"\n Traffic data is now available in the dashboard at http://localhost:3000")
    print(f"   Check the Overview and Traffic pages to see real-time updates!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate live traffic through WAF (backend /test/*). Use a moderate rate so the backend and dashboard stay responsive."
    )
    parser.add_argument(
        "duration",
        nargs="?",
        type=int,
        default=None,
        help="Duration in seconds (ignored if --requests is set)",
    )
    parser.add_argument(
        "rate",
        nargs="?",
        type=int,
        default=None,
        help="Requests per second (default 2, or use with --requests for load test)",
    )
    parser.add_argument(
        "--requests",
        "-n",
        type=int,
        default=None,
        metavar="N",
        help="Total number of requests to send (e.g. 5000). Duration is then N/rate.",
    )
    parser.add_argument(
        "--rate",
        "-r",
        type=int,
        default=None,
        metavar="R",
        help="Req/s when using --requests (default 25). Use moderate rate so WAF/dashboard stay responsive.",
    )
    args = parser.parse_args()

    if args.requests is not None:
        total_requests = args.requests
        rate = args.rate if args.rate is not None else 25
        duration = max(1, (total_requests + rate - 1) // rate)
        print(f"\n Sending {total_requests} requests at {rate} req/s (~{duration}s)\n")
        generate_traffic(duration_seconds=duration, requests_per_second=rate)
    else:
        duration = 60 if args.duration is None else args.duration
        rate = 2 if args.rate is None else args.rate
        print(f"\n Usage: python3 {sys.argv[0]} [duration_seconds] [requests_per_second]")
        print(f"   Or:    python3 {sys.argv[0]} --requests 5000 --rate 25\n")
        generate_traffic(duration_seconds=duration, requests_per_second=rate)
