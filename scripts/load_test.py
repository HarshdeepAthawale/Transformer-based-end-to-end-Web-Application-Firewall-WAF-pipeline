#!/usr/bin/env python3
"""
Week 4 Day 5 — Comprehensive Load Test Suite

Covers all 7 test scenarios from the Day-5 plan:
  1. health_throughput   - Baseline: /health endpoint, no auth
  2. api_throughput      - Single-tenant: /api/dashboard/overview under load
  3. rate_limit          - Rate limit enforcement: expect 429s under burst
  4. waf_inference       - WAF ML inference via /test/endpoint (benign + malicious)
  5. connection_stress   - 100 concurrent DB-backed requests
  6. multi_tenant        - 50 orgs concurrently hitting tenant-scoped endpoints
  7. ml_benchmark        - PyTorch vs ONNX latency (offline, no server needed)

Usage:
  # All scenarios (requires running backend on localhost:3001)
  python scripts/load_test.py

  # Specific scenario
  python scripts/load_test.py --scenario api_throughput

  # Custom host / users / duration
  python scripts/load_test.py --host localhost --port 3001 --users 50 --duration 60

  # Offline ML benchmark only (no server needed)
  python scripts/load_test.py --scenario ml_benchmark

Results are printed to stdout and saved to scripts/data/benchmark_results.json.
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 3001
DEFAULT_USERS = 50
DEFAULT_DURATION = 60  # seconds
DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PASS = "admin123"

BENIGN_PARAMS = [
    "?id=1",
    "?search=hello+world",
    "?user=alice&page=2",
    "?q=product+review",
    "?filter=active&sort=created_at",
]

MALICIOUS_PARAMS = [
    "?search=<script>alert(1)</script>",
    "?id=1' OR '1'='1",
    "?q=../../etc/passwd",
    "?user=admin%27--",
    "?input=;cat /etc/shadow",
]


# ---------------------------------------------------------------------------
# Result accumulator
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    name: str
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    status_counts: Dict[int, int] = field(default_factory=dict)
    latencies_ms: List[float] = field(default_factory=list)
    elapsed_s: float = 0.0
    notes: List[str] = field(default_factory=list)

    @property
    def throughput(self) -> float:
        return self.total_requests / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def percentile(self, pct: float) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        idx = min(int(len(s) * pct), len(s) - 1)
        return s[idx]

    def summary(self) -> Dict:
        return {
            "scenario": self.name,
            "total_requests": self.total_requests,
            "successful": self.successful,
            "failed": self.failed,
            "status_counts": self.status_counts,
            "throughput_rps": round(self.throughput, 2),
            "latency_avg_ms": round(statistics.mean(self.latencies_ms), 2) if self.latencies_ms else 0,
            "latency_p50_ms": round(self.percentile(0.50), 2),
            "latency_p95_ms": round(self.percentile(0.95), 2),
            "latency_p99_ms": round(self.percentile(0.99), 2),
            "latency_max_ms": round(max(self.latencies_ms), 2) if self.latencies_ms else 0,
            "elapsed_s": round(self.elapsed_s, 2),
            "notes": self.notes,
        }

    def passed(self) -> bool:
        """Return True if the scenario meets its pass criteria."""
        criteria = PASS_CRITERIA.get(self.name, {})
        if "min_success_rate" in criteria:
            rate = self.successful / self.total_requests if self.total_requests else 0
            if rate < criteria["min_success_rate"]:
                return False
        if "p99_max_ms" in criteria:
            if self.percentile(0.99) > criteria["p99_max_ms"]:
                return False
        if "expect_429" in criteria and criteria["expect_429"]:
            count_429 = self.status_counts.get(429, 0)
            if count_429 == 0:
                return False
        return True


PASS_CRITERIA = {
    "health_throughput": {"min_success_rate": 0.99, "p99_max_ms": 100},
    "api_throughput": {"min_success_rate": 0.95, "p99_max_ms": 200},
    "rate_limit": {"expect_429": True},
    "waf_inference": {"min_success_rate": 0.90},
    "connection_stress": {"min_success_rate": 0.95, "p99_max_ms": 500},
    "multi_tenant": {"min_success_rate": 0.95},
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

async def get_jwt_token(session: aiohttp.ClientSession, base_url: str,
                        username: str = DEFAULT_ADMIN_USER,
                        password: str = DEFAULT_ADMIN_PASS) -> Optional[str]:
    """Authenticate and return a JWT token, or None on failure."""
    try:
        async with session.post(
            f"{base_url}/api/users/login",
            json={"username": username, "password": password},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("data", {}).get("token")
            text = await resp.text()
            print(f"  [auth] Login failed ({resp.status}): {text[:120]}")
            return None
    except Exception as e:
        print(f"  [auth] Login error: {e}")
        return None


async def timed_get(session: aiohttp.ClientSession, url: str,
                    headers: Optional[Dict] = None) -> Dict:
    t0 = time.perf_counter()
    try:
        async with session.get(
            url,
            headers=headers or {},
            timeout=aiohttp.ClientTimeout(total=15),
            allow_redirects=False,
        ) as resp:
            await resp.read()
            return {"status": resp.status, "latency_ms": (time.perf_counter() - t0) * 1000}
    except asyncio.TimeoutError:
        return {"status": 0, "latency_ms": (time.perf_counter() - t0) * 1000, "error": "timeout"}
    except Exception as e:
        return {"status": 0, "latency_ms": (time.perf_counter() - t0) * 1000, "error": str(e)}


async def timed_post(session: aiohttp.ClientSession, url: str,
                     payload: Dict, headers: Optional[Dict] = None) -> Dict:
    t0 = time.perf_counter()
    try:
        async with session.post(
            url,
            json=payload,
            headers=headers or {},
            timeout=aiohttp.ClientTimeout(total=15),
            allow_redirects=False,
        ) as resp:
            await resp.read()
            return {"status": resp.status, "latency_ms": (time.perf_counter() - t0) * 1000}
    except asyncio.TimeoutError:
        return {"status": 0, "latency_ms": (time.perf_counter() - t0) * 1000, "error": "timeout"}
    except Exception as e:
        return {"status": 0, "latency_ms": (time.perf_counter() - t0) * 1000, "error": str(e)}


def _record(result: ScenarioResult, r: Dict):
    result.total_requests += 1
    result.latencies_ms.append(r["latency_ms"])
    status = r["status"]
    result.status_counts[status] = result.status_counts.get(status, 0) + 1
    if 200 <= status < 300:
        result.successful += 1
    else:
        result.failed += 1


# ---------------------------------------------------------------------------
# Scenario 1: Health endpoint throughput (baseline, no auth)
# ---------------------------------------------------------------------------

async def run_health_throughput(base_url: str, users: int, duration: int) -> ScenarioResult:
    result = ScenarioResult("health_throughput")
    url = f"{base_url}/health"
    deadline = time.monotonic() + duration

    async def worker():
        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                r = await timed_get(session, url)
                _record(result, r)

    t0 = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(users)])
    result.elapsed_s = time.perf_counter() - t0
    result.notes.append(f"Target: P99 < 100ms, success rate > 99%")
    return result


# ---------------------------------------------------------------------------
# Scenario 2: Single-tenant API throughput
# ---------------------------------------------------------------------------

async def run_api_throughput(base_url: str, users: int, duration: int) -> ScenarioResult:
    result = ScenarioResult("api_throughput")
    url = f"{base_url}/api/dashboard/overview"
    deadline = time.monotonic() + duration

    async with aiohttp.ClientSession() as bootstrap:
        token = await get_jwt_token(bootstrap, base_url)

    if not token:
        result.notes.append("SKIPPED: could not obtain JWT token (is the backend running with seeded admin?)")
        return result

    headers = {"Authorization": f"Bearer {token}"}

    async def worker():
        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                r = await timed_get(session, url, headers=headers)
                _record(result, r)

    t0 = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(users)])
    result.elapsed_s = time.perf_counter() - t0
    result.notes.append("Target: P99 < 200ms, success rate > 95%")
    return result


# ---------------------------------------------------------------------------
# Scenario 3: Rate limit enforcement — expect 429s under burst
# ---------------------------------------------------------------------------

async def run_rate_limit(base_url: str, **_) -> ScenarioResult:
    """
    Fire 400 requests as fast as possible from a single session.
    The rate limit middleware defaults to 300 req/min per IP,
    so we should hit 429 before we exhaust the burst.
    """
    result = ScenarioResult("rate_limit")
    url = f"{base_url}/health"
    BURST = 400

    async with aiohttp.ClientSession() as session:
        t0 = time.perf_counter()
        tasks = [timed_get(session, url) for _ in range(BURST)]
        responses = await asyncio.gather(*tasks)
        result.elapsed_s = time.perf_counter() - t0

    for r in responses:
        _record(result, r)

    count_429 = result.status_counts.get(429, 0)
    result.notes.append(
        f"Fired {BURST} requests as fast as possible. "
        f"Got {count_429} x 429 (rate limited), "
        f"{result.status_counts.get(200, 0)} x 200."
    )
    result.notes.append("Target: at least 1 x 429 within the burst")
    return result


# ---------------------------------------------------------------------------
# Scenario 4: WAF inference — benign vs malicious via /test/endpoint
# ---------------------------------------------------------------------------

async def run_waf_inference(base_url: str, users: int, **_) -> ScenarioResult:
    """
    Send a mix of benign and malicious requests through /test/endpoint.
    WAF middleware classifies them and either allows (200) or blocks (403).
    """
    result = ScenarioResult("waf_inference")
    params_list = BENIGN_PARAMS * 4 + MALICIOUS_PARAMS * 4  # 40 requests per worker

    async def worker():
        async with aiohttp.ClientSession() as session:
            for params in params_list:
                url = f"{base_url}/test/endpoint{params}"
                r = await timed_get(session, url)
                _record(result, r)

    t0 = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(users)])
    result.elapsed_s = time.perf_counter() - t0

    benign_total = users * 4 * len(BENIGN_PARAMS)
    malicious_total = users * 4 * len(MALICIOUS_PARAMS)
    blocked = result.status_counts.get(403, 0)
    result.notes.append(
        f"Sent ~{benign_total} benign + ~{malicious_total} malicious requests. "
        f"WAF blocked {blocked} requests (403)."
    )
    result.notes.append("WAF_ENABLED=false means all pass 200; with model loaded expect 403 on malicious")
    return result


# ---------------------------------------------------------------------------
# Scenario 5: Connection pool stress — 100 concurrent DB-backed requests
# ---------------------------------------------------------------------------

async def run_connection_stress(base_url: str, **_) -> ScenarioResult:
    """
    Send 500 requests with concurrency=100 to DB-backed endpoints.
    Pool size=20 + overflow=10 (Week 4 Day 1). Expect no 500s due to pool exhaustion.
    """
    result = ScenarioResult("connection_stress")
    CONCURRENCY = 100
    TOTAL = 500

    async with aiohttp.ClientSession() as bootstrap:
        token = await get_jwt_token(bootstrap, base_url)

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    endpoints = [
        f"{base_url}/api/metrics/realtime",
        f"{base_url}/api/events",
        f"{base_url}/api/dashboard/overview",
    ]

    sem = asyncio.Semaphore(CONCURRENCY)

    async def bounded_request(session: aiohttp.ClientSession, idx: int) -> Dict:
        async with sem:
            url = endpoints[idx % len(endpoints)]
            return await timed_get(session, url, headers=headers)

    t0 = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [bounded_request(session, i) for i in range(TOTAL)]
        responses = await asyncio.gather(*tasks)
    result.elapsed_s = time.perf_counter() - t0

    for r in responses:
        _record(result, r)

    count_500 = result.status_counts.get(500, 0)
    result.notes.append(
        f"Concurrency={CONCURRENCY}, total={TOTAL}. "
        f"DB pool_size=20, max_overflow=10. "
        f"500 errors (pool exhaustion indicator): {count_500}"
    )
    result.notes.append("Target: 0 x 500, P99 < 500ms, success rate > 95%")
    if not token:
        result.notes.append("WARNING: no auth token — 401s are expected, not pool failures")
    return result


# ---------------------------------------------------------------------------
# Scenario 6: Multi-tenant isolation — 50 concurrent tenants
# ---------------------------------------------------------------------------

async def run_multi_tenant(base_url: str, users: int, duration: int) -> ScenarioResult:
    """
    Simulate N concurrent tenants each hitting their own dashboard.
    All use the same admin JWT (single-org test environment), so the goal is to
    verify the server stays stable under concurrent tenant-scoped load.
    In a multi-org setup each would use a different token with a different org_id.
    """
    result = ScenarioResult("multi_tenant")
    deadline = time.monotonic() + duration

    async with aiohttp.ClientSession() as bootstrap:
        token = await get_jwt_token(bootstrap, base_url)

    if not token:
        result.notes.append("SKIPPED: could not obtain JWT token")
        return result

    # Each simulated tenant uses the same token (single-seeded-org test env).
    # In production each tenant would have their own JWT with their own org_id.
    tenant_endpoints = [
        f"{base_url}/api/dashboard/overview",
        f"{base_url}/api/metrics/realtime",
        f"{base_url}/api/events",
    ]

    async def tenant_worker(tenant_idx: int):
        headers = {"Authorization": f"Bearer {token}"}
        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                url = tenant_endpoints[tenant_idx % len(tenant_endpoints)]
                r = await timed_get(session, url, headers=headers)
                _record(result, r)

    t0 = time.perf_counter()
    await asyncio.gather(*[tenant_worker(i) for i in range(users)])
    result.elapsed_s = time.perf_counter() - t0

    result.notes.append(
        f"Simulated {users} concurrent tenants for {duration}s. "
        "All scoped by org_id from JWT. "
        "Cross-tenant leak would manifest as 403 or mismatched data (manual review required)."
    )
    result.notes.append("Target: success rate > 95%, no 500 errors")
    return result


# ---------------------------------------------------------------------------
# Scenario 7: ML inference latency benchmark (offline, no server needed)
# ---------------------------------------------------------------------------

def run_ml_benchmark() -> ScenarioResult:
    """
    Direct Python benchmark: PyTorch WAFClassifier vs ONNXWAFClassifier.
    100 samples, measure total and per-sample latency.
    Requires models/waf-distilbert and optionally the exported .onnx file.
    """
    result = ScenarioResult("ml_benchmark")
    result.elapsed_s = 0.0

    SAMPLES = 100
    TEST_TEXTS = [
        "GET /api/users HTTP/1.1",
        "SELECT * FROM users WHERE id=1",
        "<script>alert('xss')</script>",
        "POST /login username=admin&password=admin",
        "../../etc/passwd",
        "normal product search query",
        "'; DROP TABLE users; --",
        "curl -X GET http://example.com/api/data",
        "User-Agent: Mozilla/5.0",
        "id=1 UNION SELECT username,password FROM users--",
    ] * (SAMPLES // 10)

    # --- PyTorch ---
    pytorch_ms = None
    try:
        from backend.ml.waf_classifier import WAFClassifier

        clf = WAFClassifier()
        if not clf.is_loaded:
            result.notes.append("PyTorch model not loaded (models/waf-distilbert missing) — skipped")
        else:
            t0 = time.perf_counter()
            for text in TEST_TEXTS:
                clf.classify(text)
            pytorch_ms = (time.perf_counter() - t0) * 1000
            result.notes.append(
                f"PyTorch  {SAMPLES}x: {pytorch_ms:.1f}ms "
                f"({pytorch_ms / SAMPLES:.2f}ms/req)"
            )
    except Exception as e:
        result.notes.append(f"PyTorch benchmark error: {e}")

    # --- ONNX ---
    onnx_ms = None
    try:
        from backend.ml.onnx_classifier import ONNXWAFClassifier

        onnx_clf = ONNXWAFClassifier()
        if not onnx_clf.is_loaded:
            result.notes.append("ONNX model not loaded (models/waf-distilbert/model.onnx missing) — skipped")
            result.notes.append("Export first: WAF_USE_ONNX=true python scripts/export_onnx.py")
        else:
            t0 = time.perf_counter()
            for text in TEST_TEXTS:
                onnx_clf.classify(text)
            onnx_ms = (time.perf_counter() - t0) * 1000
            result.notes.append(
                f"ONNX     {SAMPLES}x: {onnx_ms:.1f}ms "
                f"({onnx_ms / SAMPLES:.2f}ms/req)"
            )
    except Exception as e:
        result.notes.append(f"ONNX benchmark error: {e}")

    if pytorch_ms and onnx_ms:
        speedup = pytorch_ms / onnx_ms
        result.notes.append(f"Speedup: {speedup:.2f}x (target: 3-5x)")
        result.notes.append("PASS" if speedup >= 3.0 else f"BELOW TARGET (got {speedup:.2f}x, need 3x)")
    elif not pytorch_ms and not onnx_ms:
        result.notes.append("Both models unavailable — run from project root with models present")

    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"


def print_result(r: ScenarioResult):
    s = r.summary()
    skipped = any("SKIPPED" in n for n in r.notes)
    verdict = SKIP if skipped else (PASS if r.passed() else FAIL)

    print(f"\n{'='*64}")
    print(f"  {s['scenario'].upper().replace('_', ' ')}  [{verdict}]")
    print(f"{'='*64}")
    if not skipped:
        print(f"  Requests   : {s['total_requests']}  "
              f"({s['successful']} ok, {s['failed']} fail)")
        print(f"  Throughput : {s['throughput_rps']} req/s")
        print(f"  Latency    : avg={s['latency_avg_ms']}ms  "
              f"p50={s['latency_p50_ms']}ms  "
              f"p95={s['latency_p95_ms']}ms  "
              f"p99={s['latency_p99_ms']}ms  "
              f"max={s['latency_max_ms']}ms")
        if s["status_counts"]:
            counts = "  ".join(f"HTTP {k}={v}" for k, v in sorted(s["status_counts"].items()))
            print(f"  Status     : {counts}")
    for note in s["notes"]:
        print(f"  NOTE: {note}")


def save_results(results: List[ScenarioResult], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "host": f"{DEFAULT_HOST}:{DEFAULT_PORT}",
        "results": [r.summary() for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_SCENARIOS = [
    "health_throughput",
    "api_throughput",
    "rate_limit",
    "waf_inference",
    "connection_stress",
    "multi_tenant",
    "ml_benchmark",
]


async def async_main(args):
    base_url = f"http://{args.host}:{args.port}"
    results: List[ScenarioResult] = []

    scenarios = [args.scenario] if args.scenario else ALL_SCENARIOS

    for name in scenarios:
        print(f"\nRunning scenario: {name} ...")

        if name == "ml_benchmark":
            r = run_ml_benchmark()
        elif name == "health_throughput":
            r = await run_health_throughput(base_url, args.users, args.duration)
        elif name == "api_throughput":
            r = await run_api_throughput(base_url, args.users, args.duration)
        elif name == "rate_limit":
            r = await run_rate_limit(base_url)
        elif name == "waf_inference":
            r = await run_waf_inference(base_url, args.users)
        elif name == "connection_stress":
            r = await run_connection_stress(base_url)
        elif name == "multi_tenant":
            r = await run_multi_tenant(base_url, args.users, args.duration)
        else:
            print(f"  Unknown scenario: {name}")
            continue

        print_result(r)
        results.append(r)

    # Summary table
    print(f"\n{'='*64}")
    print("  SUMMARY")
    print(f"{'='*64}")
    passed = sum(1 for r in results if r.passed() and not any("SKIPPED" in n for n in r.notes))
    skipped = sum(1 for r in results if any("SKIPPED" in n for n in r.notes))
    failed = len(results) - passed - skipped
    print(f"  {'PASS':<8} {passed}")
    print(f"  {'SKIP':<8} {skipped}")
    print(f"  {'FAIL':<8} {failed}")

    output = Path(__file__).parent / "data" / "benchmark_results.json"
    save_results(results, output)

    return 0 if failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description="Week 4 Day 5 — WAF load test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {s}" for s in ALL_SCENARIOS),
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--users", type=int, default=DEFAULT_USERS,
                        help="Concurrent users per scenario (default 50)")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                        help="Duration in seconds for time-based scenarios (default 60)")
    parser.add_argument("--scenario", choices=ALL_SCENARIOS, default=None,
                        help="Run a single scenario (default: all)")
    parser.add_argument("--admin-user", default=DEFAULT_ADMIN_USER)
    parser.add_argument("--admin-pass", default=DEFAULT_ADMIN_PASS)
    args = parser.parse_args()

    # Allow overriding credentials at module level for convenience
    global DEFAULT_ADMIN_USER, DEFAULT_ADMIN_PASS
    DEFAULT_ADMIN_USER = args.admin_user
    DEFAULT_ADMIN_PASS = args.admin_pass

    exit_code = asyncio.run(async_main(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
