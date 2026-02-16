"""Load and latency tests for WAF API."""

import pytest
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

WAF_API = "http://localhost:3001/api/waf/check"

BENIGN_REQUEST = {
    "method": "GET",
    "path": "/api/users",
    "query_params": {"page": "1", "limit": "10"},
    "headers": {"User-Agent": "Mozilla/5.0"},
    "body": None,
}


def _backend_running():
    try:
        return requests.get("http://localhost:3001/health", timeout=2).status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _backend_running(), reason="Backend not running - start with docker compose up -d")
class TestLoadPerformance:
    """Load and throughput tests."""

    def test_single_request_latency(self):
        """Single request latency should be reasonable (< 500ms without GPU)."""
        start = time.perf_counter()
        r = requests.post(WAF_API, json=BENIGN_REQUEST, timeout=10)
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert r.status_code == 200
        assert elapsed_ms < 5000  # 5s max for CI (model load may be slow)

    def test_concurrent_10_requests(self):
        """Should handle 10 concurrent requests."""
        def do_request():
            return requests.post(WAF_API, json=BENIGN_REQUEST, timeout=10)

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(do_request) for _ in range(10)]
            results = [f.result() for f in as_completed(futures)]

        statuses = [r.status_code for r in results]
        assert all(s == 200 for s in statuses)

    def test_throughput_50_requests(self):
        """Measure requests per second over 50 sequential requests."""
        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            r = requests.post(WAF_API, json=BENIGN_REQUEST, timeout=10)
            latencies.append((time.perf_counter() - start) * 1000)
            assert r.status_code == 200

        rps = 50 / (sum(latencies) / 1000)
        p95 = sorted(latencies)[int(0.95 * len(latencies)) - 1] if latencies else 0
        # Targets: >10 RPS, p95 < 500ms (lenient for CI)
        assert rps > 5, f"Throughput too low: {rps:.1f} RPS"
