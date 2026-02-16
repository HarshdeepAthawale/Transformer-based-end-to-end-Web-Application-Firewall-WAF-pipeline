"""Unit tests for backend rate limiter utility."""

import pytest
import time
from backend.utils.rate_limiter import PerIPRateLimiter


@pytest.fixture
def limiter():
    """Rate limiter: 3 requests per 2 second window."""
    return PerIPRateLimiter(max_requests=3, window_seconds=2)


def test_first_request_allowed(limiter):
    """First request from IP should be allowed."""
    assert limiter.is_allowed("192.168.1.1") is True


def test_requests_within_limit_allowed(limiter):
    """Requests within limit should all be allowed."""
    for _ in range(3):
        assert limiter.is_allowed("10.0.0.1") is True


def test_request_over_limit_blocked(limiter):
    """Request over limit should be blocked."""
    for _ in range(3):
        limiter.is_allowed("10.0.0.2")
    assert limiter.is_allowed("10.0.0.2") is False


def test_different_ips_tracked_separately(limiter):
    """Different IPs should have separate limits."""
    for _ in range(3):
        limiter.is_allowed("192.168.1.1")
    assert limiter.is_allowed("192.168.1.1") is False
    assert limiter.is_allowed("192.168.1.2") is True


def test_wait_time_returned_when_blocked(limiter):
    """get_wait_time should return positive value when at limit."""
    for _ in range(3):
        limiter.is_allowed("10.0.0.3")
    wait = limiter.get_wait_time("10.0.0.3")
    assert wait > 0


def test_wait_time_zero_when_no_requests(limiter):
    """get_wait_time should be 0 when IP has no requests in window."""
    assert limiter.get_wait_time("10.0.0.4") == 0.0


def test_stats_reflect_usage(limiter):
    """get_stats should reflect current usage."""
    limiter.is_allowed("10.0.0.5")
    limiter.is_allowed("10.0.0.5")
    stats = limiter.get_stats("10.0.0.5")
    assert stats["current_requests"] == 2
    assert stats["remaining_requests"] == 1
    assert stats["max_requests"] == 3


def test_reset_clears_ip(limiter):
    """reset(ip) should clear that IP's limit."""
    for _ in range(3):
        limiter.is_allowed("10.0.0.6")
    assert limiter.is_allowed("10.0.0.6") is False
    limiter.reset("10.0.0.6")
    assert limiter.is_allowed("10.0.0.6") is True


def test_reset_all_clears_all_ips(limiter):
    """reset() without IP should clear all IPs."""
    limiter.is_allowed("10.0.0.7")
    limiter.is_allowed("10.0.0.8")
    limiter.reset()
    assert limiter.is_allowed("10.0.0.7") is True
    assert limiter.is_allowed("10.0.0.8") is True


def test_window_expiry_allows_new_requests(limiter):
    """After window expires, new requests should be allowed."""
    for _ in range(3):
        limiter.is_allowed("10.0.0.9")
    assert limiter.is_allowed("10.0.0.9") is False
    time.sleep(2.5)
    assert limiter.is_allowed("10.0.0.9") is True
