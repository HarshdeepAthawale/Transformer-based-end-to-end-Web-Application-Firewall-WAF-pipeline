"""Unit tests for gateway DDoS protection."""

import pytest

from gateway.ddos_protection import DDoSProtection


def test_check_request_size_allowed_none():
    """Content-Length None is allowed."""
    protection = DDoSProtection(
        redis_url="redis://localhost:6379",
        max_body_bytes=1024,
    )
    allowed, reason = protection.check_request_size(None)
    assert allowed is True
    assert reason == ""


def test_check_request_size_allowed_small():
    """Small content length is allowed."""
    protection = DDoSProtection(
        redis_url="redis://localhost:6379",
        max_body_bytes=10 * 1024 * 1024,
    )
    allowed, reason = protection.check_request_size(1000)
    assert allowed is True
    assert reason == ""


def test_check_request_size_rejected():
    """Content length exceeding max is rejected."""
    protection = DDoSProtection(
        redis_url="redis://localhost:6379",
        max_body_bytes=1024,
    )
    allowed, reason = protection.check_request_size(2048)
    assert allowed is False
    assert "2048" in reason
    assert "1024" in reason


@pytest.mark.asyncio
async def test_is_blocked_when_redis_unavailable():
    """When Redis unavailable and fail_open=True, IP is not blocked."""
    protection = DDoSProtection(
        redis_url="redis://localhost:16379",
        fail_open=True,
    )
    is_blocked, ttl = await protection.is_blocked("192.168.1.1")
    assert is_blocked is False
    assert ttl == 0.0
