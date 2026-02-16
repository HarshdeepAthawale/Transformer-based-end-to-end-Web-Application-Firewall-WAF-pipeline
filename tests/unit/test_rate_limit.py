"""Unit tests for gateway rate limiting."""

import pytest

from gateway.rate_limit import RedisRateLimiter


@pytest.fixture
def limiter_fail_open():
    """Rate limiter that fails open when Redis unavailable."""
    return RedisRateLimiter(
        redis_url="redis://localhost:16379",
        max_requests=10,
        window_seconds=60,
        fail_open=True,
    )


@pytest.fixture
def limiter_fail_closed():
    """Rate limiter that fails closed when Redis unavailable."""
    return RedisRateLimiter(
        redis_url="redis://localhost:16379",
        max_requests=10,
        window_seconds=60,
        fail_open=False,
    )


@pytest.mark.asyncio
async def test_fail_open_when_redis_unavailable(limiter_fail_open):
    """When Redis is down and fail_open=True, allow requests."""
    allowed, retry_after = await limiter_fail_open.is_allowed("192.168.1.1")
    assert allowed is True
    assert retry_after == 0.0


@pytest.mark.asyncio
async def test_fail_closed_when_redis_unavailable(limiter_fail_closed):
    """When Redis is down and fail_open=False, block requests."""
    allowed, retry_after = await limiter_fail_closed.is_allowed("192.168.1.1")
    assert allowed is False
    assert retry_after > 0
