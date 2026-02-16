"""
Redis-backed rate limiter for the WAF Gateway.
Uses sliding window algorithm for accurate per-IP throttling.
"""

import time
from typing import Optional, Tuple

from loguru import logger

from gateway.config import gateway_config


class RedisRateLimiter:
    """Per-IP rate limiter using Redis sliding window."""

    def __init__(
        self,
        redis_url: str,
        max_requests: int = 120,
        window_seconds: int = 60,
        fail_open: bool = True,
    ):
        self.redis_url = redis_url
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.fail_open = fail_open
        self._redis = None
        self._connected = False
        self._init_redis()

    def _init_redis(self) -> None:
        """Initialize Redis connection."""
        try:
            import redis.asyncio as redis_async

            self._redis = redis_async.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            self._connected = True
            logger.info("Rate limiter: Redis connected")
        except ImportError:
            logger.warning("redis package not installed; rate limiting disabled")
            self._redis = None
            self._connected = False
        except Exception as e:
            logger.warning(f"Rate limiter: Redis connection failed ({e}); will fail-open")
            self._redis = None
            self._connected = False

    def _key(self, ip: str) -> str:
        return f"rl:ip:{ip}"

    async def is_allowed(self, ip: str) -> Tuple[bool, float]:
        """
        Check if request from IP is allowed.

        Returns:
            (allowed, retry_after_seconds)
            If allowed=True, retry_after is 0.
            If allowed=False, retry_after is seconds until window allows next request.
        """
        if not self._connected or self._redis is None:
            return (True, 0.0) if self.fail_open else (False, self.window_seconds)

        now = time.time()
        key = self._key(ip)
        window_start = now - self.window_seconds

        try:
            pipe = self._redis.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zadd(key, {str(now): now})
            pipe.zcard(key)
            pipe.expire(key, self.window_seconds + 10)
            results = await pipe.execute()

            count = results[2] if len(results) > 2 else 0

            if count > self.max_requests:
                # Get oldest timestamp to compute retry_after
                oldest = await self._redis.zrange(key, 0, 0, withscores=True)
                if oldest:
                    oldest_ts = oldest[0][1]
                    retry_after = max(0, self.window_seconds - (now - oldest_ts))
                else:
                    retry_after = self.window_seconds
                return False, retry_after

            return True, 0.0

        except Exception as e:
            logger.warning(f"Rate limiter Redis error: {e}")
            return (True, 0.0) if self.fail_open else (False, self.window_seconds)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            try:
                await self._redis.close()
            except Exception:
                pass
            self._redis = None
            self._connected = False


def create_rate_limiter() -> Optional[RedisRateLimiter]:
    """Create rate limiter from gateway config."""
    if not gateway_config.RATE_LIMIT_ENABLED:
        return None

    limiter = RedisRateLimiter(
        redis_url=gateway_config.REDIS_URL,
        max_requests=gateway_config.RATE_LIMIT_REQUESTS_PER_MINUTE,
        window_seconds=gateway_config.RATE_LIMIT_WINDOW_SECONDS,
        fail_open=gateway_config.RATE_LIMIT_FAIL_OPEN,
    )

    if not limiter._connected:
        return None

    return limiter
