"""
DDoS protection layer for the WAF Gateway.
Handles burst detection, request size limits, and temporary IP blocking.
"""

import time
from typing import Optional, Tuple

from loguru import logger

from gateway.config import gateway_config


class DDoSProtection:
    """L7 DDoS protection: burst detection and request size limits."""

    def __init__(
        self,
        redis_url: str,
        max_body_bytes: int = 10 * 1024 * 1024,
        burst_threshold: int = 50,
        burst_window_seconds: int = 5,
        block_duration_seconds: int = 60,
        fail_open: bool = True,
    ):
        self.redis_url = redis_url
        self.max_body_bytes = max_body_bytes
        self.burst_threshold = burst_threshold
        self.burst_window_seconds = burst_window_seconds
        self.block_duration_seconds = block_duration_seconds
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
            logger.info("DDoS protection: Redis connected")
        except ImportError:
            logger.warning("redis package not installed; DDoS protection disabled")
            self._redis = None
            self._connected = False
        except Exception as e:
            logger.warning(f"DDoS protection: Redis connection failed ({e}); will fail-open")
            self._redis = None
            self._connected = False

    def _burst_key(self, ip: str) -> str:
        return f"ddos:burst:{ip}"

    def _blocked_key(self, ip: str) -> str:
        return f"ddos:blocked:{ip}"

    def check_request_size(self, content_length: Optional[int]) -> Tuple[bool, str]:
        """
        Check if request body size is within limits (before reading body).

        Returns:
            (allowed, reason) - if not allowed, reason explains why.
        """
        if content_length is None:
            return True, ""

        if content_length > self.max_body_bytes:
            return False, f"Request body too large ({content_length} > {self.max_body_bytes})"

        return True, ""

    async def is_blocked(self, ip: str) -> Tuple[bool, float]:
        """
        Check if IP is temporarily blocked due to burst detection.

        Returns:
            (is_blocked, ttl_seconds) - if blocked, ttl is remaining block duration.
        """
        if not self._connected or self._redis is None:
            return False, 0.0

        key = self._blocked_key(ip)
        try:
            ttl = await self._redis.ttl(key)
            if ttl > 0:
                return True, float(ttl)
            return False, 0.0
        except Exception as e:
            logger.warning(f"DDoS check blocked: Redis error {e}")
            return (False, 0.0) if self.fail_open else (True, float(self.block_duration_seconds))

    async def record_request_and_check_burst(self, ip: str) -> Tuple[bool, bool]:
        """
        Record request and check if IP exceeds burst threshold.
        If burst exceeded, blocks the IP for block_duration_seconds.

        Returns:
            (allowed, triggered_block) - if triggered_block, we just blocked this IP.
        """
        if not self._connected or self._redis is None:
            return True, False

        now = time.time()
        burst_key = self._burst_key(ip)
        blocked_key = self._blocked_key(ip)

        try:
            pipe = self._redis.pipeline()
            pipe.zremrangebyscore(burst_key, 0, now - self.burst_window_seconds)
            pipe.zadd(burst_key, {str(now): now})
            pipe.zcard(burst_key)
            pipe.expire(burst_key, self.burst_window_seconds + 10)
            results = await pipe.execute()

            count = results[2] if len(results) > 2 else 0

            if count >= self.burst_threshold:
                await self._redis.setex(
                    blocked_key,
                    self.block_duration_seconds,
                    "1",
                )
                logger.warning(
                    f"DDoS: IP {ip} exceeded burst threshold "
                    f"({count} req in {self.burst_window_seconds}s); "
                    f"blocked for {self.block_duration_seconds}s"
                )
                return False, True

            return True, False

        except Exception as e:
            logger.warning(f"DDoS burst check: Redis error {e}")
            return (True, False) if self.fail_open else (False, False)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            try:
                await self._redis.close()
            except Exception:
                pass
            self._redis = None
            self._connected = False


def create_ddos_protection() -> Optional[DDoSProtection]:
    """Create DDoS protection from gateway config."""
    if not gateway_config.DDOS_ENABLED:
        return None

    protection = DDoSProtection(
        redis_url=gateway_config.REDIS_URL,
        max_body_bytes=gateway_config.DDOS_MAX_BODY_BYTES,
        burst_threshold=gateway_config.DDOS_BURST_THRESHOLD,
        burst_window_seconds=gateway_config.DDOS_BURST_WINDOW_SECONDS,
        block_duration_seconds=gateway_config.DDOS_BLOCK_DURATION_SECONDS,
        fail_open=gateway_config.DDOS_FAIL_OPEN,
    )

    if not protection._connected:
        return None

    return protection
