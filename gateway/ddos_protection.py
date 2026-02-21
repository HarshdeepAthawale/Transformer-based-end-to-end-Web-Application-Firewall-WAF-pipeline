"""
DDoS protection layer for the WAF Gateway.
Handles burst detection, request size limits, and temporary IP blocking.
Supports adaptive threshold from Redis (backend job writes; gateway reads).
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
        adaptive_enabled: bool = False,
        adaptive_redis_key: str = "waf:ddos:adaptive_threshold",
        adaptive_learning_window_minutes: int = 60,
        adaptive_refresh_seconds: int = 60,
    ):
        self.redis_url = redis_url
        self.max_body_bytes = max_body_bytes
        self.burst_threshold = burst_threshold
        self.burst_window_seconds = burst_window_seconds
        self.block_duration_seconds = block_duration_seconds
        self.fail_open = fail_open
        self.adaptive_enabled = adaptive_enabled
        self.adaptive_redis_key = adaptive_redis_key
        self.adaptive_learning_window_minutes = adaptive_learning_window_minutes
        self.adaptive_refresh_seconds = adaptive_refresh_seconds
        self._redis = None
        self._connected = False
        self._adaptive_threshold_cache: Tuple[float, int] = (0.0, burst_threshold)  # (timestamp, value)
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

    def _adaptive_counter_key(self, ip: str) -> str:
        """Key for learning: ddos:adaptive:{minute}:{ip} (backend job scans ddos:adaptive:*)."""
        minute_bucket = int(time.time()) // 60
        return f"ddos:adaptive:{minute_bucket}:{ip}"

    async def _get_effective_burst_threshold(self) -> int:
        """Return burst threshold: from Redis if adaptive enabled, else static."""
        if not self.adaptive_enabled or not self._connected or self._redis is None:
            return self.burst_threshold
        now = time.monotonic()
        cached_ts, cached_val = self._adaptive_threshold_cache
        if now - cached_ts < self.adaptive_refresh_seconds:
            return cached_val
        try:
            val = await self._redis.get(self.adaptive_redis_key)
            if val is not None:
                threshold = int(val)
                self._adaptive_threshold_cache = (now, threshold)
                return threshold
        except Exception as e:
            logger.debug(f"DDoS adaptive threshold read failed: {e}")
        return self.burst_threshold

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
        When adaptive enabled, also increments learning counter and uses threshold from Redis.
        """
        if not self._connected or self._redis is None:
            return True, False

        effective_threshold = await self._get_effective_burst_threshold()
        now = time.time()
        burst_key = self._burst_key(ip)
        blocked_key = self._blocked_key(ip)

        try:
            pipe = self._redis.pipeline()
            pipe.zremrangebyscore(burst_key, 0, now - self.burst_window_seconds)
            pipe.zadd(burst_key, {str(now): now})
            pipe.zcard(burst_key)
            pipe.expire(burst_key, self.burst_window_seconds + 10)
            # Learning counter for adaptive job (per-IP per-minute)
            if self.adaptive_enabled:
                adj_key = self._adaptive_counter_key(ip)
                pipe.incr(adj_key)
                pipe.expire(adj_key, self.adaptive_learning_window_minutes * 60 + 120)
            results = await pipe.execute()

            count = results[2] if len(results) > 2 else 0

            if count >= effective_threshold:
                await self._redis.setex(
                    blocked_key,
                    self.block_duration_seconds,
                    "1",
                )
                logger.warning(
                    f"DDoS: IP {ip} exceeded burst threshold "
                    f"({count} >= {effective_threshold} in {self.burst_window_seconds}s); "
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
        adaptive_enabled=getattr(gateway_config, "ADAPTIVE_DDOS_ENABLED", False),
        adaptive_redis_key=getattr(gateway_config, "ADAPTIVE_DDOS_REDIS_KEY", "waf:ddos:adaptive_threshold"),
        adaptive_learning_window_minutes=getattr(gateway_config, "ADAPTIVE_DDOS_LEARNING_WINDOW_MINUTES", 60),
        adaptive_refresh_seconds=getattr(gateway_config, "ADAPTIVE_DDOS_REFRESH_SECONDS", 60),
    )

    if not protection._connected:
        return None

    return protection
