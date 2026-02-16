"""
Redis-backed IP blacklist for B2B scalable blocking.
Backend syncs blacklist to Redis; gateway checks before rate limit/DDoS.
Supports exact IPs and CIDR ranges (synced via blacklist_sync).
"""

from typing import Optional, Tuple

from loguru import logger

from gateway.config import gateway_config


def _blacklist_key(ip: str, tenant_id: str = "default") -> str:
    """Redis key for blacklist. tenant_id enables B2B multi-tenant isolation."""
    return f"blacklist:{tenant_id}:{ip}"


def _cidr_key(tenant_id: str = "default") -> str:
    """Redis key for CIDR blacklist set."""
    return f"blacklist:{tenant_id}:cidr"


class RedisBlacklistChecker:
    """Check if IP is blacklisted via Redis (synced from backend)."""

    def __init__(self, redis_url: str, tenant_id: str = "default", fail_open: bool = True):
        self.redis_url = redis_url
        self.tenant_id = tenant_id
        self.fail_open = fail_open
        self._redis = None
        self._connected = False
        self._init_redis()

    def _init_redis(self) -> None:
        try:
            import redis.asyncio as redis_async

            self._redis = redis_async.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
            )
            self._connected = True
            logger.info("Blacklist checker: Redis connected")
        except ImportError:
            logger.warning("redis package not installed; IP blacklist disabled")
            self._redis = None
            self._connected = False
        except Exception as e:
            logger.warning(f"Blacklist checker: Redis connection failed ({e}); will fail-open")
            self._redis = None
            self._connected = False

    async def is_blocked(self, ip: str) -> Tuple[bool, Optional[str]]:
        """
        Check if IP is blacklisted (exact match or CIDR).
        Returns: (is_blocked, reason or None)
        """
        if not self._connected or self._redis is None:
            return (False, None) if self.fail_open else (True, "Blacklist unavailable")

        try:
            # 1. Check exact IP
            key = _blacklist_key(ip, self.tenant_id)
            val = await self._redis.get(key)
            if val:
                try:
                    import json
                    data = json.loads(val)
                    reason = data.get("reason", "IP is blacklisted")
                except (json.JSONDecodeError, TypeError):
                    reason = "IP is blacklisted"
                return True, reason

            # 2. Check CIDR ranges
            cidr_key = _cidr_key(self.tenant_id)
            cidrs = await self._redis.smembers(cidr_key)
            if cidrs:
                import ipaddress
                try:
                    ip_obj = ipaddress.ip_address(ip)
                    for cidr in cidrs:
                        try:
                            network = ipaddress.ip_network(cidr, strict=False)
                            if ip_obj in network:
                                return True, "IP is in blacklisted range"
                        except ValueError:
                            continue
                except ValueError:
                    pass
            return False, None
        except Exception as e:
            logger.debug(f"Blacklist check error: {e}")
            return (False, None) if self.fail_open else (True, "Blacklist check failed")

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            self._connected = False


def create_blacklist_checker() -> Optional[RedisBlacklistChecker]:
    """Create blacklist checker from gateway config."""
    if not getattr(gateway_config, "BLACKLIST_ENABLED", True):
        return None
    redis_url = gateway_config.REDIS_URL
    if not redis_url:
        return None
    tenant_id = getattr(gateway_config, "BLACKLIST_TENANT_ID", "default")
    fail_open = getattr(gateway_config, "BLACKLIST_FAIL_OPEN", True)
    checker = RedisBlacklistChecker(redis_url=redis_url, tenant_id=tenant_id, fail_open=fail_open)
    return checker if checker._connected else None
