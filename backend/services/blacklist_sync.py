"""
Sync IP blacklist to Redis for gateway enforcement (B2B scalable).
Backend is source of truth (DB); Redis is fast lookup for gateway.

REDIS_URL is REQUIRED for Block IP to work. Without Redis, blocked IPs are stored
in the database but not enforced at the gateway.
"""

import json
import os
from datetime import datetime

from loguru import logger

from backend.lib.datetime_utils import utc_now
from typing import Optional


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
BLACKLIST_TENANT_ID = os.getenv("BLACKLIST_TENANT_ID", "default")

REDIS_REQUIRED_MSG = (
    "REDIS_URL is required for Block IP enforcement. Set REDIS_URL and ensure Redis is running. "
    "Block IP from the dashboard will not take effect at the gateway without Redis."
)


def _key(ip: str) -> str:
    return f"blacklist:{BLACKLIST_TENANT_ID}:{ip}"


def _cidr_key() -> str:
    """Redis key for CIDR blacklist set (tenant-scoped)."""
    return f"blacklist:{BLACKLIST_TENANT_ID}:cidr"


def _get_redis():
    """Get sync Redis client. Returns None if unavailable."""
    try:
        import redis
        r = redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        r.ping()
        return r
    except Exception as e:
        logger.debug(f"Blacklist sync: Redis unavailable ({e})")
        return None


def is_redis_available() -> bool:
    """Check if Redis is available for blacklist sync."""
    return _get_redis() is not None


def sync_add(ip: str, reason: Optional[str] = None, expires_at: Optional[datetime] = None) -> bool:
    """Add IP to Redis blacklist. Returns True if synced. Raises RuntimeError if Redis is required but unavailable."""
    r = _get_redis()
    if not r:
        raise RuntimeError(REDIS_REQUIRED_MSG)
    key = _key(ip)
    val = json.dumps({"reason": reason or "IP is blacklisted"})
    try:
        if expires_at:
            ttl = max(1, int((expires_at - utc_now()).total_seconds()))
            r.setex(key, ttl, val)
        else:
            r.set(key, val)
        return True
    except Exception as e:
        logger.warning(f"Blacklist sync add failed: {e}")
        raise RuntimeError(REDIS_REQUIRED_MSG) from e


def sync_remove(ip: str) -> bool:
    """Remove IP from Redis blacklist. Returns True if synced. Raises RuntimeError if Redis is required but unavailable."""
    r = _get_redis()
    if not r:
        raise RuntimeError(REDIS_REQUIRED_MSG)
    key = _key(ip)
    try:
        r.delete(key)
        return True
    except Exception as e:
        logger.warning(f"Blacklist sync remove failed: {e}")
        raise RuntimeError(REDIS_REQUIRED_MSG) from e


def _normalize_cidr(cidr: str) -> str | None:
    """Validate and normalize CIDR. Returns None if invalid."""
    try:
        import ipaddress
        return str(ipaddress.ip_network(cidr, strict=False))
    except ValueError:
        return None


def sync_full_blacklist(entries: list) -> int:
    """
    Sync full blacklist to Redis. Replaces existing keys for tenant.
    Supports exact IPs and CIDR ranges (is_range=True, ip_range or ip).
    entries: list of IPBlacklist or dict with ip, ip_range, reason, expires_at, is_range
    Returns count synced.
    Raises RuntimeError if entries exist and Redis is unavailable (fail visibly).
    """
    if not entries:
        return 0
    r = _get_redis()
    if not r:
        raise RuntimeError(REDIS_REQUIRED_MSG)
    count = 0
    prefix = f"blacklist:{BLACKLIST_TENANT_ID}:"
    cidr_key = _cidr_key()
    try:
        # Delete stale keys for this tenant (we'll repopulate)
        for k in r.scan_iter(match=f"{prefix}*"):
            r.delete(k)
        cidrs_to_add = []
        for e in entries:
            ip = e.ip if hasattr(e, "ip") else e.get("ip")
            is_range = getattr(e, "is_range", None) or e.get("is_range", False)
            ip_range = getattr(e, "ip_range", None) or e.get("ip_range")
            if is_range:
                cidr = ip_range or ip
                if cidr:
                    normalized = _normalize_cidr(cidr)
                    if normalized:
                        cidrs_to_add.append(normalized)
                continue
            if not ip:
                continue
            key = _key(ip)
            reason = getattr(e, "reason", None) or e.get("reason")
            val = json.dumps({"reason": reason or "IP is blacklisted"})
            expires_at = getattr(e, "expires_at", None) or e.get("expires_at")
            if expires_at:
                if isinstance(expires_at, str):
                    expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                ttl = max(1, int((expires_at - utc_now()).total_seconds()))
                r.setex(key, ttl, val)
            else:
                r.set(key, val)
            count += 1
        if cidrs_to_add:
            r.sadd(cidr_key, *cidrs_to_add)
            count += len(cidrs_to_add)
        return count
    except RuntimeError:
        raise
    except Exception as e:
        logger.warning(f"Blacklist full sync failed: {e}")
        raise RuntimeError(REDIS_REQUIRED_MSG) from e
