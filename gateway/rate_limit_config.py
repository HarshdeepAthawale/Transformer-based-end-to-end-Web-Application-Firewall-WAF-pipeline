"""
Rate limit config loader with in-process caching.
Polls /api/rate-limits/ from backend with TTL. Fails open on errors.
"""

import time
from typing import Optional

import httpx
from loguru import logger

from gateway.config import gateway_config

# Module-level cache: zone_id -> (configs_list, expires_at_timestamp)
_cache: dict[str, tuple[list, float]] = {}


async def get_rate_limit_configs(zone_id: str = "default") -> list[dict]:
    """
    Fetch rate limit configs for zone_id from backend.
    Cache with TTL. Fail open (return stale cache or empty list) on error.
    """
    now = time.time()
    ttl = gateway_config.RATE_LIMIT_CONFIG_CACHE_TTL_SECONDS

    # Return cache if still fresh
    if zone_id in _cache:
        configs, expires_at = _cache[zone_id]
        if now < expires_at:
            return configs

    # Poll backend
    backend_url = gateway_config.RATE_LIMIT_BACKEND_URL
    if not backend_url:
        return _stale_or_empty(zone_id)

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(
                f"{backend_url}/api/rate-limits/",
                params={"zone_id": zone_id, "active_only": "true"},
            )
            if resp.status_code == 200:
                data = resp.json()
                configs = data.get("data", [])
                _cache[zone_id] = (configs, now + ttl)
                return configs
            else:
                logger.warning(
                    f"Rate limit config fetch returned {resp.status_code}"
                )
    except Exception as e:
        logger.warning(f"Failed to fetch rate limit configs: {e}")

    return _stale_or_empty(zone_id)


def _stale_or_empty(zone_id: str) -> list[dict]:
    """Return stale cache if available, otherwise empty list."""
    if zone_id in _cache:
        return _cache[zone_id][0]
    return []


def get_limit_for_path(configs: list[dict], path: str) -> Optional[dict]:
    """
    Find the most specific (longest prefix) matching config for path.
    Returns None if no config matches.
    """
    matches = [c for c in configs if path.startswith(c.get("path_prefix", ""))]
    if not matches:
        return None
    # Longest prefix wins (most specific match)
    return max(matches, key=lambda c: len(c.get("path_prefix", "")))
