"""
Managed rules: fetch enabled packs from backend, cache, evaluate requests.
No hardcoded patterns; all from backend API.
"""
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from gateway.config import gateway_config

# In-memory cache: list of packs with rules; refreshed on TTL
_cached_packs: List[Dict[str, Any]] = []
_cache_updated_at: float = 0


def _fetch_from_backend() -> List[Dict[str, Any]]:
    """GET enabled packs and rules from backend."""
    url = (gateway_config.MANAGED_RULES_BACKEND_URL or "").rstrip("/")
    if not url:
        return []
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{url}/api/rules/managed?enabled_only=true")
            r.raise_for_status()
            data = r.json()
            return data.get("packs") or []
    except Exception as e:
        logger.debug(f"Managed rules fetch failed: {e}")
        return []


def get_cached_packs() -> List[Dict[str, Any]]:
    """Return cached packs; refresh if TTL exceeded."""
    global _cached_packs, _cache_updated_at
    ttl = max(60, gateway_config.MANAGED_RULES_CACHE_TTL_SECONDS)
    if not _cached_packs or (time.monotonic() - _cache_updated_at) > ttl:
        _cached_packs = _fetch_from_backend()
        _cache_updated_at = time.monotonic()
    return _cached_packs


def invalidate_cache() -> None:
    """Force next get_cached_packs() to refetch."""
    global _cached_packs, _cache_updated_at
    _cached_packs = []
    _cache_updated_at = 0


def _rule_matches(
    rule: Dict[str, Any],
    method: str,
    path: str,
    headers: Dict[str, str],
    query_string: str,
    body: str,
) -> bool:
    """Check if a single rule matches the request."""
    pattern = rule.get("pattern")
    if not pattern:
        return False
    applies_to = (rule.get("applies_to") or "all").lower()
    if applies_to == "all":
        text = f"{method} {path} {query_string} {str(headers)} {body}"
    elif applies_to == "path":
        text = path
    elif applies_to == "query":
        text = query_string
    elif applies_to == "headers":
        text = str(headers)
    elif applies_to == "body":
        text = body
    else:
        text = f"{method} {path} {query_string} {str(headers)} {body}"
    try:
        return bool(re.search(pattern, text, re.IGNORECASE))
    except re.error:
        return False


def evaluate(
    method: str,
    path: str,
    headers: Dict[str, str],
    query_string: str,
    body: str,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate request against cached managed rules. Returns first match or None.
    Match: { "rule_id": int, "rule_name": str, "action": str, "pack_id": str }
    """
    if not gateway_config.MANAGED_RULES_ENABLED:
        return None
    packs = get_cached_packs()
    if not packs:
        return None
    for pack in packs:
        for rule in pack.get("rules") or []:
            if _rule_matches(rule, method, path, headers, query_string, body):
                return {
                    "rule_id": rule.get("id"),
                    "rule_name": rule.get("name"),
                    "action": (rule.get("action") or "block").lower(),
                    "pack_id": pack.get("pack_id"),
                }
    return None
