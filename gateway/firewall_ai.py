"""
Firewall for AI: match request to LLM endpoints (cached from backend),
call backend evaluate; if block return 403 and report event.
"""

import time
from typing import Any, Optional

import httpx
from loguru import logger

from gateway.config import gateway_config

# Cache: list of endpoint dicts; path_pattern, methods, label
_cached_endpoints: list[dict[str, Any]] = []
_cache_updated_at: float = 0


def _fetch_endpoints() -> list[dict[str, Any]]:
    """GET active LLM endpoints from backend."""
    url = (gateway_config.FIREWALL_AI_BACKEND_URL or "").rstrip("/")
    if not url:
        return []
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{url}/api/firewall-ai/endpoints?active_only=true")
            r.raise_for_status()
            data = r.json()
            return data.get("data") or []
    except Exception as e:
        logger.debug(f"Firewall AI endpoints fetch failed: {e}")
        return []


def _get_cached_endpoints() -> list[dict[str, Any]]:
    """Return cached endpoints; refresh if TTL exceeded."""
    global _cached_endpoints, _cache_updated_at
    ttl = max(30, gateway_config.FIREWALL_AI_CACHE_TTL_SECONDS)
    if not _cached_endpoints or (time.monotonic() - _cache_updated_at) > ttl:
        _cached_endpoints = _fetch_endpoints()
        _cache_updated_at = time.monotonic()
    return _cached_endpoints


def _match_path_method(path: str, method: str) -> Optional[dict[str, Any]]:
    """Return first endpoint that matches path and method."""
    import re
    endpoints = _get_cached_endpoints()
    method_upper = (method or "").upper()
    for ep in endpoints:
        methods_str = ep.get("methods") or "POST"
        methods_list = [m.strip().upper() for m in str(methods_str).split(",") if m.strip()]
        if methods_list and method_upper not in methods_list:
            continue
        pattern = (ep.get("path_pattern") or "").strip()
        if not pattern:
            continue
        if pattern.startswith("^") or "(" in pattern:
            try:
                if re.search(pattern, path):
                    return ep
            except re.error:
                continue
        if path.startswith(pattern) or path == pattern.rstrip("/"):
            return ep
    return None


async def evaluate_request(
    path: str,
    method: str,
    body: bytes,
    headers: dict,
    client_ip: str,
) -> tuple[bool, Optional[str], Optional[str]]:
    """
    If path matches an LLM endpoint: POST to backend evaluate.
    Returns (should_block, event_type, pattern).
    event_type: firewall_ai_prompt_block | firewall_ai_pii | firewall_ai_abuse_rate
    """
    if not gateway_config.FIREWALL_AI_ENABLED:
        return False, None, None

    if not _match_path_method(path, method):
        return False, None, None

    url = (gateway_config.FIREWALL_AI_BACKEND_URL or "").rstrip("/")
    if not url:
        if gateway_config.FIREWALL_AI_FAIL_OPEN:
            return False, None, None
        return True, "firewall_ai_unavailable", None

    timeout = gateway_config.FIREWALL_AI_TIMEOUT
    body_str = (body or b"").decode("utf-8", errors="replace")
    headers_flat = {k: str(v) for k, v in (headers or {}).items()}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{url}/api/firewall-ai/evaluate",
                json={
                    "path": path,
                    "method": method,
                    "body": body_str,
                    "headers": headers_flat,
                    "client_ip": client_ip,
                },
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"Firewall AI evaluate failed: {e}")
        if gateway_config.FIREWALL_AI_FAIL_OPEN:
            return False, None, None
        return True, "firewall_ai_error", None

    if not data.get("applicable"):
        return False, None, None

    if not data.get("block"):
        return False, data.get("reason"), None

    reason = data.get("reason") or "prompt_injection"
    pattern = data.get("pattern")
    event_type = "firewall_ai_prompt_block"
    if reason == "pii":
        event_type = "firewall_ai_pii"
    elif reason == "abuse_rate":
        event_type = "firewall_ai_abuse_rate"
    return True, event_type, pattern
