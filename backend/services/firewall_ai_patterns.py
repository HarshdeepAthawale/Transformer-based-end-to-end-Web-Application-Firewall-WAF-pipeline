"""
Load Firewall-for-AI patterns from DB and optional URLs.
Cache in memory; refresh on interval. No hardcoded pattern strings.
"""

import json
import threading
import time
from typing import List, Tuple

import httpx
from loguru import logger
from sqlalchemy.orm import Session

from backend.config import config
from backend.models.firewall_ai_pattern import FirewallAIPattern

# In-memory cache: (prompt_injection_patterns, pii_patterns)
_cached: Tuple[List[str], List[str]] = ([], [])
_cache_updated_at: float = 0
_lock = threading.Lock()

REFRESH_INTERVAL = max(60, getattr(config, "FIREWALL_AI_PATTERNS_REFRESH_SECONDS", 300))


def _load_from_db(db: Session) -> Tuple[List[str], List[str]]:
    """Load active patterns from DB by type."""
    prompt = [
        row.pattern_value
        for row in db.query(FirewallAIPattern.pattern_value)
        .filter(
            FirewallAIPattern.pattern_type == "prompt_injection",
            FirewallAIPattern.is_active.is_(True),
        )
        .all()
    ]
    pii = [
        row.pattern_value
        for row in db.query(FirewallAIPattern.pattern_value)
        .filter(
            FirewallAIPattern.pattern_type == "pii",
            FirewallAIPattern.is_active.is_(True),
        )
        .all()
    ]
    return (prompt, pii)


def _fetch_url(url: str) -> List[str]:
    """Fetch patterns from URL: plain text one per line, or JSON array of strings."""
    if not url or not url.strip():
        return []
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(url.strip())
            r.raise_for_status()
            text = r.text.strip()
            if not text:
                return []
            # Try JSON array first
            if text.startswith("["):
                data = json.loads(text)
                if isinstance(data, list):
                    return [str(x).strip() for x in data if x]
            # Plain text, one per line
            return [line.strip() for line in text.splitlines() if line.strip()]
    except Exception as e:
        logger.debug(f"Firewall AI pattern URL fetch failed: {e}")
        return []
    return []


def get_prompt_injection_patterns(db: Session) -> List[str]:
    """Return cached prompt-injection patterns; refresh if TTL exceeded."""
    global _cached, _cache_updated_at
    with _lock:
        now = time.monotonic()
        if not _cached[0] and not _cached[1] or (now - _cache_updated_at) > REFRESH_INTERVAL:
            prompt_db, pii_db = _load_from_db(db)
            prompt_url = _fetch_url(getattr(config, "FIREWALL_AI_PROMPT_PATTERNS_URL", "") or "")
            pii_url = _fetch_url(getattr(config, "FIREWALL_AI_PII_PATTERNS_URL", "") or "")
            _cached = (
                list(dict.fromkeys(prompt_db + prompt_url)),
                list(dict.fromkeys(pii_db + pii_url)),
            )
            _cache_updated_at = now
        return list(_cached[0])


def get_pii_patterns(db: Session) -> List[str]:
    """Return cached PII patterns; refresh if TTL exceeded."""
    global _cached, _cache_updated_at
    with _lock:
        now = time.monotonic()
        if not _cached[0] and not _cached[1] or (now - _cache_updated_at) > REFRESH_INTERVAL:
            prompt_db, pii_db = _load_from_db(db)
            prompt_url = _fetch_url(getattr(config, "FIREWALL_AI_PROMPT_PATTERNS_URL", "") or "")
            pii_url = _fetch_url(getattr(config, "FIREWALL_AI_PII_PATTERNS_URL", "") or "")
            _cached = (
                list(dict.fromkeys(prompt_db + prompt_url)),
                list(dict.fromkeys(pii_db + pii_url)),
            )
            _cache_updated_at = now
        return list(_cached[1])


def invalidate_cache() -> None:
    """Force next get to refetch from DB/URL."""
    global _cached, _cache_updated_at
    with _lock:
        _cached = ([], [])
        _cache_updated_at = 0
