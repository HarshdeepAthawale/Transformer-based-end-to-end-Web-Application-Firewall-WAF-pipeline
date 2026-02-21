"""
Firewall-for-AI detection: prompt-injection, PII, and abuse rate limit.
Uses patterns from firewall_ai_patterns (DB + optional URL).
"""

import re
import time
from typing import Optional

from sqlalchemy.orm import Session

from backend.config import config
from backend.services.firewall_ai_patterns import get_prompt_injection_patterns, get_pii_patterns

# In-memory abuse rate limit: key -> count; bucket by minute
_abuse_counts: dict[str, int] = {}
_abuse_cleanup_at: float = 0


def _abuse_key(ip: str) -> str:
    minute_ts = int(time.time()) // 60
    return f"firewall_ai:abuse:{ip}:{minute_ts}"


def check_abuse_rate(ip: str) -> bool:
    """
    Return True if IP is over the abuse rate limit (should block).
    Per-IP per-minute limit from config.
    """
    global _abuse_counts, _abuse_cleanup_at
    limit = max(1, getattr(config, "FIREWALL_AI_ABUSE_RATE_PER_MINUTE", 60))
    key = _abuse_key(ip)
    now = time.time()
    if now - _abuse_cleanup_at > 120:
        # Drop keys older than 2 minutes
        cutoff = int(now) // 60 - 2
        _abuse_counts = {k: v for k, v in _abuse_counts.items() if int(k.split(":")[-1]) >= cutoff}
        _abuse_cleanup_at = now
    _abuse_counts[key] = _abuse_counts.get(key, 0) + 1
    return _abuse_counts[key] > limit


def _match_patterns(text: str, patterns: list[str]) -> Optional[str]:
    """Return first matching pattern (regex or substring)."""
    if not text:
        return None
    text_lower = text.lower()
    for p in patterns:
        if not p:
            continue
        try:
            if re.search(p, text, re.IGNORECASE | re.DOTALL):
                return p
        except re.error:
            if p.lower() in text_lower:
                return p
    return None


def check_prompt_injection(body: str, headers: dict, db: Session) -> tuple[bool, Optional[str]]:
    """
    Run body and optionally headers against prompt-injection patterns.
    Returns (matched, pattern).
    """
    if not getattr(config, "FIREWALL_AI_ENABLED", False):
        return False, None
    patterns = get_prompt_injection_patterns(db)
    if not patterns:
        return False, None
    text = body or ""
    if headers:
        text += " " + " ".join(f"{k}:{v}" for k, v in (headers or {}).items())
    match = _match_patterns(text, patterns)
    return (match is not None, match)


def check_pii(text: str, db: Session) -> tuple[bool, Optional[str]]:
    """Run text against PII patterns. Returns (matched, pattern)."""
    if not getattr(config, "FIREWALL_AI_ENABLED", False):
        return False, None
    patterns = get_pii_patterns(db)
    if not patterns:
        return False, None
    match = _match_patterns(text or "", patterns)
    return (match is not None, match)


def should_block_prompt_match() -> bool:
    """True if config action for prompt match is block."""
    action = (getattr(config, "FIREWALL_AI_ACTION_PROMPT_MATCH", "block") or "block").strip().lower()
    return action == "block"


def should_block_pii() -> bool:
    """True if config action for PII is block."""
    action = (getattr(config, "FIREWALL_AI_ACTION_PII", "log") or "log").strip().lower()
    return action == "block"
