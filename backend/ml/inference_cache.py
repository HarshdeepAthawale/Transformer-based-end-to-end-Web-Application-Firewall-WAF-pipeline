"""
Two-tier inference cache: in-process L1 (dict) + Redis L2.

Architecture:
  Request -> L1 (~0.01ms) -> L2 Redis (~0.2ms) -> Model inference (~3ms)

Graceful degradation: if Redis is unavailable, falls back to L1-only.
"""
import hashlib
import json
import os
import re
import threading
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlencode, urlparse

from loguru import logger

# Headers that carry per-request dynamic tokens (kill cache hit rate)
_STRIP_HEADERS = frozenset({
    "cookie",
    "x-csrf-token",
    "x-request-id",
    "x-correlation-id",
    "x-trace-id",
    "authorization",
    "if-none-match",
    "if-modified-since",
})

# Regex for common timestamp patterns (ISO 8601, unix epoch)
_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.\d]*Z?\b"
    r"|\b1[6-9]\d{8,9}\b"
)

_REDIS_PREFIX = "waf:cache:"


def _safe_json_value(v: Any) -> Any:
    """Convert non-JSON-serializable values (torch tensors, numpy arrays) to plain Python."""
    if hasattr(v, "item"):
        return v.item()
    if hasattr(v, "tolist"):
        return v.tolist()
    return v


def normalize_request_text(text: str) -> str:
    """Normalize request text before hashing to improve cache hit rate.

    Strips dynamic tokens (cookies, CSRF, timestamps) and sorts query params
    so semantically identical requests produce the same cache key.
    """
    lines = text.split("\n")
    normalized = []

    for line in lines:
        lower = line.lower().strip()

        # Strip headers with dynamic per-request values
        header_name = lower.split(":", 1)[0] if ":" in lower else ""
        if header_name in _STRIP_HEADERS:
            continue

        # Normalize timestamps to reduce key cardinality
        line = _TIMESTAMP_RE.sub("__TS__", line)

        normalized.append(line)

    result = "\n".join(normalized)

    # Sort query parameters if a URL line is present
    if normalized and " " in normalized[0]:
        parts = normalized[0].split(" ", 2)
        if len(parts) >= 2:
            parsed = urlparse(parts[1])
            if parsed.query:
                sorted_qs = urlencode(
                    sorted(parse_qs(parsed.query, keep_blank_values=True).items()),
                    doseq=True,
                )
                rebuilt = parsed._replace(query=sorted_qs).geturl()
                parts[1] = rebuilt
                normalized[0] = " ".join(parts)
                result = "\n".join(normalized)

    return result


def cache_key(text: str) -> str:
    """Compute a blake2b cache key from normalized request text."""
    normalized = normalize_request_text(text)
    return hashlib.blake2b(
        normalized.encode("utf-8", errors="replace"), digest_size=16
    ).hexdigest()


class InferenceCache:
    """Two-tier L1 (in-process) + L2 (Redis) inference cache."""

    def __init__(self, l1_maxsize: int = 10000, ttl: int = 300):
        self._l1: Dict[str, Dict[str, Any]] = {}
        self._l1_order: list = []
        self._l1_maxsize = l1_maxsize
        self._l1_lock = threading.Lock()
        self._ttl = ttl

        # Metrics (atomic-enough for counters; reads are approximate)
        self.l1_hits = 0
        self.l1_misses = 0
        self.l2_hits = 0
        self.l2_misses = 0

        # Lazy Redis connection
        self._redis = None
        self._redis_checked = False

    def _get_redis(self):
        """Singleton Redis connection with graceful fallback."""
        if self._redis_checked and self._redis is None:
            return None
        if self._redis is not None:
            return self._redis
        self._redis_checked = True
        try:
            import redis
            url = os.getenv("REDIS_URL", "redis://localhost:6379")
            self._redis = redis.Redis.from_url(
                url, decode_responses=True, socket_connect_timeout=2
            )
            self._redis.ping()
            logger.info("Inference cache: Redis L2 connected")
            return self._redis
        except Exception:
            logger.debug("Inference cache: Redis unavailable, L1-only mode")
            self._redis = None
            return None

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Look up a classification result by cache key. L1 first, then L2."""
        # L1 lookup
        with self._l1_lock:
            if key in self._l1:
                self.l1_hits += 1
                return self._l1[key]
        self.l1_misses += 1

        # L2 lookup (Redis)
        r = self._get_redis()
        if r is not None:
            try:
                raw = r.get(f"{_REDIS_PREFIX}{key}")
                if raw is not None:
                    self.l2_hits += 1
                    result = json.loads(raw)
                    # Promote to L1
                    self._l1_put(key, result)
                    return result
            except Exception as exc:
                logger.debug(f"Redis L2 get failed: {exc}")
        self.l2_misses += 1
        return None

    def put(self, key: str, result: Dict[str, Any]) -> None:
        """Store a classification result in both L1 and L2."""
        self._l1_put(key, result)

        # Write-through to L2 with safe serialization
        r = self._get_redis()
        if r is not None:
            try:
                safe_result = {k: _safe_json_value(v) for k, v in result.items()}
                r.setex(
                    f"{_REDIS_PREFIX}{key}",
                    self._ttl,
                    json.dumps(safe_result),
                )
            except Exception as exc:
                logger.debug(f"Redis L2 put failed: {exc}")

    def _l1_put(self, key: str, result: Dict[str, Any]) -> None:
        """Insert into L1 with LRU eviction (thread-safe)."""
        with self._l1_lock:
            if key in self._l1:
                return
            if len(self._l1_order) >= self._l1_maxsize:
                evict_key = self._l1_order.pop(0)
                self._l1.pop(evict_key, None)
            self._l1[key] = result
            self._l1_order.append(key)

    def metrics(self) -> Dict[str, Any]:
        """Return cache performance metrics."""
        total_l1 = self.l1_hits + self.l1_misses
        total_l2 = self.l2_hits + self.l2_misses
        with self._l1_lock:
            l1_size = len(self._l1)
        return {
            "l1_cache_hits": self.l1_hits,
            "l1_cache_misses": self.l1_misses,
            "l1_cache_hit_ratio": round(self.l1_hits / total_l1 * 100, 1) if total_l1 else 0.0,
            "l1_cache_size": l1_size,
            "l2_cache_hits": self.l2_hits,
            "l2_cache_misses": self.l2_misses,
            "l2_cache_hit_ratio": round(self.l2_hits / total_l2 * 100, 1) if total_l2 else 0.0,
            "redis_connected": self._redis is not None,
        }
