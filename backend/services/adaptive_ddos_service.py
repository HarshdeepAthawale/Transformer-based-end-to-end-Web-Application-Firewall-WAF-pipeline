"""
Adaptive DDoS: compute baseline from Redis request counts per IP,
derive burst threshold (percentile * multiplier, clamped), write to Redis.
"""

import json
import time
from typing import Optional

from loguru import logger

from backend.config import config


def _get_redis():
    """Sync Redis client for backend job."""
    try:
        import redis
        url = getattr(config, "REDIS_URL", None) or __import__("os").environ.get("REDIS_URL", "redis://localhost:6379")
        return redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=5)
    except Exception as e:
        logger.warning(f"Adaptive DDoS: Redis unavailable: {e}")
        return None


def _adaptive_key_prefix() -> str:
    """Key prefix for gateway-written counters: ddos:adaptive:{minute}:{ip}."""
    return "ddos:adaptive:"


def compute_and_publish_threshold() -> Optional[dict]:
    """
    Read request counts from Redis (keys ddos:adaptive:{minute}:{ip}),
    aggregate per-IP over learning window, compute percentile baseline,
    threshold = clamp(round(baseline * multiplier), min, max).
    Write threshold and meta to Redis. Return stats dict or None on failure.
    """
    r = _get_redis()
    if not r:
        return None

    window_min = getattr(config, "ADAPTIVE_DDOS_LEARNING_WINDOW_MINUTES", 60) or 60
    percentile = getattr(config, "ADAPTIVE_DDOS_PERCENTILE", 95) or 95
    multiplier = getattr(config, "ADAPTIVE_DDOS_MULTIPLIER", 1.5) or 1.5
    thresh_min = max(1, getattr(config, "ADAPTIVE_DDOS_THRESHOLD_MIN", 20))
    thresh_max = max(thresh_min, getattr(config, "ADAPTIVE_DDOS_THRESHOLD_MAX", 500))
    key_threshold = getattr(config, "ADAPTIVE_DDOS_REDIS_KEY", "waf:ddos:adaptive_threshold")
    key_meta = getattr(config, "ADAPTIVE_DDOS_REDIS_KEY_META", "waf:ddos:adaptive_meta")

    now_ts = int(time.time())
    window_start_minute = (now_ts // 60) - window_min
    prefix = _adaptive_key_prefix()

    try:
        # Scan keys ddos:adaptive:*
        per_ip_total: dict[str, int] = {}
        for key in r.scan_iter(match=f"{prefix}*", count=5000):
            try:
                # Key format: ddos:adaptive:{minute}:{ip}  (ip may contain : for IPv6)
                parts = key.split(":", 3)  # ddos, adaptive, minute, ip
                if len(parts) < 4:
                    continue
                minute_bucket = int(parts[2])
                if minute_bucket < window_start_minute:
                    continue
                ip = parts[3]
                count = r.get(key)
                cnt = int(count) if count else 0
                per_ip_total[ip] = per_ip_total.get(ip, 0) + cnt
            except (ValueError, IndexError):
                continue

        if not per_ip_total:
            # No data: do not overwrite existing threshold with zero
            existing = r.get(key_threshold)
            if existing is not None:
                try:
                    cur = int(existing)
                    meta_str = r.get(key_meta)
                    meta = json.loads(meta_str) if meta_str else {}
                    return {
                        "current_threshold": cur,
                        "baseline_percentile_value": meta.get("baseline_percentile_value"),
                        "last_updated": meta.get("updated_at"),
                        "learning_window_minutes": window_min,
                        "config": {
                            "multiplier": multiplier,
                            "threshold_min": thresh_min,
                            "threshold_max": thresh_max,
                            "percentile": percentile,
                        },
                        "message": "No traffic data; keeping existing threshold",
                    }
                except (ValueError, TypeError):
                    pass
            return {
                "current_threshold": None,
                "baseline_percentile_value": None,
                "last_updated": None,
                "learning_window_minutes": window_min,
                "config": {"multiplier": multiplier, "threshold_min": thresh_min, "threshold_max": thresh_max, "percentile": percentile},
                "message": "No traffic data yet",
            }

        # Percentile of per-IP request counts (e.g. P95 = 95% of IPs have rate <= this value)
        counts = sorted(per_ip_total.values())
        idx = min(len(counts) - 1, max(0, int((percentile / 100.0) * len(counts))))
        baseline_rate = counts[idx] if counts else 0

        # Threshold = baseline * multiplier, clamped
        raw = baseline_rate * multiplier
        threshold = max(thresh_min, min(thresh_max, int(round(raw))))

        r.set(key_threshold, str(threshold))
        meta = {
            "baseline_percentile_value": baseline_rate,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts)),
        }
        r.set(key_meta, json.dumps(meta))
        logger.info(f"Adaptive DDoS: threshold={threshold} (baseline P{percentile}={baseline_rate})")
        return {
            "current_threshold": threshold,
            "baseline_percentile_value": baseline_rate,
            "last_updated": meta["updated_at"],
            "learning_window_minutes": window_min,
            "config": {
                "multiplier": multiplier,
                "threshold_min": thresh_min,
                "threshold_max": thresh_max,
                "percentile": percentile,
            },
        }
    except Exception as e:
        logger.warning(f"Adaptive DDoS compute error: {e}")
        return None


def get_adaptive_stats() -> dict:
    """Read current threshold and meta from Redis for API."""
    r = _get_redis()
    if not r:
        return {
            "enabled": getattr(config, "ADAPTIVE_DDOS_ENABLED", False),
            "current_threshold": None,
            "baseline_percentile_value": None,
            "last_updated": None,
            "learning_window_minutes": getattr(config, "ADAPTIVE_DDOS_LEARNING_WINDOW_MINUTES", 60),
            "config": {
                "multiplier": getattr(config, "ADAPTIVE_DDOS_MULTIPLIER", 1.5),
                "threshold_min": getattr(config, "ADAPTIVE_DDOS_THRESHOLD_MIN", 20),
                "threshold_max": getattr(config, "ADAPTIVE_DDOS_THRESHOLD_MAX", 500),
                "percentile": getattr(config, "ADAPTIVE_DDOS_PERCENTILE", 95),
            },
        }

    key_threshold = getattr(config, "ADAPTIVE_DDOS_REDIS_KEY", "waf:ddos:adaptive_threshold")
    key_meta = getattr(config, "ADAPTIVE_DDOS_REDIS_KEY_META", "waf:ddos:adaptive_meta")
    try:
        val = r.get(key_threshold)
        current_threshold = int(val) if val else None
        meta_str = r.get(key_meta)
        meta = json.loads(meta_str) if meta_str else {}
        return {
            "enabled": getattr(config, "ADAPTIVE_DDOS_ENABLED", False),
            "current_threshold": current_threshold,
            "baseline_percentile_value": meta.get("baseline_percentile_value"),
            "last_updated": meta.get("updated_at"),
            "learning_window_minutes": getattr(config, "ADAPTIVE_DDOS_LEARNING_WINDOW_MINUTES", 60),
            "config": {
                "multiplier": getattr(config, "ADAPTIVE_DDOS_MULTIPLIER", 1.5),
                "threshold_min": getattr(config, "ADAPTIVE_DDOS_THRESHOLD_MIN", 20),
                "threshold_max": getattr(config, "ADAPTIVE_DDOS_THRESHOLD_MAX", 500),
                "percentile": getattr(config, "ADAPTIVE_DDOS_PERCENTILE", 95),
            },
        }
    except (ValueError, TypeError, json.JSONDecodeError):
        return {
            "enabled": getattr(config, "ADAPTIVE_DDOS_ENABLED", False),
            "current_threshold": None,
            "baseline_percentile_value": None,
            "last_updated": None,
            "learning_window_minutes": getattr(config, "ADAPTIVE_DDOS_LEARNING_WINDOW_MINUTES", 60),
            "config": {
                "multiplier": getattr(config, "ADAPTIVE_DDOS_MULTIPLIER", 1.5),
                "threshold_min": getattr(config, "ADAPTIVE_DDOS_THRESHOLD_MIN", 20),
                "threshold_max": getattr(config, "ADAPTIVE_DDOS_THRESHOLD_MAX", 500),
                "percentile": getattr(config, "ADAPTIVE_DDOS_PERCENTILE", 95),
            },
        }
