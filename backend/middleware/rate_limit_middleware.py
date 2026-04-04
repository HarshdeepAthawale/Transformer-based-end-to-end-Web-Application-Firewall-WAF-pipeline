"""
Rate limiting middleware. Throttles by client IP with org-aware per-endpoint configuration.
Uses Redis-backed config cache. Falls back to default limits if Redis/DB unavailable.
"""
import os
import time
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

# Skip rate limiting for these paths
SKIP_PATHS = frozenset({
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
})

# Default: 300 req/min per IP
DEFAULT_MAX_REQUESTS = 300
DEFAULT_WINDOW_SECONDS = 60

# Redis-backed config cache: (org_id, path) -> (limit, cache_expires_at)
_config_cache = {}
CACHE_TTL_SECONDS = 30

# Redis-backed rate tracking
_redis_client = None
_redis_checked = False


def _get_redis():
    """Get Redis client for rate tracking. Returns None if unavailable."""
    global _redis_client, _redis_checked
    if _redis_checked and _redis_client is None:
        return None
    if _redis_client is not None:
        return _redis_client
    try:
        import redis
        url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _redis_client = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=2)
        _redis_client.ping()
        _redis_checked = True
        return _redis_client
    except Exception:
        _redis_checked = True
        return None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP rate limiting with Redis tracking and org-aware DB configuration."""

    def __init__(self, app, max_requests: int = DEFAULT_MAX_REQUESTS, window_seconds: int = DEFAULT_WINDOW_SECONDS):
        super().__init__(app)
        self.default_limit = max_requests
        self.window_seconds = window_seconds

    def _get_effective_limit(self, org_id, path: str) -> int:
        """Get effective rate limit with caching. No DB session needed for cache hits."""
        if not org_id:
            return self.default_limit

        cache_key = (org_id, path)
        now = time.time()

        if cache_key in _config_cache:
            limit, expires_at = _config_cache[cache_key]
            if now < expires_at:
                return limit

        # Query DB for config (only on cache miss)
        try:
            from backend.database import SessionLocal
            from backend.models.rate_limit_config import RateLimitConfig
            db = SessionLocal()
            try:
                config = db.query(RateLimitConfig).filter(
                    RateLimitConfig.org_id == org_id,
                    RateLimitConfig.path_prefix == path,
                    RateLimitConfig.is_active
                ).first()
                limit = config.requests_per_minute if config else self.default_limit
                _config_cache[cache_key] = (limit, now + CACHE_TTL_SECONDS)
                return limit
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Error fetching rate limit config: {e}")
            return self.default_limit

    def _is_allowed_redis(self, ip: str, max_requests: int) -> tuple[bool, float]:
        """Check rate limit via Redis sliding window. Returns (allowed, retry_after)."""
        r = _get_redis()
        if r is None:
            return True, 0.0  # Fail open

        key = f"rl:backend:{ip}"
        now = time.time()
        window_start = now - self.window_seconds

        try:
            pipe = r.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zadd(key, {str(now): now})
            pipe.zcard(key)
            pipe.expire(key, self.window_seconds + 10)
            results = pipe.execute()

            count = results[2] if len(results) > 2 else 0

            if count > max_requests:
                oldest = r.zrange(key, 0, 0, withscores=True)
                if oldest:
                    retry_after = max(0, self.window_seconds - (now - oldest[0][1]))
                else:
                    retry_after = self.window_seconds
                return False, retry_after

            return True, 0.0
        except Exception as e:
            logger.warning(f"Redis rate limit error: {e}")
            return True, 0.0  # Fail open

    async def dispatch(self, request: Request, call_next):
        if request.url.path in SKIP_PATHS:
            return await call_next(request)

        org_id = getattr(request.state, "org_id", None)
        ip = request.client.host if request.client else "unknown"
        path = request.url.path

        limit = self._get_effective_limit(org_id, path)

        allowed, wait = self._is_allowed_redis(ip, limit)
        if not allowed:
            logger.warning(f"Rate limit exceeded for IP {ip} on path {path} (limit: {limit} req/min)")
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "message": "Too many requests",
                    "retry_after_seconds": max(1, int(wait)),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                headers={"Retry-After": str(max(1, int(wait)))},
            )
        return await call_next(request)
