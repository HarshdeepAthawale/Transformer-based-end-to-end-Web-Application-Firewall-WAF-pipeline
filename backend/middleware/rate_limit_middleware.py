"""
Rate limiting middleware. Throttles by client IP with org-aware per-endpoint configuration.
"""
import time
from collections import defaultdict
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

# Module-level cache: (org_id, path) -> (limit, cache_expires_at)
_config_cache = {}
CACHE_TTL_SECONDS = 30

# Per-(org_id, path, ip) request tracking: (org_id, path, ip) -> list of timestamps
_request_tracker = defaultdict(list)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP rate limiting with org-aware database configuration."""

    def __init__(self, app, max_requests: int = DEFAULT_MAX_REQUESTS, window_seconds: int = DEFAULT_WINDOW_SECONDS):
        super().__init__(app)
        self.default_limit = max_requests
        self.window_seconds = window_seconds

    def _get_effective_limit(self, org_id, path: str, db) -> int:
        """Get effective rate limit with caching. Returns limit from DB or default."""
        if not org_id:
            return self.default_limit

        cache_key = (org_id, path)
        now = time.time()

        # Check cache
        if cache_key in _config_cache:
            limit, expires_at = _config_cache[cache_key]
            if now < expires_at:
                return limit

        # Query DB for config
        try:
            from backend.models.rate_limit_config import RateLimitConfig
            config = db.query(RateLimitConfig).filter(
                RateLimitConfig.org_id == org_id,
                RateLimitConfig.path_prefix == path,
                RateLimitConfig.is_active == True
            ).first()

            limit = config.requests_per_minute if config else self.default_limit
            _config_cache[cache_key] = (limit, now + CACHE_TTL_SECONDS)
            return limit
        except Exception as e:
            logger.warning(f"Error fetching rate limit config for org {org_id} path {path}: {e}")
            return self.default_limit

    def _is_allowed(self, org_id, path: str, ip: str, max_requests: int) -> bool:
        """Check if request is allowed using dynamic per-config limits."""
        now = time.time()
        tracker_key = (org_id, path, ip)

        # Clean old requests outside window
        _request_tracker[tracker_key] = [
            ts for ts in _request_tracker[tracker_key]
            if now - ts < self.window_seconds
        ]

        # Check limit
        if len(_request_tracker[tracker_key]) >= max_requests:
            return False

        # Record this request
        _request_tracker[tracker_key].append(now)
        return True

    def _get_wait_time(self, org_id, path: str, ip: str) -> float:
        """Get time to wait before next request is allowed."""
        now = time.time()
        tracker_key = (org_id, path, ip)

        if tracker_key not in _request_tracker or not _request_tracker[tracker_key]:
            return 0.0

        oldest = min(_request_tracker[tracker_key])
        wait = self.window_seconds - (now - oldest)
        return max(0.0, wait)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in SKIP_PATHS:
            return await call_next(request)

        org_id = getattr(request.state, "org_id", None)
        ip = request.client.host if request.client else "unknown"
        path = request.url.path

        # Get effective limit (org-scoped or default)
        limit = self.default_limit
        if org_id:
            try:
                from backend.database import SessionLocal
                db = SessionLocal()
                try:
                    limit = self._get_effective_limit(org_id, path, db)
                finally:
                    db.close()
            except Exception as e:
                logger.warning(f"Could not fetch org rate limit, using default: {e}")

        # Check rate limit with effective limit
        if not self._is_allowed(org_id, path, ip, limit):
            wait = self._get_wait_time(org_id, path, ip)
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
