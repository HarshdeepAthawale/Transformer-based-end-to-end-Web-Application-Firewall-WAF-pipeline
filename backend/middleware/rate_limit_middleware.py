"""
Rate limiting middleware. Throttles by client IP.
"""
import time
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from backend.utils.rate_limiter import PerIPRateLimiter

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


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP rate limiting."""

    def __init__(self, app, max_requests: int = DEFAULT_MAX_REQUESTS, window_seconds: int = DEFAULT_WINDOW_SECONDS):
        super().__init__(app)
        self.limiter = PerIPRateLimiter(max_requests=max_requests, window_seconds=window_seconds)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in SKIP_PATHS:
            return await call_next(request)

        ip = request.client.host if request.client else "unknown"
        if not self.limiter.is_allowed(ip):
            wait = self.limiter.get_wait_time(ip)
            logger.warning(f"Rate limit exceeded for IP {ip}")
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
