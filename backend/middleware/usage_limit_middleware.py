"""
Usage limit enforcement middleware.
Checks if org has exceeded their plan's monthly request quota.
Returns 402 Payment Required if quota exceeded.

Uses Redis for fast usage tracking (no per-request DB writes).
"""
import time
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from backend.services.usage_service import increment_usage_redis, is_within_limit_redis

# In-memory cache: org_id -> (within_limit, limit, cache_expires_at)
_limit_cache = {}
CACHE_TTL_SECONDS = 30

SKIP_PATHS = frozenset({
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/api/billing",
})


class UsageLimitMiddleware(BaseHTTPMiddleware):
    """Enforce plan request limits per organization via Redis."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        for skip in SKIP_PATHS:
            if path.startswith(skip):
                return await call_next(request)

        org_id = getattr(request.state, "org_id", None)
        if not org_id:
            return await call_next(request)

        # Check cache first
        now = time.time()
        if org_id in _limit_cache:
            within_limit, limit, expires_at = _limit_cache[org_id]
            if now < expires_at:
                if not within_limit:
                    return self._quota_exceeded_response()
                # Increment via Redis (O(1), no DB)
                increment_usage_redis(org_id)
                return await call_next(request)

        # Cache miss: fetch plan limit from DB, check Redis usage
        try:
            from backend.database import SessionLocal
            from backend.services.usage_service import UsageService
            db = SessionLocal()
            try:
                svc = UsageService(db)
                limit = svc._get_plan_limit(org_id)
                within_limit = is_within_limit_redis(org_id, limit)
                _limit_cache[org_id] = (within_limit, limit, now + CACHE_TTL_SECONDS)

                if not within_limit:
                    logger.warning(f"Org {org_id} exceeded usage quota")
                    return self._quota_exceeded_response()
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Usage limit check failed for org {org_id}: {e}")

        # Increment via Redis (O(1), no DB write)
        increment_usage_redis(org_id)

        return await call_next(request)

    def _quota_exceeded_response(self) -> JSONResponse:
        return JSONResponse(
            status_code=402,
            content={
                "success": False,
                "message": "Monthly request quota exceeded. Please upgrade your plan.",
                "upgrade_url": "/api/billing/plans",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )
