"""
Usage limit enforcement middleware.
Checks if org has exceeded their plan's monthly request quota.
Returns 402 Payment Required if quota exceeded.
"""
import time
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

# In-memory cache: org_id -> (within_limit, cache_expires_at)
_limit_cache = {}
CACHE_TTL_SECONDS = 10  # Check DB every 10 seconds per org

SKIP_PATHS = frozenset({
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/api/billing",  # Billing routes always accessible
})


class UsageLimitMiddleware(BaseHTTPMiddleware):
    """Enforce plan request limits per organization."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip paths that should always be accessible
        for skip in SKIP_PATHS:
            if path.startswith(skip):
                return await call_next(request)

        org_id = getattr(request.state, "org_id", None)
        if not org_id:
            return await call_next(request)

        # Check cache first
        now = time.time()
        if org_id in _limit_cache:
            within_limit, expires_at = _limit_cache[org_id]
            if now < expires_at:
                if not within_limit:
                    return self._quota_exceeded_response()
                return await call_next(request)

        # Check DB
        try:
            from backend.database import SessionLocal
            from backend.services.usage_service import UsageService
            db = SessionLocal()
            try:
                svc = UsageService(db)
                within_limit = svc.is_within_limit(org_id)
                _limit_cache[org_id] = (within_limit, now + CACHE_TTL_SECONDS)

                if not within_limit:
                    logger.warning(f"Org {org_id} exceeded usage quota")
                    return self._quota_exceeded_response()

                # Increment usage counter
                svc.increment_usage(org_id)
            finally:
                db.close()
        except Exception as e:
            # Fail open: allow request if usage check fails
            logger.warning(f"Usage limit check failed for org {org_id}: {e}")

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
