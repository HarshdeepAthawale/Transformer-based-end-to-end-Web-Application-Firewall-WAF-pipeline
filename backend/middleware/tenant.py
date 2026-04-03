"""
Tenant Middleware - extracts org_id from JWT into request.state
"""
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger


class TenantMiddleware(BaseHTTPMiddleware):
    """Middleware to extract and enforce tenant (org_id) from JWT"""

    # Public endpoints that don't require auth
    SKIP_PATHS = {
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/api/users/login",
    }

    async def dispatch(self, request: Request, call_next):
        """Extract org_id from JWT and attach to request state"""
        # Skip auth enforcement for public endpoints
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        # Default to org_id = None (will be checked in route if required)
        request.state.org_id = None

        # Try to extract org_id from Authorization header
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                from backend.auth import verify_token
                token = auth_header.split(" ", 1)[1]
                payload = verify_token(token)

                if payload:
                    org_id = payload.get("org_id")
                    if org_id:
                        request.state.org_id = org_id
                    else:
                        # Token exists but missing org_id (backward compatibility)
                        # This can happen with old tokens before org_id was added
                        logger.warning(f"JWT token missing org_id field (user: {payload.get('user_id')})")
            except Exception as e:
                logger.debug(f"Failed to extract org_id from JWT: {e}")

        return await call_next(request)
