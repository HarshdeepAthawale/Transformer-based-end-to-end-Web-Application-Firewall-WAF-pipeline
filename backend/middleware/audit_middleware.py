"""
Audit Logging Middleware
"""
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime
from loguru import logger

from backend.database import SessionLocal
from backend.models.audit_log import AuditLog, AuditAction

# Path prefix -> resource_type for audit logs
_PATH_RESOURCE_MAP = [
    ("/api/ip", "ip"),
    ("/api/rules", "rule"),
    ("/api/users", "user"),
    ("/api/geo", "geo_rule"),
    ("/api/audit", "audit_log"),
    ("/api/bots", "bot_signature"),
    ("/api/threat-intel", "threat_intel"),
    ("/api/waf", "waf_config"),
]


def _resource_type_for_path(path: str) -> str:
    for prefix, resource_type in _PATH_RESOURCE_MAP:
        if path.startswith(prefix):
            return resource_type
    return "unknown"


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware for audit logging"""

    async def dispatch(self, request: Request, call_next):
        """Process request and log audit events"""
        if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        username = None
        user_id = None
        try:
            from backend.auth import verify_token
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ", 1)[1]
                payload = verify_token(token)
                if payload:
                    username = payload.get("username")
                    user_id = payload.get("user_id")
        except Exception:
            pass

        ip_address = request.client.host if request.client else None

        response = await call_next(request)

        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            db = None
            try:
                db = SessionLocal()
                action = AuditAction.UPDATE
                if request.method == "POST":
                    action = AuditAction.CREATE
                elif request.method == "DELETE":
                    action = AuditAction.DELETE

                resource_type = _resource_type_for_path(request.url.path)
                log = AuditLog(
                    user_id=user_id,
                    username=username,
                    ip_address=ip_address,
                    action=action,
                    resource_type=resource_type,
                    description=f"{request.method} {request.url.path}",
                    success=response.status_code < 400,
                )
                db.add(log)
                db.commit()
            except Exception as e:
                logger.error(f"Failed to create audit log: {e}")
            finally:
                if db is not None:
                    db.close()

        return response
