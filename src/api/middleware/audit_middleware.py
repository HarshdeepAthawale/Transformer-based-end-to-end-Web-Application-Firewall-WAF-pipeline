"""
Audit Logging Middleware
"""
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
from loguru import logger

from src.api.database import SessionLocal
from src.api.models.audit_log import AuditLog, AuditAction


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware for audit logging"""
    
    async def dispatch(self, request: Request, call_next):
        """Process request and log audit events"""
        # Skip audit for health checks and static files
        if request.url.path in ['/health', '/docs', '/openapi.json']:
            return await call_next(request)
        
        # Get user info from token if available
        username = None
        user_id = None
        try:
            from src.api.auth import verify_token
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                payload = verify_token(token)
                if payload:
                    username = payload.get("username")
                    user_id = payload.get("user_id")
        except Exception:
            pass
        
        # Get client IP
        ip_address = request.client.host if request.client else None
        
        # Process request
        start_time = datetime.utcnow()
        response = await call_next(request)
        end_time = datetime.utcnow()
        
        # Log audit event for important actions
        if request.method in ['POST', 'PUT', 'DELETE', 'PATCH']:
            try:
                db = SessionLocal()
                try:
                    # Determine action type
                    action = AuditAction.UPDATE
                    if request.method == 'POST':
                        action = AuditAction.CREATE
                    elif request.method == 'DELETE':
                        action = AuditAction.DELETE
                    
                    # Determine resource type from path
                    resource_type = "unknown"
                    if '/ip/' in request.url.path:
                        resource_type = "ip"
                    elif '/rules' in request.url.path:
                        resource_type = "rule"
                    elif '/users' in request.url.path:
                        resource_type = "user"
                    elif '/geo' in request.url.path:
                        resource_type = "geo_rule"
                    
                    audit_log = AuditLog(
                        user_id=user_id,
                        username=username,
                        ip_address=ip_address,
                        action=action,
                        resource_type=resource_type,
                        description=f"{request.method} {request.url.path}",
                        success=response.status_code < 400
                    )
                    
                    db.add(audit_log)
                    db.commit()
                finally:
                    db.close()
            except Exception as e:
                logger.error(f"Failed to create audit log: {e}")
        
        return response
