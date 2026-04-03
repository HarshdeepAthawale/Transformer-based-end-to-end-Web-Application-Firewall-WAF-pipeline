"""
Authentication and Authorization
"""
from datetime import timedelta
from backend.lib.datetime_utils import utc_now
from typing import Optional, Union
import os
import jwt
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.users import User, UserRole
from backend.config import config

# JWT settings (use config so API_AUTH can share)
JWT_SECRET = getattr(config, "JWT_SECRET", None)
if not JWT_SECRET:
    if os.getenv("ENV", "").lower() == "production":
        raise SystemExit("CRITICAL: JWT_SECRET must be set in production")
    JWT_SECRET = "dev-only-insecure-secret-change-in-prod"
JWT_ALGORITHM = "HS256"
JWT_EXPIRY = getattr(config, 'JWT_EXPIRY', 3600)  # 1 hour

security = HTTPBearer()
optional_security = HTTPBearer(auto_error=False)


def create_access_token(user_id: int, username: str, role: UserRole, org_id: int) -> str:
    """Create JWT access token with org_id for multi-tenancy"""
    payload = {
        "user_id": user_id,
        "username": username,
        "role": role.value,
        "org_id": org_id,
        "exp": utc_now() + timedelta(seconds=JWT_EXPIRY),
        "iat": utc_now()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> Optional[dict]:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get current authenticated user"""
    token = credentials.credentials
    payload = verify_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    user = db.query(User).filter(User.id == payload.get("user_id")).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user if token provided; otherwise return None (allow unauthenticated access)"""
    if not credentials:
        return None
    payload = verify_token(credentials.credentials)
    if not payload:
        return None
    user = db.query(User).filter(User.id == payload.get("user_id")).first()
    if not user or not user.is_active:
        return None
    return user


async def get_current_tenant(current_user: User = Depends(get_current_user)) -> int:
    """Get organization ID from authenticated user — used for multi-tenancy enforcement"""
    if not current_user.org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no organization assigned"
        )
    return current_user.org_id


def require_role(required_role: UserRole):
    """Dependency to require specific role"""
    async def role_checker(current_user: User = Depends(get_current_user)):
        user_role = UserRole(current_user.role)
        if user_role != required_role and user_role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires {required_role.value} role"
            )
        return current_user
    return role_checker


def optional_admin():
    """Dependency: returns User if admin token present, else None. Allows unauthenticated access for dev."""
    async def checker(current_user: Optional[User] = Depends(get_current_user_optional)) -> Optional[User]:
        if current_user is None:
            return None
        user_role = UserRole(current_user.role)
        if user_role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Requires admin role"
            )
        return current_user
    return checker


def require_any_role(*roles: UserRole):
    """Dependency to require any of the specified roles"""
    async def role_checker(current_user: User = Depends(get_current_user)):
        user_role = UserRole(current_user.role)
        if user_role not in roles and user_role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of: {[r.value for r in roles]}"
            )
        return current_user
    return role_checker


# --- WAF API auth (Feature 9): JWT or API key for mutating endpoints ---

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_valid_api_keys() -> set:
    """Parse API_KEYS config into a set of valid keys."""
    raw = getattr(config, "API_KEYS", "") or ""
    return {k.strip() for k in raw.split(",") if k.strip()}


async def get_waf_api_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_security),
    api_key: Optional[str] = Depends(_api_key_header),
    db: Session = Depends(get_db),
) -> Union[User, str, None]:
    """
    Dependency for WAF API: accept JWT (Bearer) or API key.
    Returns User if JWT valid, or the string 'api_key' if API key valid, else None.
    """
    header_name = getattr(config, "API_KEY_HEADER", "X-API-Key") or "X-API-Key"
    key_from_header = request.headers.get(header_name) or api_key
    if key_from_header:
        valid_keys = _get_valid_api_keys()
        if valid_keys and key_from_header in valid_keys:
            return "api_key"
    if credentials:
        payload = verify_token(credentials.credentials)
        if payload:
            user = db.query(User).filter(User.id == payload.get("user_id")).first()
            if user and user.is_active:
                return user
    return None


async def require_waf_api_auth(
    auth_result: Union[User, str, None] = Depends(get_waf_api_user),
) -> Union[User, str]:
    """Require auth for WAF API mutating endpoints when API_AUTH_REQUIRED is True."""
    if not getattr(config, "API_AUTH_REQUIRED", True):
        return auth_result or "skip"  # type: ignore
    if auth_result is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required (Bearer token or API key)",
        )
    return auth_result
