"""
Authentication and Authorization
"""
from datetime import datetime, timedelta
from typing import Optional
import jwt
import secrets
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from loguru import logger

from backend.database import get_db
from backend.models.users import User, UserRole
from backend.config import config

# JWT settings
JWT_SECRET = getattr(config, 'JWT_SECRET', secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRY = getattr(config, 'JWT_EXPIRY', 3600)  # 1 hour

security = HTTPBearer()


def create_access_token(user_id: int, username: str, role: UserRole) -> str:
    """Create JWT access token"""
    payload = {
        "user_id": user_id,
        "username": username,
        "role": role.value,
        "exp": datetime.utcnow() + timedelta(seconds=JWT_EXPIRY),
        "iat": datetime.utcnow()
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
