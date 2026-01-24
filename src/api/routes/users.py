"""
User Management API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from src.api.database import get_db
from src.api.models.users import User, UserRole
from src.api.auth import get_current_user, require_role, create_access_token

router = APIRouter()


class UserCreateRequest(BaseModel):
    username: str
    email: str
    password: str
    role: str = "viewer"
    full_name: Optional[str] = None


class UserUpdateRequest(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
async def login(
    request: LoginRequest,
    db: Session = Depends(get_db)
):
    """User login"""
    user = db.query(User).filter(User.username == request.username).first()
    
    if not user or not user.check_password(request.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not user.is_active:
        raise HTTPException(status_code=403, detail="User account is inactive")
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Create token
    token = create_access_token(user.id, user.username, user.role)
    
    return {
        "success": True,
        "data": {
            "token": token,
            "user": user.to_dict()
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("")
async def get_users(
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: Session = Depends(get_db)
):
    """Get all users"""
    users = db.query(User).all()
    
    return {
        "success": True,
        "data": [user.to_dict() for user in users],
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("")
async def create_user(
    request: UserCreateRequest,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: Session = Depends(get_db)
):
    """Create new user"""
    # Check if username exists
    if db.query(User).filter(User.username == request.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Check if email exists
    if db.query(User).filter(User.email == request.email).first():
        raise HTTPException(status_code=400, detail="Email already exists")
    
    # Create user
    role = UserRole[request.role.upper()] if hasattr(UserRole, request.role.upper()) else UserRole.VIEWER
    
    user = User(
        username=request.username,
        email=request.email,
        role=role,
        full_name=request.full_name,
        created_by=current_user.username
    )
    user.set_password(request.password)
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return {
        "success": True,
        "data": user.to_dict(),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/me")
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user information"""
    return {
        "success": True,
        "data": current_user.to_dict(),
        "timestamp": datetime.utcnow().isoformat()
    }
