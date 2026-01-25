"""Users API request schemas."""
from pydantic import BaseModel
from typing import Optional


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
