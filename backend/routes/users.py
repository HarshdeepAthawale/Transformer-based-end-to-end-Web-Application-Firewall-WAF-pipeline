"""User Management API endpoints."""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.users import User
from backend.auth import get_current_user, optional_admin
from backend.schemas.users import UserCreateRequest, LoginRequest, ApiKeyCreateRequest
from backend.controllers import users as ctrl

router = APIRouter()


@router.post("/login")
async def login(payload: LoginRequest, db: Session = Depends(get_db)):
    try:
        return ctrl.login(db, payload.username, payload.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.get("")
async def get_users(
    current_user: Optional[User] = Depends(optional_admin()),
    db: Session = Depends(get_db),
):
    return ctrl.get_users(db)


@router.post("")
async def create_user(
    request: UserCreateRequest,
    current_user: Optional[User] = Depends(optional_admin()),
    db: Session = Depends(get_db),
):
    created_by = current_user.username if current_user else "anonymous"
    try:
        return ctrl.create_user(
            db,
            username=request.username,
            email=request.email,
            password=request.password,
            role=request.role,
            full_name=request.full_name,
            created_by=created_by,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    return ctrl.get_me(current_user)


@router.get("/me/keys")
async def list_my_api_keys(current_user: User = Depends(get_current_user)):
    return ctrl.list_api_keys(current_user)


@router.post("/me/keys")
async def create_api_key(
    request: ApiKeyCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return ctrl.create_api_key(db, current_user, name=request.name or "")


@router.delete("/me/keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        return ctrl.revoke_api_key(db, current_user, key_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
