"""User Management API endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.users import User, UserRole
from backend.auth import get_current_user, require_role
from backend.schemas.users import UserCreateRequest, LoginRequest
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
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: Session = Depends(get_db),
):
    return ctrl.get_users(db)


@router.post("")
async def create_user(
    request: UserCreateRequest,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: Session = Depends(get_db),
):
    try:
        return ctrl.create_user(
            db,
            username=request.username,
            email=request.email,
            password=request.password,
            role=request.role,
            full_name=request.full_name,
            created_by=current_user.username,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    return ctrl.get_me(current_user)
