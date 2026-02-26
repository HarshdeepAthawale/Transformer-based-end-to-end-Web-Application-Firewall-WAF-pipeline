"""User Management API endpoints."""
import time
from collections import defaultdict
from threading import Lock

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models.users import User, UserRole
from backend.auth import get_current_user, require_role
from backend.schemas.users import UserCreateRequest, LoginRequest, ApiKeyCreateRequest
from backend.controllers import users as ctrl

router = APIRouter()

# --- Login rate limiter: 5 attempts per IP per 60 seconds ---
_login_attempts: dict[str, list[float]] = defaultdict(list)
_login_lock = Lock()
LOGIN_MAX_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 60


def _check_login_rate_limit(request: Request):
    """Block login if IP exceeds 5 attempts in 60 seconds."""
    ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
        request.client.host if request.client else "unknown"
    )
    now = time.monotonic()
    with _login_lock:
        attempts = _login_attempts[ip]
        # Prune old entries
        _login_attempts[ip] = [t for t in attempts if now - t < LOGIN_WINDOW_SECONDS]
        if len(_login_attempts[ip]) >= LOGIN_MAX_ATTEMPTS:
            raise HTTPException(
                status_code=429,
                detail="Too many login attempts. Try again later.",
            )
        _login_attempts[ip].append(now)


@router.post("/login")
async def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)):
    _check_login_rate_limit(request)
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
