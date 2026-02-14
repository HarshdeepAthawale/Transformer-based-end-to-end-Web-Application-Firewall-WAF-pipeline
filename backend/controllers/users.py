"""Users controller."""
import hashlib
import json
import secrets
import uuid
from datetime import datetime
from sqlalchemy.orm import Session

from backend.models.users import User, UserRole
from backend.auth import create_access_token

API_KEY_PREFIX = "waf_"


def _get_keys_list(user: User) -> list:
    if not user.api_keys:
        return []
    try:
        return json.loads(user.api_keys)
    except (TypeError, json.JSONDecodeError):
        return []


def login(db: Session, username: str, password: str) -> dict:
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.check_password(password):
        raise ValueError("Invalid credentials")
    if not user.is_active:
        raise PermissionError("User account is inactive")
    user.last_login = datetime.utcnow()
    db.commit()
    token = create_access_token(user.id, user.username, user.role)
    return {
        "success": True,
        "data": {"token": token, "user": user.to_dict()},
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_users(db: Session) -> dict:
    users = db.query(User).all()
    return {
        "success": True,
        "data": [u.to_dict() for u in users],
        "timestamp": datetime.utcnow().isoformat(),
    }


def create_user(
    db: Session,
    *,
    username: str,
    email: str,
    password: str,
    role: str = "viewer",
    full_name: str | None = None,
    created_by: str,
) -> dict:
    if db.query(User).filter(User.username == username).first():
        raise ValueError("Username already exists")
    if db.query(User).filter(User.email == email).first():
        raise ValueError("Email already exists")
    r = UserRole[role.upper()] if hasattr(UserRole, role.upper()) else UserRole.VIEWER
    user = User(username=username, email=email, role=r, full_name=full_name, created_by=created_by)
    user.set_password(password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"success": True, "data": user.to_dict(), "timestamp": datetime.utcnow().isoformat()}


def get_me(user: User) -> dict:
    return {
        "success": True,
        "data": user.to_dict(),
        "timestamp": datetime.utcnow().isoformat(),
    }


def list_api_keys(user: User) -> dict:
    keys_list = _get_keys_list(user)
    out = []
    for k in keys_list:
        if isinstance(k, dict):
            out.append({
                "id": k.get("id"),
                "name": k.get("name") or "Unnamed",
                "prefix": k.get("prefix", "waf_***"),
                "created_at": k.get("created_at"),
            })
    return {
        "success": True,
        "data": out,
        "timestamp": datetime.utcnow().isoformat(),
    }


def create_api_key(db: Session, user: User, name: str = "") -> dict:
    raw_key = API_KEY_PREFIX + secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_id = str(uuid.uuid4())
    prefix = raw_key[:12] + "…"
    created_at = datetime.utcnow().isoformat()
    keys_list = _get_keys_list(user)
    keys_list.append({
        "id": key_id,
        "name": name or "Unnamed",
        "key_hash": key_hash,
        "prefix": prefix,
        "created_at": created_at,
    })
    user.api_keys = json.dumps(keys_list)
    db.commit()
    return {
        "success": True,
        "data": {
            "key": raw_key,
            "id": key_id,
            "name": name or "Unnamed",
            "created_at": created_at,
        },
        "message": "Copy the key now; it will not be shown again.",
        "timestamp": datetime.utcnow().isoformat(),
    }


def revoke_api_key(db: Session, user: User, key_id: str) -> dict:
    keys_list = _get_keys_list(user)
    new_list = [k for k in keys_list if isinstance(k, dict) and k.get("id") != key_id]
    if len(new_list) == len(keys_list):
        raise ValueError("API key not found")
    user.api_keys = json.dumps(new_list)
    db.commit()
    return {
        "success": True,
        "data": {"revoked": key_id},
        "timestamp": datetime.utcnow().isoformat(),
    }
