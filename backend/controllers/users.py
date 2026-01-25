"""Users controller."""
from datetime import datetime
from sqlalchemy.orm import Session

from backend.models.users import User, UserRole
from backend.auth import create_access_token


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
