"""Settings API: account preferences and retention."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Any, Dict

from backend.database import get_db
from backend.controllers import settings as ctrl

router = APIRouter()


@router.get("")
async def get_settings(db: Session = Depends(get_db)):
    """Get all account settings (with defaults for missing keys)."""
    data = ctrl.get_settings(db)
    return {
        "success": True,
        "data": data,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
    }


@router.put("")
async def update_settings(
    payload: Dict[str, Any],
    db: Session = Depends(get_db),
):
    """Update one or more settings. Only allowed keys are persisted."""
    data = ctrl.update_settings(db, payload)
    return {
        "success": True,
        "data": data,
        "message": "Settings updated",
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
    }


@router.get("/retention")
async def get_retention():
    """Get data retention days (read-only from server config)."""
    data = ctrl.get_retention()
    return {
        "success": True,
        "data": data,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
    }
