"""Toxic Combinations API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.auth import get_current_tenant
from backend.services.toxic_combination_service import ToxicCombinationService

router = APIRouter()
_service = ToxicCombinationService()


@router.get("")
async def list_toxic_combinations(
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """List active toxic combinations for the current org."""
    return {"success": True, "data": _service.get_active(db, org_id)}


@router.get("/stats")
async def get_toxic_combination_stats(
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Get summary statistics for toxic combinations."""
    return {"success": True, "data": _service.get_stats(db, org_id)}


@router.get("/{combination_id}")
async def get_toxic_combination(
    combination_id: int,
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Get a specific toxic combination by ID."""
    from backend.models.toxic_combination import ToxicCombination
    tc = (
        db.query(ToxicCombination)
        .filter(ToxicCombination.id == combination_id, ToxicCombination.org_id == org_id)
        .first()
    )
    if not tc:
        raise HTTPException(status_code=404, detail="Toxic combination not found")
    return {"success": True, "data": tc.to_dict()}


class StatusUpdate(BaseModel):
    status: str  # active, investigating, resolved, dismissed


@router.patch("/{combination_id}")
async def update_toxic_combination_status(
    combination_id: int,
    body: StatusUpdate,
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Update the status of a toxic combination."""
    if body.status not in ("active", "investigating", "resolved", "dismissed"):
        raise HTTPException(status_code=400, detail="Invalid status")

    result = _service.update_status(db, org_id, combination_id, body.status)
    if not result:
        raise HTTPException(status_code=404, detail="Toxic combination not found")
    return {"success": True, "data": result}


@router.post("/evaluate")
async def trigger_evaluation(
    window_minutes: int = Query(5, ge=1, le=60),
    org_id: int = Depends(get_current_tenant),
    db: Session = Depends(get_db),
):
    """Manually trigger toxic combination evaluation for the current window."""
    detections = _service.evaluate_window(db, org_id, window_minutes)
    return {
        "success": True,
        "data": {
            "detections_count": len(detections),
            "detections": detections,
        },
    }
