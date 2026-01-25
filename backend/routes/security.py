"""Security API endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.controllers import security as ctrl

router = APIRouter()


@router.get("/checks")
async def get_security_checks(db: Session = Depends(get_db)):
    return ctrl.get_checks(db)


@router.post("/checks/{check_id}/run")
async def run_security_check(check_id: int, db: Session = Depends(get_db)):
    return ctrl.run_check(db, check_id)


@router.get("/compliance-score")
async def get_compliance_score(db: Session = Depends(get_db)):
    return ctrl.get_compliance_score(db)
