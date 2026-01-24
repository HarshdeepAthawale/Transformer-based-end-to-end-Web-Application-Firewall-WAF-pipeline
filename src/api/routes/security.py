"""
Security API endpoints
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime

from src.api.database import get_db
from src.api.services.security_service import SecurityService

router = APIRouter()


@router.get("/checks")
async def get_security_checks(db: Session = Depends(get_db)):
    """Get security checks"""
    service = SecurityService(db)
    checks = service.get_security_checks()
    
    return {
        "success": True,
        "data": checks,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/checks/{check_id}/run")
async def run_security_check(check_id: int, db: Session = Depends(get_db)):
    """Run a specific security check"""
    service = SecurityService(db)
    check = service.run_security_check(check_id)
    
    if check:
        return {
            "success": True,
            "data": check,
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        return {
            "success": False,
            "message": "Security check not found",
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/compliance-score")
async def get_compliance_score(db: Session = Depends(get_db)):
    """Get compliance score"""
    service = SecurityService(db)
    score = service.get_compliance_score()
    
    return {
        "success": True,
        "data": score,
        "timestamp": datetime.utcnow().isoformat()
    }
