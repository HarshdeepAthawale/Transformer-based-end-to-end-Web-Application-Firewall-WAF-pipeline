"""Security controller."""
from datetime import datetime
from sqlalchemy.orm import Session

from backend.services.security_service import SecurityService


def get_checks(db: Session) -> dict:
    service = SecurityService(db)
    checks = service.get_security_checks()
    return {"success": True, "data": checks, "timestamp": datetime.utcnow().isoformat()}


def run_check(db: Session, check_id: int) -> dict:
    service = SecurityService(db)
    check = service.run_security_check(check_id)
    if check:
        return {"success": True, "data": check, "timestamp": datetime.utcnow().isoformat()}
    return {
        "success": False,
        "message": "Security check not found",
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_compliance_score(db: Session) -> dict:
    service = SecurityService(db)
    score = service.get_compliance_score()
    return {"success": True, "data": score, "timestamp": datetime.utcnow().isoformat()}
