"""Security Rules API endpoints."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas.security_rules import SecurityRuleRequest
from backend.controllers import security_rules as ctrl

router = APIRouter()


@router.get("/")
async def get_security_rules(active_only: bool = Query(True), db: Session = Depends(get_db)):
    return ctrl.get_rules(db, active_only)


@router.post("/")
async def create_security_rule(request: SecurityRuleRequest, db: Session = Depends(get_db)):
    return ctrl.create_rule(
        db,
        name=request.name,
        rule_type=request.rule_type,
        pattern=request.pattern,
        applies_to=request.applies_to,
        action=request.action,
        priority=request.priority,
        description=request.description,
        owasp_category=request.owasp_category,
    )


@router.get("/owasp")
async def get_owasp_rules(db: Session = Depends(get_db)):
    return ctrl.get_owasp_rules(db)
