"""
Security Rules API endpoints
"""
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from src.api.database import get_db
from src.api.services.rules_service import RulesService
from src.api.models.security_rules import RuleAction, RulePriority

router = APIRouter()


class SecurityRuleRequest(BaseModel):
    name: str
    rule_type: str
    pattern: str
    applies_to: str = "all"
    action: str = "block"  # block, log, alert, redirect, challenge
    priority: str = "medium"  # high, medium, low
    description: Optional[str] = None
    owasp_category: Optional[str] = None


@router.get("")
async def get_security_rules(
    active_only: bool = Query(True),
    db: Session = Depends(get_db)
):
    """Get security rules"""
    service = RulesService(db)
    rules = service.get_rules(active_only)
    
    return {
        "success": True,
        "data": [rule.to_dict() for rule in rules],
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("")
async def create_security_rule(
    request: SecurityRuleRequest,
    db: Session = Depends(get_db)
):
    """Create security rule"""
    service = RulesService(db)
    
    action = RuleAction[request.action.upper()] if hasattr(RuleAction, request.action.upper()) else RuleAction.BLOCK
    priority = RulePriority[request.priority.upper()] if hasattr(RulePriority, request.priority.upper()) else RulePriority.MEDIUM
    
    rule = service.create_rule(
        name=request.name,
        rule_type=request.rule_type,
        pattern=request.pattern,
        applies_to=request.applies_to,
        action=action,
        priority=priority,
        description=request.description,
        owasp_category=request.owasp_category
    )
    
    return {
        "success": True,
        "data": rule.to_dict(),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/owasp")
async def get_owasp_rules(
    db: Session = Depends(get_db)
):
    """Get OWASP Top 10 rules"""
    service = RulesService(db)
    rules = service.get_owasp_rules()
    
    return {
        "success": True,
        "data": [rule.to_dict() for rule in rules],
        "timestamp": datetime.utcnow().isoformat()
    }
