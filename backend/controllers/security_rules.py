"""Security rules controller."""
from datetime import datetime
from sqlalchemy.orm import Session

from backend.services.rules_service import RulesService
from backend.models.security_rules import RuleAction, RulePriority


def get_rules(db: Session, active_only: bool) -> dict:
    service = RulesService(db)
    rules = service.get_rules(active_only)
    return {
        "success": True,
        "data": [r.to_dict() for r in rules],
        "timestamp": datetime.utcnow().isoformat(),
    }


def create_rule(
    db: Session,
    *,
    name: str,
    rule_type: str,
    pattern: str,
    applies_to: str = "all",
    action: str = "block",
    priority: str = "medium",
    description: str | None = None,
    owasp_category: str | None = None,
) -> dict:
    service = RulesService(db)
    act = RuleAction[action.upper()] if hasattr(RuleAction, action.upper()) else RuleAction.BLOCK
    prio = RulePriority[priority.upper()] if hasattr(RulePriority, priority.upper()) else RulePriority.MEDIUM
    rule = service.create_rule(
        name=name,
        rule_type=rule_type,
        pattern=pattern,
        applies_to=applies_to,
        action=act,
        priority=prio,
        description=description,
        owasp_category=owasp_category,
    )
    return {"success": True, "data": rule.to_dict(), "timestamp": datetime.utcnow().isoformat()}


def get_owasp_rules(db: Session) -> dict:
    service = RulesService(db)
    rules = service.get_owasp_rules()
    return {
        "success": True,
        "data": [r.to_dict() for r in rules],
        "timestamp": datetime.utcnow().isoformat(),
    }
