"""Security rules controller (Feature 9 API)."""
from backend.lib.datetime_utils import utc_now
from sqlalchemy.orm import Session

from backend.services.rules_service import RulesService
from backend.models.security_rules import RuleAction, RulePriority


def get_rules(
    db: Session,
    org_id: int,
    active_only: bool = True,
    pack_id: int | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> dict:
    service = RulesService(db)
    rules, total = service.get_rules(org_id, active_only=active_only, pack_id=pack_id, limit=limit, offset=offset)
    return {
        "success": True,
        "data": [r.to_dict() for r in rules],
        "total": total,
        "timestamp": utc_now().isoformat(),
    }


def get_rule_by_id(db: Session, org_id: int, rule_id: int) -> dict | None:
    service = RulesService(db)
    rule = service.get_rule_by_id(org_id, rule_id)
    return {"success": True, "data": rule.to_dict()} if rule else None


def create_rule(
    db: Session,
    org_id: int,
    *,
    name: str,
    rule_type: str,
    pattern: str,
    applies_to: str = "all",
    action: str = "block",
    priority: str = "medium",
    description: str | None = None,
    owasp_category: str | None = None,
    match_conditions: dict | None = None,
    is_active: bool = True,
) -> dict:
    service = RulesService(db)
    act = RuleAction[action.upper()] if hasattr(RuleAction, action.upper()) else RuleAction.BLOCK
    prio = RulePriority[priority.upper()] if hasattr(RulePriority, priority.upper()) else RulePriority.MEDIUM
    rule = service.create_rule(
        org_id,
        name=name,
        rule_type=rule_type,
        pattern=pattern,
        applies_to=applies_to,
        action=act,
        priority=prio,
        description=description,
        owasp_category=owasp_category,
        match_conditions=match_conditions,
        is_active=is_active,
    )
    return {"success": True, "data": rule.to_dict(), "timestamp": utc_now().isoformat()}


def update_rule(db: Session, org_id: int, rule_id: int, **kwargs) -> dict | None:
    service = RulesService(db)
    rule = service.update_rule(org_id, rule_id, **kwargs)
    return {"success": True, "data": rule.to_dict(), "timestamp": utc_now().isoformat()} if rule else None


def delete_rule(db: Session, org_id: int, rule_id: int) -> bool:
    service = RulesService(db)
    return service.delete_rule(org_id, rule_id)


def get_owasp_rules(db: Session, org_id: int) -> dict:
    service = RulesService(db)
    rules = service.get_owasp_rules(org_id)
    return {
        "success": True,
        "data": [r.to_dict() for r in rules],
        "timestamp": utc_now().isoformat(),
    }
