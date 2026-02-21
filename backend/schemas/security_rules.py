"""Security rules API request/response schemas (Feature 9)."""
from pydantic import BaseModel
from typing import Optional, Any


class SecurityRuleRequest(BaseModel):
    name: str
    rule_type: str
    pattern: str
    match_conditions: Optional[dict[str, Any]] = None
    applies_to: str = "all"
    action: str = "block"
    priority: str = "medium"
    description: Optional[str] = None
    owasp_category: Optional[str] = None
    is_active: Optional[bool] = True


class SecurityRuleUpdate(BaseModel):
    name: Optional[str] = None
    rule_type: Optional[str] = None
    pattern: Optional[str] = None
    match_conditions: Optional[dict[str, Any]] = None
    applies_to: Optional[str] = None
    action: Optional[str] = None
    priority: Optional[str] = None
    description: Optional[str] = None
    owasp_category: Optional[str] = None
    is_active: Optional[bool] = None


class SecurityRuleResponse(BaseModel):
    """Single rule as returned by API (matches SecurityRule.to_dict())."""
    id: int
    name: str
    description: Optional[str] = None
    rule_type: str
    pattern: Optional[str] = None
    match_conditions: Optional[dict] = None
    applies_to: str
    action: Optional[str] = None
    priority: Optional[str] = None
    owasp_category: Optional[str] = None
    is_active: bool
    timestamp: Optional[str] = None
    # optional fields from to_dict
    rule_pack_id: Optional[int] = None
    rule_pack_version: Optional[str] = None
    external_id: Optional[str] = None
    is_system_rule: Optional[bool] = None
    created_by: Optional[str] = None
    notes: Optional[str] = None
    match_count: Optional[int] = None
    last_matched: Optional[str] = None
