"""Security rules API request schemas."""
from pydantic import BaseModel
from typing import Optional


class SecurityRuleRequest(BaseModel):
    name: str
    rule_type: str
    pattern: str
    applies_to: str = "all"
    action: str = "block"
    priority: str = "medium"
    description: Optional[str] = None
    owasp_category: Optional[str] = None
