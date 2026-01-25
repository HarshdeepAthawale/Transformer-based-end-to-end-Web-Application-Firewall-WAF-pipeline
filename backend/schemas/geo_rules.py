"""Geo rules API request schemas."""
from pydantic import BaseModel
from typing import Optional, List


class GeoRuleRequest(BaseModel):
    rule_type: str
    country_code: str
    country_name: str
    priority: int = 0
    exception_ips: Optional[List[str]] = None
    reason: Optional[str] = None
