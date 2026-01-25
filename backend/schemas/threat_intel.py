"""Threat intelligence API request schemas."""
from pydantic import BaseModel
from typing import Optional


class ThreatIntelRequest(BaseModel):
    threat_type: str
    value: str
    severity: str
    category: str
    source: str
    description: Optional[str] = None
    expires_at: Optional[str] = None
