"""IP management API request schemas."""
from pydantic import BaseModel
from typing import Optional


class IPBlacklistRequest(BaseModel):
    ip: str
    reason: Optional[str] = None
    duration_hours: Optional[int] = None
    source: str = "manual"


class IPWhitelistRequest(BaseModel):
    ip: str
    reason: Optional[str] = None
