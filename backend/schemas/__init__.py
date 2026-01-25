"""
Pydantic request/response schemas for API. Per-domain modules.
"""
from backend.schemas.waf import CheckRequest, CheckResponse
from backend.schemas.security_rules import SecurityRuleRequest
from backend.schemas.ip_management import IPBlacklistRequest, IPWhitelistRequest
from backend.schemas.users import UserCreateRequest, UserUpdateRequest, LoginRequest
from backend.schemas.bot_detection import BotSignatureRequest
from backend.schemas.geo_rules import GeoRuleRequest
from backend.schemas.threat_intel import ThreatIntelRequest

__all__ = [
    "CheckRequest",
    "CheckResponse",
    "SecurityRuleRequest",
    "IPBlacklistRequest",
    "IPWhitelistRequest",
    "UserCreateRequest",
    "UserUpdateRequest",
    "LoginRequest",
    "BotSignatureRequest",
    "GeoRuleRequest",
    "ThreatIntelRequest",
]
