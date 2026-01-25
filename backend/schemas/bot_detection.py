"""Bot detection API request schemas."""
from pydantic import BaseModel


class BotSignatureRequest(BaseModel):
    user_agent_pattern: str
    name: str
    category: str
    action: str = "block"
    is_whitelisted: bool = False
