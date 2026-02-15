"""Agent execution context — passed to every tool handler."""

from dataclasses import dataclass
from sqlalchemy.orm import Session


@dataclass
class AgentContext:
    db: Session
    waf_service: object  # optional WAF service from app.state
    user_id: int | None = None
    session_id: str | None = None
