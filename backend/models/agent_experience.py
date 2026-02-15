"""Agent experience store — SQLAlchemy model for conversation turns and feedback."""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Index, Integer, String, Text
from backend.database import Base


class AgentExperience(Base):
    __tablename__ = "agent_experiences"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    session_id = Column(String(64), index=True, nullable=False)
    user_id = Column(Integer, nullable=True)

    user_message = Column(Text, nullable=False)
    agent_intent = Column(String(32), nullable=False, index=True)
    agent_response = Column(Text, nullable=False)

    tools_used = Column(Text, default="[]")  # JSON list
    tool_call_count = Column(Integer, default=0)
    suggested_actions = Column(Text, default="[]")  # JSON list

    feedback_score = Column(Integer, nullable=True)  # 1 or -1
    feedback_text = Column(Text, nullable=True)
    feedback_at = Column(DateTime, nullable=True)

    keywords = Column(Text, default="")  # space-separated for LIKE search
    steps_taken = Column(Integer, default=0)
    response_length = Column(Integer, default=0)

    __table_args__ = (
        Index("ix_agent_exp_session_ts", "session_id", "timestamp"),
        Index("ix_agent_exp_intent_score", "agent_intent", "feedback_score"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "user_message": self.user_message,
            "agent_intent": self.agent_intent,
            "agent_response": self.agent_response,
            "tools_used": self.tools_used,
            "tool_call_count": self.tool_call_count,
            "suggested_actions": self.suggested_actions,
            "feedback_score": self.feedback_score,
            "feedback_text": self.feedback_text,
            "steps_taken": self.steps_taken,
            "response_length": self.response_length,
        }
