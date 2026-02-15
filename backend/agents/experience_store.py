"""Experience store — save/retrieve agent conversation turns with keyword-based retrieval."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.models.agent_experience import AgentExperience


# Simple stop words to exclude from keyword extraction
_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could of in on at to for "
    "with by from and or not no but if then so it its i me my we our "
    "you your they them their this that these those what which who whom "
    "how when where why all each every some any many much more most "
    "show get give tell find check look see please help".split()
)


def _extract_keywords(text: str) -> str:
    """Extract meaningful keywords from text, space-separated."""
    words = re.findall(r"[a-zA-Z0-9_./-]+", text.lower())
    keywords = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
    return " ".join(dict.fromkeys(keywords))  # dedupe preserving order


class ExperienceStore:
    def __init__(self, db: Session) -> None:
        self.db = db

    def save_experience(
        self,
        session_id: str,
        user_id: int | None,
        user_message: str,
        agent_intent: str,
        agent_response: str,
        tools_used: List[str],
        suggested_actions: List[dict],
        steps_taken: int,
    ) -> AgentExperience:
        exp = AgentExperience(
            session_id=session_id,
            user_id=user_id,
            user_message=user_message,
            agent_intent=agent_intent,
            agent_response=agent_response,
            tools_used=json.dumps(tools_used),
            tool_call_count=len(tools_used),
            suggested_actions=json.dumps(suggested_actions),
            keywords=_extract_keywords(user_message),
            steps_taken=steps_taken,
            response_length=len(agent_response),
        )
        self.db.add(exp)
        self.db.commit()
        self.db.refresh(exp)
        return exp

    def update_feedback(
        self, experience_id: int, score: int, text: Optional[str] = None
    ) -> AgentExperience | None:
        exp = self.db.query(AgentExperience).filter(AgentExperience.id == experience_id).first()
        if exp is None:
            return None
        exp.feedback_score = score
        exp.feedback_text = text
        exp.feedback_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(exp)
        return exp

    def retrieve_similar(
        self, message: str, intent: str, limit: int = 3
    ) -> List[AgentExperience]:
        """Keyword LIKE + intent filter, prefer positive feedback."""
        keywords = _extract_keywords(message).split()
        if not keywords:
            return []

        query = self.db.query(AgentExperience).filter(
            AgentExperience.agent_intent == intent
        )

        # Build OR conditions for keyword matching
        from sqlalchemy import or_

        conditions = [AgentExperience.keywords.contains(kw) for kw in keywords[:5]]
        query = query.filter(or_(*conditions))

        # Order: positive feedback first, then most recent
        query = query.order_by(
            desc(AgentExperience.feedback_score),
            desc(AgentExperience.timestamp),
        )

        return query.limit(limit).all()

    def get_session_history(self, session_id: str) -> List[AgentExperience]:
        return (
            self.db.query(AgentExperience)
            .filter(AgentExperience.session_id == session_id)
            .order_by(AgentExperience.timestamp)
            .all()
        )
