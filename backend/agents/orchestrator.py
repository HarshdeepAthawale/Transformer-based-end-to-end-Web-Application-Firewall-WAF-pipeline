"""Orchestrator — main entry point for the agent pipeline."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, AsyncIterator

from loguru import logger
from sqlalchemy.orm import Session

from backend.agents.context import AgentContext
from backend.agents.experience_store import ExperienceStore
from backend.agents.llm_client import get_llm_client, get_model
from backend.agents.router import AgentIntent, classify_intent
from backend.agents.specialists.analyst import create_analyst
from backend.agents.specialists.explainer import create_explainer
from backend.agents.specialists.forensics import create_forensics
from backend.agents.specialists.investigator import create_investigator
from backend.agents.specialists.remediation import (
    create_remediation,
    extract_suggested_actions,
)

# Ensure all tools are registered on import
import backend.agents.tools.investigation_tools  # noqa: F401
import backend.agents.tools.remediation_tools  # noqa: F401
import backend.agents.tools.analytics_tools  # noqa: F401
import backend.agents.tools.forensics_tools  # noqa: F401
import backend.agents.tools.explainer_tools  # noqa: F401

AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT_SECONDS", "30"))

_SPECIALIST_FACTORIES = {
    AgentIntent.INVESTIGATE: create_investigator,
    AgentIntent.ANALYZE: create_analyst,
    AgentIntent.EXPLAIN: create_explainer,
    AgentIntent.FORENSICS: create_forensics,
    AgentIntent.REMEDIATE: create_remediation,
}


class Orchestrator:
    def __init__(self, db: Session, waf_service: Any = None, user_id: int | None = None) -> None:
        self.db = db
        self.waf_service = waf_service
        self.user_id = user_id
        self.client = get_llm_client()
        self.model = get_model()
        self.store = ExperienceStore(db)

    def _build_context(self, session_id: str) -> AgentContext:
        return AgentContext(
            db=self.db,
            waf_service=self.waf_service,
            user_id=self.user_id,
            session_id=session_id,
        )

    def _enrich_message(self, message: str, similar: list) -> str:
        """Add past experience context to the user message."""
        if not similar:
            return message
        context_lines = []
        for exp in similar[:2]:
            snippet = exp.agent_response[:300]
            context_lines.append(
                f"- Previous Q: \"{exp.user_message[:100]}\" → Key info: {snippet}"
            )
        context_block = "\n".join(context_lines)
        return (
            f"{message}\n\n"
            f"[Context from previous conversations]\n{context_block}"
        )

    async def run(self, message: str, session_id: str | None = None) -> dict:
        """Non-streaming agent execution."""
        session_id = session_id or str(uuid.uuid4())
        intent = classify_intent(message)
        ctx = self._build_context(session_id)

        # Retrieve similar past experiences
        similar = self.store.retrieve_similar(message, intent.value)
        enriched = self._enrich_message(message, similar)

        # Create specialist
        factory = _SPECIALIST_FACTORIES[intent]
        agent = factory(self.client, self.model)

        try:
            result = await asyncio.wait_for(
                agent.run(enriched, ctx), timeout=AGENT_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning(f"Agent timed out after {AGENT_TIMEOUT}s for intent={intent.value}")
            result = {
                "content": "I'm sorry, the request took too long. Please try a simpler query.",
                "tool_calls_made": [],
                "steps": 0,
            }
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            result = {
                "content": f"An error occurred while processing your request: {str(e)}",
                "tool_calls_made": [],
                "steps": 0,
            }

        content = result["content"]
        suggested_actions = []

        if intent == AgentIntent.REMEDIATE:
            content, suggested_actions = extract_suggested_actions(content)

        # Save experience
        exp = self.store.save_experience(
            session_id=session_id,
            user_id=self.user_id,
            user_message=message,
            agent_intent=intent.value,
            agent_response=content,
            tools_used=result["tool_calls_made"],
            suggested_actions=suggested_actions,
            steps_taken=result["steps"],
        )

        return {
            "content": content,
            "intent": intent.value,
            "experience_id": exp.id,
            "session_id": session_id,
            "suggested_actions": suggested_actions,
            "tools_used": result["tool_calls_made"],
        }

    async def run_streaming(self, message: str, session_id: str | None = None) -> AsyncIterator[str]:
        """Streaming agent execution — yields SSE-formatted events."""
        session_id = session_id or str(uuid.uuid4())
        intent = classify_intent(message)
        ctx = self._build_context(session_id)

        # Yield intent event
        yield f"data: {json.dumps({'type': 'intent', 'intent': intent.value})}\n\n"

        # Retrieve similar past experiences
        similar = self.store.retrieve_similar(message, intent.value)
        enriched = self._enrich_message(message, similar)

        factory = _SPECIALIST_FACTORIES[intent]
        agent = factory(self.client, self.model)

        full_content = ""
        tool_calls_made = []

        try:
            async for chunk in agent.run_streaming(enriched, ctx):
                if chunk.startswith("__TOOL_USE__:"):
                    tool_name = chunk.split(":", 1)[1]
                    tool_calls_made.append(tool_name)
                    yield f"data: {json.dumps({'type': 'tool_use', 'tool': tool_name})}\n\n"
                else:
                    full_content += chunk
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
        except asyncio.TimeoutError:
            error_msg = "Request timed out. Please try a simpler query."
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
            full_content = error_msg
        except Exception as e:
            logger.error(f"Streaming agent error: {e}", exc_info=True)
            error_msg = f"An error occurred: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
            full_content = error_msg

        # Extract suggested actions if remediation
        suggested_actions = []
        if intent == AgentIntent.REMEDIATE:
            full_content, suggested_actions = extract_suggested_actions(full_content)

        # Save experience
        exp = self.store.save_experience(
            session_id=session_id,
            user_id=self.user_id,
            user_message=message,
            agent_intent=intent.value,
            agent_response=full_content,
            tools_used=tool_calls_made,
            suggested_actions=suggested_actions,
            steps_taken=len(tool_calls_made),
        )

        # Yield done event
        yield f"data: {json.dumps({'type': 'done', 'experience_id': exp.id, 'session_id': session_id, 'suggested_actions': suggested_actions, 'intent': intent.value})}\n\n"
