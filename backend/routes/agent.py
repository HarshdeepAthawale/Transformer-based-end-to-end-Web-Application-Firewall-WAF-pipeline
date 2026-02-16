"""Agent API routes — chat, streaming, feedback, action execution, history."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from loguru import logger

from backend.database import get_db

router = APIRouter()


# --- Request/Response schemas ---

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class FeedbackRequest(BaseModel):
    score: int  # 1 or -1
    text: str | None = None


class ActionRequest(BaseModel):
    action: str
    params: dict = {}


# Whitelisted actions that can be executed via the /action/execute endpoint
ALLOWED_ACTIONS = frozenset([
    "block_ip", "unblock_ip", "whitelist_ip",
    "dismiss_alert", "acknowledge_alert",
    "create_security_rule", "create_geo_rule",
])


def _get_orchestrator(request: Request, db: Session):
    from backend.agents.orchestrator import Orchestrator

    waf_service = getattr(request.app.state, "waf_service", None)
    return Orchestrator(db=db, waf_service=waf_service)


# --- Endpoints ---

@router.post("/chat")
async def chat(body: ChatRequest, request: Request, db: Session = Depends(get_db)):
    """Non-streaming chat endpoint."""
    orchestrator = _get_orchestrator(request, db)
    result = await orchestrator.run(body.message, session_id=body.session_id)
    return {
        "success": True,
        "data": result,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request, db: Session = Depends(get_db)):
    """SSE streaming chat endpoint."""
    orchestrator = _get_orchestrator(request, db)

    async def event_generator():
        async for event in orchestrator.run_streaming(body.message, session_id=body.session_id):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/feedback/{experience_id}")
async def submit_feedback(
    experience_id: int, body: FeedbackRequest, db: Session = Depends(get_db)
):
    """Submit thumbs up/down feedback for an agent response."""
    if body.score not in (1, -1):
        raise HTTPException(status_code=400, detail="Score must be 1 or -1")

    from backend.agents.experience_store import ExperienceStore

    store = ExperienceStore(db)
    exp = store.update_feedback(experience_id, body.score, body.text)
    if exp is None:
        raise HTTPException(status_code=404, detail="Experience not found")
    return {
        "success": True,
        "message": "Feedback recorded",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/action/execute")
async def execute_action(
    body: ActionRequest, request: Request, db: Session = Depends(get_db)
):
    """Execute a suggested action (whitelisted action names only)."""
    if body.action not in ALLOWED_ACTIONS:
        raise HTTPException(status_code=400, detail=f"Action '{body.action}' is not allowed")

    from backend.agents.tools.registry import registry
    from backend.agents.context import AgentContext

    tool = registry.get(body.action)
    if tool is None:
        raise HTTPException(status_code=400, detail=f"Tool '{body.action}' not found")

    waf_service = getattr(request.app.state, "waf_service", None)
    ctx = AgentContext(db=db, waf_service=waf_service)

    try:
        result = tool.handler(ctx, **body.params)
    except Exception as e:
        logger.error(f"Action execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "success": True,
        "data": result,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/history")
async def get_history(session_id: str, db: Session = Depends(get_db)):
    """Retrieve conversation history for a session."""
    from backend.agents.experience_store import ExperienceStore

    store = ExperienceStore(db)
    experiences = store.get_session_history(session_id)
    return {
        "success": True,
        "data": [e.to_dict() for e in experiences],
        "timestamp": datetime.utcnow().isoformat(),
    }
